"""Build the Qwen3.8-Flash-Next NVFP4 artifact from the official BF16 checkpoint.

Unlike ``qwen3_8_27b`` there is no official NVFP4 checkpoint to repack: the
converter reads the single official BF16 source and quantizes every NVFP4 / FP8
object on the fly.  The PLE n-gram embedding and its three I64 config tensors
are deliberately excluded from this ``.ninfer`` artifact; they live in a
separate ``.ngram`` store (``ngram_store``).

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.convert_nvfp4 \
      --model Z:/ninfer_flash_next_tmp/flash_next_bf16 \
      --out out/qwen3_8_flash_next.ninfer \
      --device cuda

A full plan validation without streaming weights (uses the safetensors header
table captured during research)::

    python -m tools.convert.qwen3_8_flash_next.convert_nvfp4 \
      --model <dir> --dry-run \
      --shape-table .logs/research_qfn/tensor_shapes.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
    encoded_size,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import inventory as family_inventory
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe
from . import shape_table
from .artifact_writer import ResumableArtifactWriter


RECIPE_ID = "qwen3_8_flash_next_nvfp4-v1"
OUTPUT_BASENAME = "qwen3_8_flash_next.ninfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preflight_inventory() -> None:
    """Weight-free structural checks on the inventory + recipe."""

    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or not shard
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path}: invalid weight_map entry")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(
            f"{model_dir}: safetensors shard set does not match the index "
            f"({len(actual)} on disk vs {len(referenced)} indexed)"
        )
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def _config_summary(config: Mapping[str, object]) -> dict[str, object]:
    tc = config["text_config"]
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": tc.get("hidden_size"),
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "num_experts": tc.get("num_experts"),
        "num_experts_per_tok": tc.get("num_experts_per_tok"),
        "moe_intermediate_size": tc.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size"),
        "vocab_size": tc.get("vocab_size"),
        "full_attention_interval": tc.get("full_attention_interval"),
        "partial_rotary_factor": tc.get("partial_rotary_factor"),
        "tie_word_embeddings": tc.get("tie_word_embeddings"),
        "max_position_embeddings": tc.get("max_position_embeddings"),
    }


def _validate_official_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("config.json must be an object")
    if config.get("quantization_config") is not None:
        raise ValueError(
            "official source must not declare quantization_config "
            "(this target self-quantizes from BF16)"
        )
    cfg.validate_config(dict(config))
    return _config_summary(config)


def preflight_conversion(
    model_dir: str | Path,
) -> ConversionPreflight:
    """Validate the single BF16 source end to end and plan the artifact.

    Opens every indexed shard header (not the weight bytes) to verify the
    recipe's 1527 source tensors exist with the declared shape and dtype.
    """

    model = Path(model_dir)
    _validate_index(model)
    config_summary = _validate_official_config(
        family_conversion.load_json(model / "config.json")
    )
    preflight_inventory()
    with ShardReader(model) as reader:
        source = recipe.preflight_source_reader(reader)
    resources = family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        object_plan=object_plan,
    )


def _encode_payload(
    spec: family_inventory.StoredObjectSpec,
    reader: ShardReader,
    resources: Mapping[str, bytes],
    divisor: torch.Tensor,
    device: torch.device,
) -> bytes:
    """Encode one planned object from its recipe (single BF16 source)."""

    if isinstance(spec, family_inventory.ResourceSpec):
        return resources[spec.name]
    if spec.name in recipe.DIVISOR_NAMES:
        # Placeholder activation scale (BLOCKER: runtime kernel required to
        # calibrate); the divisor spec is a 0-dim FP32 scalar.
        return encode_direct(divisor, spec.format)
    if spec.format == inventory.NVFP4:
        tensor = recipe.materialize(spec, reader, device)
        qz = recipe.quantize_nvfp4_object(spec, tensor)
        del tensor
        return encode_nvfp4(
            qz.packed_codes, qz.natural_scales, qz.weight_scale_divisor, spec.shape
        )
    if spec.format == inventory.FP8:
        tensor = recipe.materialize(spec, reader, device)
        qz = recipe.quantize_fp8_object(spec, tensor)
        del tensor
        return encode_fp8_row_scaled(qz.code_words, qz.row_scales, spec.shape)
    tensor = recipe.materialize(spec, reader, device)
    payload = family_conversion.encode_tensor_payload(tensor, spec, device)
    del tensor
    return payload


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "arguments": dict(arguments),
        "config_summary": preflight.config_summary,
        "source": {
            "model_path": str(preflight.model_dir.resolve()),
            "self_quantized": True,
        },
        "source_preflight": {
            "recipes": preflight.source.recipe_count,
            "tensors": preflight.source.source_tensor_count,
            "shards": preflight.source.source_shard_count,
            "dtypes": dict(preflight.source.source_dtype_counts),
        },
        "objects": family_conversion.object_statistics(objects),
        "elapsed_seconds": elapsed_seconds,
        "artifact": {"path": str(output), "bytes": final_bytes},
        "device": str(device),
    }


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    overwrite: bool = False,
) -> Path:
    """Run the single-source self-quantizing conversion; return the report path.

    Resumable: an interrupted run re-opens the artifact, verifies the on-disk
    directory still matches the recomputed plan, and skips the completed
    objects (``--overwrite`` discards the partial file and restarts).
    """

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"converter output basename must be {OUTPUT_BASENAME!r}")
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(inventory.NVFP4_TENSOR_SPECS)} NVFP4 + "
        f"{len(inventory.FP8_TENSOR_SPECS)} FP8 + "
        f"{len(recipe.DIVISOR_NAMES)} divisor objects, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    divisor = torch.tensor(cfg.INPUT_SCALE_DIVISOR_PLACEHOLDER, dtype=torch.float32)

    with ShardReader(preflight.model_dir) as reader, ResumableArtifactWriter(
        output,
        identity,
        preflight.object_plan.specs,
        overwrite=overwrite,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer object plan differs from completed preflight")
        total = len(inventory.OBJECT_SPECS)
        skip = writer.completed
        if skip:
            print(
                f"resuming: {skip}/{total} objects already complete",
                flush=True,
            )
        for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
            if index <= skip:
                continue
            payload = _encode_payload(spec, reader, resources, divisor, resolved_device)
            writer.write(spec.name, payload)
            del payload
            print(f"[{index}/{total}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": str(device),
        "overwrite": overwrite,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def dry_run(
    model_dir: str | Path,
    shape_table_path: str | Path,
) -> dict[str, object]:
    """Validate the full artifact plan without streaming any weights.

    Checks the config, the inventory/recipe, the frontend resources, and the
    recipe's source coverage against the safetensors *header* table (real
    geometry, no bytes).  Returns a report describing the planned objects.
    """

    model = Path(model_dir)
    config_summary = _validate_official_config(
        family_conversion.load_json(model / "config.json")
    )
    preflight_inventory()
    reader = shape_table.ShapeTableReader(shape_table_path)
    source = recipe.preflight_source_reader(reader)
    resources = family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)

    tensor_specs = [
        spec for spec in inventory.OBJECT_SPECS
        if isinstance(spec, family_inventory.TensorSpec)
    ]
    tensor_bytes = sum(
        encoded_size(spec.layout, spec.format, spec.shape) for spec in tensor_specs
    )
    report = {
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "dry_run": True,
        "config_summary": config_summary,
        "source_preflight": {
            "recipes": source.recipe_count,
            "tensors": source.source_tensor_count,
            "shards": source.source_shard_count,
            "dtypes": dict(source.source_dtype_counts),
        },
        "objects": family_conversion.object_statistics(object_plan.objects),
        "tensor_payload_bytes": tensor_bytes,
        "nvfp4_objects": len(inventory.NVFP4_TENSOR_SPECS),
        "fp8_objects": len(inventory.FP8_TENSOR_SPECS),
        "divisor_objects": len(recipe.DIVISOR_NAMES),
        "resource_bytes": sum(len(data) for data in resource_map.values()),
    }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="artifact path (full run only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the plan against the header table without writing",
    )
    parser.add_argument(
        "--shape-table",
        type=Path,
        help="tensor_shapes.json (required with --dry-run)",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        if args.shape_table is None:
            parser.error("--dry-run requires --shape-table <tensor_shapes.json>")
        report = dry_run(args.model, args.shape_table)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.out is None:
        parser.error("--out is required (or use --dry-run)")
    convert(args.model, args.out, device=args.device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

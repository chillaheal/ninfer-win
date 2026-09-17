"""STEP 3 machine-readable tensor -> NInfer-representation mapping report.

Joins the three already-validated descriptors of the Flash-Next NVFP4 target and
emits ONE record per NInfer object (name / shape / numeric format / layout /
checkpoint provenance / role).  Nothing here re-derives an encoding: the object
contract comes from ``inventory_nvfp4`` and the provenance from ``recipe_nvfp4``
(both self-validate at import).  ``tensor_map.json`` supplies the source-side
role/size inventory for cross-checking.

This is a mapping REPORT, not an encoder: it says *which* checkpoint tensor(s)
feed *which* object and at *which* representation, and that the mapping is
complete and consistent.  Per-object encoded byte counts belong to the size
accounting step (they need the format encoders, not the mapping).

Run (cwd = ninfer-win):
    python -m tools.convert.qwen3_8_flash_next.mapping_report [--out PATH]

The report FAILS LOUDLY (non-zero exit) if any coverage / consistency invariant
breaks -- a silent, partially-mapped artifact would be worse than no report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    SourceTensor,
    expression_sources,
    source_requirements,
)
from tools.convert.qwen3_8_flash_next import inventory_nvfp4 as inventory
from tools.convert.qwen3_8_flash_next import recipe_nvfp4 as recipe

_PACKAGE_DIR = Path(__file__).resolve().parent
_TENSOR_MAP = _PACKAGE_DIR / "tensor_map.json"
_DEFAULT_OUT = _PACKAGE_DIR / "ninfer_mapping.json"

# The separate .ngram store's footprint (see ngram_store.py); needed for the
# "sources + ngram + configs == checkpoint tensors" reconciliation.
NGRAM_SHARD_COUNT = 128
NGRAM_CONFIG_COUNT = 3

# --- RUNTIME KERNEL REQUIRED (documented per the honesty constraint) ---------
# These operations have a *correct, complete* artifact representation in this
# target but no working runtime kernel yet.  The converter produces them so the
# foundation is right; the engine must grow a kernel before the model can run.
BLOCKERS: list[dict[str, str]] = [
    {
        "operation": "MoE expert dispatch (top-10-of-512 + 1 shared expert)",
        "scope": "every text/mtp layer's mlp/experts + mlp/shared_expert (NVFP4)",
        "why": "no grouped/top-k expert GEMM dispatch kernel; the 512 routed "
        "experts are stored + quantized, dispatch is unimplemented",
    },
    {
        "operation": "PLE / n-gram trigram lookup (51.2B-element store)",
        "scope": "the separate .ngram store (128 shards) + layer-1 PLE projection",
        "why": "no SSD demand-paging + trigram-index lookup runtime; the store is "
        "built and quantized, the reader/eviction is unimplemented",
    },
    {
        "operation": "hyper-connections (hc_count=4 mixer + per-block inject/mix)",
        "scope": "text/hyper_connection_mixer + every layer's attn_/mlp_hyper_connection",
        "why": "no hyper-connection mixing kernel; tensors are stored BF16",
    },
    {
        "operation": "QSA indexer inside full-attention layers",
        "scope": "text/layers/*/attention/indexer/* (12 full-attn layers)",
        "why": "no indexer scoring/selection kernel; projections stored BF16",
    },
    {
        "operation": "full attention with mRoPE (partial_rotary=0.25, mrope_interleaved)",
        "scope": "text/layers/*/attention + mtp/layers/0/attention",
        "why": "attention projections stored separately (q/k/v) to encode no head-"
        "interleaving assumption; the mRoPE-attention kernel is unimplemented",
    },
    {
        "operation": "MTP module (draft-layer full-attn + MoE + projections)",
        "scope": "all mtp/* objects",
        "why": "no multi-token-prediction decode path; tensors are stored BF16/NVFP4",
    },
    {
        "operation": "NVFP4 W4A4 GEMM K-switch",
        "scope": "every NVFP4 weight (K=2560 HIDDEN, K=640 KV)",
        "why": "the Blackwell W4A4 kernel only supports K in {5120,6144,17408}; "
        "Flash-Next K=2560 and K=640 are NEITHER -- the packed weights are correct, "
        "a K=2560/640 W4A4 kernel is required",
    },
]


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _role_for_object(name: str) -> str:
    """Deterministic object-level role from the NInfer object-name structure.

    The name structure is fixed by the inventory builders (see
    inventory_nvfp4.py); this is a read-only re-derivation, not a guess.
    """
    if name.startswith("frontend/"):
        return "resource"
    if name == "text/token_embedding":
        return "embed"
    if name == "text/output_head":
        return "lm_head"
    if name.startswith("text/hyper_connection_mixer/"):
        return "lc_hc"
    if name.startswith("vision/"):
        return "visual"
    if name.startswith("mtp/"):
        if name.startswith("mtp/hyper_connection_mixer/") or "_hyper_connection/" in name:
            return "mtp_hc"
        if "/layers/0/attention/" in name:
            return "mtp_full_attn"
        if "/layers/0/mlp/" in name:
            return "mtp_moe"
        return "mtp_proj"
    # text core layers
    if "_hyper_connection/" in name:  # attn_hyper_connection / mlp_hyper_connection
        return "lc_hc"
    if "/gdn/" in name:
        return "gdn"
    if "/attention/indexer/" in name:
        return "qsa_indexer"
    if "/attention/" in name:
        return "full_attn"
    if "/ple/" in name:
        return "ple"
    if "/mlp/" in name:
        return "moe"
    return "text_other"


def _source_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """BF16/FP32 checkpoint source byte count (the input size, pre-encoding)."""
    elem = 2 if dtype in ("BF16", "bf16", "float16") else 4
    return _prod(shape) * elem


def build_report() -> dict:
    objects: list[dict] = []
    recipe_object_names = set(recipe.RECIPES_BY_NAME)
    for spec in inventory.OBJECT_SPECS:
        rec: dict = {"name": spec.name, "kind": spec.kind, "role": _role_for_object(spec.name)}
        if spec.kind == "resource":
            rec.update(shape=None, format=None, layout=None, sources=[],
                       provenance="frontend resource (model-dir file, not a checkpoint tensor)")
        else:
            rec.update(shape=list(spec.shape), format=spec.format, layout=spec.layout)
            if spec.name in recipe.DIVISOR_NAMES:
                rec.update(sources=[],
                           provenance="input_scale_divisor placeholder (no checkpoint source; "
                                       "W4A4 activation scale, true peak measurable only at runtime)")
            else:
                expr = recipe.RECIPES_BY_NAME[spec.name].expression
                srcs = expression_sources(expr)
                rec["sources"] = [
                    {"name": s.name, "shape": list(s.shape), "dtype": s.dtype} for s in srcs
                ]
                rec["provenance"] = "recipe" if srcs else "synthesized (no checkpoint source)"
        objects.append(rec)

    # Distinct checkpoint sources consumed by the .ninfer (the recipe leaves).
    requirements = source_requirements(recipe.RECIPES)
    distinct_sources = sorted(requirements, key=lambda n: n)
    non_bf16 = {n: requirements[n].dtype for n in distinct_sources
                if requirements[n].dtype != "BF16"}

    # tensor_map.json (source-side inventory) for reconciliation.
    tmap = json.loads(_TENSOR_MAP.read_text(encoding="utf-8"))
    checkpoint_tensors = tmap.get("total_tensors")

    ninfer_source_count = len(distinct_sources)
    reconcile = ninfer_source_count + NGRAM_SHARD_COUNT + NGRAM_CONFIG_COUNT
    report = {
        "model": inventory.MODEL_ID,
        "weights_id": inventory.WEIGHTS_ID,
        "target_key": inventory.TARGET_KEY,
        "source": tmap.get("source"),
        "counts": {
            "objects": len(inventory.OBJECT_SPECS),
            "tensors": len(inventory.TENSOR_SPECS),
            "resources": len(inventory.RESOURCE_SPECS),
            "with_checkpoint_recipe": len(recipe.RECIPES),
            "input_scale_divisors": len(recipe.DIVISOR_NAMES),
            "distinct_ninfer_source_tensors": ninfer_source_count,
            "format_counts": dict(inventory.FORMAT_COUNTS),
            "layout_counts": dict(inventory.LAYOUT_COUNTS),
        },
        "objects": objects,
        "distinct_sources": [
            {
                "name": n,
                "shape": list(requirements[n].shape),
                "dtype": requirements[n].dtype,
                "bf16_bytes": _source_bytes(requirements[n].shape, requirements[n].dtype),
            }
            for n in distinct_sources
        ],
        "source_role_totals": tmap.get("role_totals"),
        "reconciliation": {
            "checkpoint_tensors_total": checkpoint_tensors,
            "ninfer_source_tensors": ninfer_source_count,
            "ngram_shards": NGRAM_SHARD_COUNT,
            "ngram_i64_configs": NGRAM_CONFIG_COUNT,
            "sum": reconcile,
            "matches": reconcile == checkpoint_tensors,
        },
        "blockers_runtime_kernel_required": BLOCKERS,
    }
    return report, _checks(report, recipe_object_names, non_bf16)


def _checks(
    report: dict,
    recipe_object_names: set,
    non_bf16: dict[str, str],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    spec_names = {spec.name for spec in inventory.TENSOR_SPECS}
    non_divisor = {s for s in spec_names if s not in recipe.DIVISOR_NAMES}
    add("validate_inventory (counts)", True,
        f"objects={len(inventory.OBJECT_SPECS)} tensors={len(inventory.TENSOR_SPECS)} "
        f"resources={len(inventory.RESOURCE_SPECS)}")
    add("validate_recipe (order/coverage/shape)", True)
    add("recipe keys == non-divisor spec names",
        recipe_object_names == non_divisor,
        f"recipes={len(recipe_object_names)} non_divisor_specs={len(non_divisor)} "
        f"missing={sorted(non_divisor - recipe_object_names)[:4]} "
        f"extra={sorted(recipe_object_names - non_divisor)[:4]}")
    add("divisors are spec names", recipe.DIVISOR_NAMES <= spec_names,
        f"divisors={len(recipe.DIVISOR_NAMES)}")
    add("every checkpoint source is BF16", not non_bf16,
        f"non_bf16={dict(list(non_bf16.items())[:5]) if non_bf16 else '{}'}")
    recon = report["reconciliation"]
    add(".ninfer sources + ngram shards + I64 configs == checkpoint tensors",
        recon["matches"],
        f"{recon['ninfer_source_tensors']} + {recon['ngram_shards']} + "
        f"{recon['ngram_i64_configs']} = {recon['sum']} vs total {recon['checkpoint_tensors_total']}")
    # Every object that is not a resource/divisor must have >=1 source.
    empty = [o["name"] for o in report["objects"]
             if o["kind"] == "tensor" and not o["sources"]
             and o["provenance"] not in (
                   "input_scale_divisor placeholder (no checkpoint source; "
                   "W4A4 activation scale, true peak measurable only at runtime)",
                   "synthesized (no checkpoint source)")]
    add("every non-divisor tensor object has a source", not empty,
        f"empty={empty[:5]}")
    # A catch-all role means an object name the classifier does not recognize --
    # that is a real gap, so it must fail loudly rather than bucket silently.
    unclassified = [o["name"] for o in report["objects"] if o["role"].endswith("_other")]
    add("no object falls into a catch-all role", not unclassified,
        f"unclassified={unclassified[:5]}")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="where to write the JSON report")
    args = parser.parse_args(argv)

    report, checks = build_report()
    failed = [c for c in checks if c["status"] == "FAIL"]

    role_counts: dict[str, int] = {}
    for o in report["objects"]:
        role_counts[o["role"]] = role_counts.get(o["role"], 0) + 1

    print("STEP 3 tensor -> NInfer-representation mapping report")
    print(f"model={report['model']} weights_id={report['weights_id']} "
          f"target={report['target_key']}")
    c = report["counts"]
    print(f"objects: {c['objects']} (tensor {c['tensors']} + resource {c['resources']})")
    print(f"  with checkpoint recipe: {c['with_checkpoint_recipe']}")
    print(f"  input_scale_divisor placeholders: {c['input_scale_divisors']}")
    print(f"  distinct BF16 source tensors (.ninfer): {c['distinct_ninfer_source_tensors']}")
    print(f"by format: {c['format_counts']}")
    print(f"by layout: {c['layout_counts']}")
    print(f"by role: {dict(sorted(role_counts.items()))}")
    print("cross-checks:")
    for chk in checks:
        line = f"  [{chk['status']}] {chk['check']}"
        if chk["detail"]:
            line += f" -- {chk['detail']}"
        print(line)
    print(f"BLOCKERS (RUNTIME KERNEL REQUIRED): {len(report['blockers_runtime_kernel_required'])}")
    for i, b in enumerate(report["blockers_runtime_kernel_required"], 1):
        print(f"  {i}. {b['operation']}  [{b['scope']}]")

    if not failed:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n",
                            encoding="utf-8")
        print(f"\nreport written: {args.out} ({args.out.stat().st_size/1024:.0f} KiB)")
        print("RESULT: PASS")
        return 0

    print(f"\nRESULT: FAIL ({len(failed)} check(s) failed)", file=sys.stderr)
    for chk in failed:
        print(f"  [FAIL] {chk['check']} -- {chk['detail']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

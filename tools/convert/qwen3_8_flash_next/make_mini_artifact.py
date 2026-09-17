"""P1: build the Flash-Next MINI artifact (production codecs, scaled shapes).

The mini is the fixture the rest of the Flash-Next work runs against
(agent_prompt.md P1 / §4.2).  It is a real v2 ``.ninfer`` + a real v1
``.ngram`` PLE store, byte-for-byte on the production codec paths (same
encoders, same layouts, same identity, same per-object name scheme as the
canonical 27B-class inventory at ``inventory_nvfp4``), scaled to §4.2 shapes:

    hidden=256  layers=4 (L0-2 GDN, L3 full-attention)  experts=8 top-2
    vocab=256   hc_count=4 hc_lowrank=32   q 4/2 heads x 32   gdn 2/4 heads x 32
    moe_intermediate=64   PLE at L1        PLE table: 1024 rows x 4 heads

Deliberate scaled-shape decisions (documented, load-bearing for P4/P8):

* **PLE table geometry**: production shards are (2500012, 160) = 16 heads x
  10 values/head; a block is 128 rows so that ``128*160 = 20480 = 5*4096``
  bytes -- the 4096-aligned NVMe block contract.  With 4 heads x 10 values =
  40 columns NO sub-shard block can be 4096-aligned (128*40 = 5120), so the
  mini uses 8 values/head: cols = 32, and ``128*32 = 4096`` exactly.  The
  quant codec (``fp8e4m3fn_row_bf16s-v1``), row-scale layout, per-block
  paging, and the row addressing ``row = head*rows_per_head + ngram_id`` are
  unchanged.
* **Expert indexability**: the fused expert matrices pack experts
  expert-major; rows per expert are gate_up 2*inter = 128 and down hidden =
  256 (mini), both multiples of the blockscale layout's 128-row N-tile, so a
  ``(layer, expert_id)`` slice starts at a scale-plane boundary.
* **Identity**: the mini carries the PRODUCTION identity
  (``qwen3.8-flash-next`` / ``nvfp4``) -- it is the same target at scaled
  shapes; the directory (per-object shapes/bytes), not a cfg constant, is the
  source of truth for the runtime.  Mini-ness is carried by the file name +
  the mini inventory JSON emitted next to it.
* **MTP and Vision are OUT OF SCOPE until P13+**: the mini text core contains
  no ``mtp/`` or ``vision/`` objects (the six ``frontend/`` resource stubs
  are still written -- the resource set is the family convention).

Determinism: every tensor payload is a seeded function of its object name
(sha256 -> torch.Generator), so a rebuild is byte-identical.  The PLE golden
file ``<mini>.ple_golden.bin`` stores the exact dequantized FP32 values of
ngram shard 0 / block 0 (magic + dims + row-major f32) so the C++ reader test
can compare its own E4M3FN/BF16 dequant bit-for-bit.

Usage::

    python -m tools.convert.qwen3_8_flash_next.make_mini_artifact \
      --out out/flash_next_dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity
from tools.artifact.layouts import (
    PLANE_ALIGNMENT,
    align_up,
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
    encoded_size,
)
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.inventory import BF16, FP32, ResourceSpec, TensorSpec

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import ngram_store
from .artifact_writer import ResumableArtifactWriter
from .quantize_fp8_row import quantize_fp8_row
from .quantize_nvfp4 import quantize_nvfp4

NVFP4 = inventory.NVFP4
FP8 = inventory.FP8

# --- §4.2 locked mini geometry ------------------------------------------------

@dataclass(frozen=True, slots=True)
class MiniGeometry:
    hidden: int = 256
    vocab: int = 256
    num_layers: int = 4
    full_attention_interval: int = 4   # layer 3 full-attention; 0-2 GDN
    ple_layer: int = 1
    q_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 32
    gdn_key_heads: int = 2
    gdn_value_heads: int = 4
    gdn_key_dim: int = 32
    gdn_value_dim: int = 32
    experts: int = 8
    top_experts: int = 2
    shared_experts: int = 1
    moe_intermediate: int = 64
    shared_intermediate: int = 64
    hc_count: int = 4
    hc_lowrank: int = 32
    ple_rows: int = 1024
    ple_heads: int = 4

    # --- derived shapes (production formulas, mini inputs) ---

    @property
    def hc_stream(self) -> int:
        return self.hc_count * self.hidden

    @property
    def full_attention_layers(self) -> frozenset[int]:
        return frozenset(l for l in range(self.num_layers)
                         if l % self.full_attention_interval == 3)

    @property
    def gdn_layers(self) -> frozenset[int]:
        return frozenset(
            l for l in range(self.num_layers) if l not in self.full_attention_layers
        )

    # full attention
    @property
    def q_proj_out(self) -> int:
        # production: NUM_ATTENTION_HEADS * (2 * HEAD_DIM) (validate_config)
        return self.q_heads * (2 * self.head_dim)

    @property
    def kv_proj_out(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def o_proj_out(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def qk_norm(self) -> int:
        return self.head_dim

    @property
    def indexer_qk_proj_out(self) -> int:
        # production indexer = 4Q + 1K heads x head_dim (640 = 5 * 128)
        return (self.q_heads + 1) * self.head_dim

    @property
    def indexer_norm(self) -> int:
        return self.head_dim

    # GDN
    @property
    def gdn_a(self) -> int:
        # production GDN_A = value_heads (48)
        return self.gdn_value_heads

    @property
    def gdn_qkv_out(self) -> int:
        return (
            self.gdn_key_heads * self.gdn_key_dim
            + self.gdn_key_heads * self.gdn_key_dim
            + self.gdn_value_heads * self.gdn_value_dim
        )

    @property
    def gdn_z(self) -> int:
        return self.gdn_value_heads * self.gdn_value_dim

    @property
    def gdn_qkvz_out(self) -> int:
        return self.gdn_qkv_out + self.gdn_z

    @property
    def gdn_ab_proj_out(self) -> int:
        return 2 * self.gdn_a

    @property
    def gdn_norm(self) -> int:
        # production GDN_NORM = key_dim (128)
        return self.gdn_key_dim

    # HC
    @property
    def hc_block_inject(self) -> tuple[int, int]:
        return (self.hc_count, self.hc_stream)

    @property
    def hc_norm_shape(self) -> tuple[int, ...]:
        return (self.hc_stream,)

    @property
    def hc_mix_down(self) -> tuple[int, int]:
        return (self.hc_lowrank, self.hc_stream)

    @property
    def hc_mix_up(self) -> tuple[int, int]:
        return (self.hc_stream, self.hc_lowrank)

    # MoE
    @property
    def router_out(self) -> int:
        return self.experts

    @property
    def experts_gate_up(self) -> tuple[int, int]:
        return (self.experts * 2 * self.moe_intermediate, self.hidden)

    @property
    def experts_down(self) -> tuple[int, int]:
        return (self.experts * self.hidden, self.moe_intermediate)

    @property
    def shared_gate_up(self) -> tuple[int, int]:
        return (2 * self.shared_intermediate, self.hidden)

    @property
    def shared_down(self) -> tuple[int, int]:
        return (self.hidden, self.shared_intermediate)

    @property
    def shared_gate(self) -> tuple[int, int]:
        return (1, self.hidden)

    # PLE projections (layer ple_layer only)
    @property
    def ple_conv(self) -> tuple[int, int]:
        return (4, self.hc_stream)

    @property
    def ple_key_proj(self) -> tuple[int, int]:
        return (self.hc_stream, self.hidden)

    @property
    def ple_value_proj(self) -> tuple[int, int]:
        return (self.hidden, self.hidden)

    @property
    def ple_norm_shape(self) -> tuple[int, ...]:
        return (self.hc_stream,)

    # PLE n-gram table (.ngram)
    @property
    def ple_cols(self) -> int:
        # 8 values/head (production 10) so that 128 rows * 32 cols = 4096
        # bytes exactly (the aligned-NVMe-block contract; 40 cols admits no
        # 4096-aligned sub-shard block).
        return self.ple_heads * 8

    @property
    def ple_rows_per_head(self) -> int:
        return self.ple_rows // self.ple_heads

    def validate(self) -> None:
        assert self.full_attention_layers == {3}, "mini L3 must be the full-attention layer"
        assert self.gdn_layers == {0, 1, 2}, "mini L0-L2 must be GDN"
        assert self.ple_layer in self.gdn_layers
        # production invariants (config.py validate_config) at mini inputs
        assert self.q_proj_out == self.q_heads * (2 * self.head_dim)
        assert self.gdn_qkvz_out == self.gdn_qkv_out + self.gdn_z
        # NVFP4 blockscale geometry: N % 128 == 0, K % 64 == 0
        for n, k in (
            self.experts_gate_up, self.experts_down,
            self.shared_gate_up, self.shared_down,
        ):
            assert n % 128 == 0 and k % 64 == 0, f"NVFP4 geometry violated: {(n, k)}"
        # expert-major indexability: per-expert row strides span whole
        # 128-row blockscale N-tiles
        assert self.experts_gate_up[0] % self.experts == 0
        assert (self.experts_gate_up[0] // self.experts) % 128 == 0
        assert self.experts_down[0] % self.experts == 0
        assert (self.experts_down[0] // self.experts) % 128 == 0
        # PLE table: block contract (128 rows * cols is 4096-aligned)
        assert self.ple_rows % ngram_store.BLOCK_ROWS == 0
        assert self.ple_rows % self.ple_heads == 0
        assert self.ple_rows * self.ple_cols % ngram_store.NGRAM_ALIGN == 0
        assert self.ple_cols % 4096 == 0 or (self.ple_rows % 128 == 0
                                             and (128 * self.ple_cols) % 4096 == 0)


GEOM = MiniGeometry()
GEOM.validate()

MINI_NINFER_BASENAME = "qwen3_8_flash_next_mini.ninfer"
MINI_NGRAM_BASENAME = "qwen3_8_flash_next_mini.ngram"
MINI_INVENTORY_BASENAME = "qwen3_8_flash_next_mini.inventory.json"
MINI_PLE_GOLDEN_BASENAME = "qwen3_8_flash_next_mini.ple_golden.bin"
PLE_GOLDEN_MAGIC = b"PLEGOLDN"

# Identity is PRODUCTION (see module docstring): the mini is the same target
# at scaled shapes; the directory carries the shapes.
IDENTITY = ArtifactIdentity(model_id=inventory.MODEL_ID, weights_id=inventory.WEIGHTS_ID)


# --- mini tensor inventory (mirrors inventory_nvfp4 text core at mini shapes) --

def _hyper_connection_specs(prefix: str, block: str) -> tuple[TensorSpec, ...]:
    p = f"{prefix}{block}_hyper_connection/"
    return (
        TensorSpec(p + "block_inject_weight", GEOM.hc_block_inject, BF16,
                   inventory.CONTIGUOUS_LAYOUT),
        TensorSpec(p + "hc_norm", GEOM.hc_norm_shape, BF16, inventory.CONTIGUOUS_LAYOUT),
        TensorSpec(p + "input_mix_weight_down", GEOM.hc_mix_down, BF16,
                   inventory.CONTIGUOUS_LAYOUT),
        TensorSpec(p + "input_mix_weight_up", GEOM.hc_mix_up, BF16,
                   inventory.CONTIGUOUS_LAYOUT),
    )


def _gdn_specs(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        inventory.tensor_spec(prefix + "gdn/a_log", (GEOM.gdn_a,), FP32),
        inventory.tensor_spec(prefix + "gdn/dt_bias", (GEOM.gdn_a,), FP32),
        inventory.tensor_spec(prefix + "gdn/convolution", (4, GEOM.gdn_qkv_out), BF16),
        inventory.tensor_spec(
            prefix + "gdn/a_b_projection", (GEOM.gdn_ab_proj_out, GEOM.hidden), BF16
        ),
        inventory.tensor_spec(
            prefix + "gdn/query_key_value_z", (GEOM.gdn_qkvz_out, GEOM.hidden), FP8
        ),
        inventory.tensor_spec(prefix + "gdn/norm", (GEOM.gdn_norm,), BF16),
        inventory.tensor_spec(prefix + "gdn/output", (GEOM.hidden, GEOM.gdn_z), FP8),
    )


def _full_attention_specs(prefix: str) -> tuple[TensorSpec, ...]:
    p = prefix + "attention/"
    return (
        inventory.tensor_spec(p + "query", (GEOM.q_proj_out, GEOM.hidden), FP8),
        inventory.tensor_spec(p + "key", (GEOM.kv_proj_out, GEOM.hidden), FP8),
        inventory.tensor_spec(p + "value", (GEOM.kv_proj_out, GEOM.hidden), FP8),
        inventory.tensor_spec(p + "output", (GEOM.hidden, GEOM.o_proj_out), FP8),
        inventory.tensor_spec(p + "query_norm", (GEOM.qk_norm,), BF16),
        inventory.tensor_spec(p + "key_norm", (GEOM.qk_norm,), BF16),
        inventory.tensor_spec(
            p + "indexer/qk_proj", (GEOM.indexer_qk_proj_out, GEOM.hidden), BF16
        ),
        inventory.tensor_spec(p + "indexer/k_norm", (GEOM.indexer_norm,), BF16),
        inventory.tensor_spec(p + "indexer/q_norm", (GEOM.indexer_norm,), BF16),
    )


def _moe_specs(prefix: str) -> tuple[TensorSpec, ...]:
    return (
        inventory.tensor_spec(prefix + "mlp/experts/gate_up", GEOM.experts_gate_up, NVFP4),
        inventory.tensor_spec(
            prefix + "mlp/experts/gate_up_projection/input_scale_divisor", (), FP32
        ),
        inventory.tensor_spec(prefix + "mlp/experts/down", GEOM.experts_down, NVFP4),
        inventory.tensor_spec(
            prefix + "mlp/experts/down_projection/input_scale_divisor", (), FP32
        ),
        inventory.tensor_spec(
            prefix + "mlp/experts/routing", (GEOM.router_out, GEOM.hidden), BF16
        ),
        inventory.tensor_spec(prefix + "mlp/shared_expert/gate_up", GEOM.shared_gate_up, NVFP4),
        inventory.tensor_spec(
            prefix + "mlp/shared_expert/gate_up_projection/input_scale_divisor", (), FP32
        ),
        inventory.tensor_spec(prefix + "mlp/shared_expert/down", GEOM.shared_down, NVFP4),
        inventory.tensor_spec(
            prefix + "mlp/shared_expert/down_projection/input_scale_divisor", (), FP32
        ),
        inventory.tensor_spec(prefix + "mlp/shared_expert_gate", GEOM.shared_gate, BF16),
    )


def _ple_specs(prefix: str) -> tuple[TensorSpec, ...]:
    p = prefix + "ple/"
    return (
        inventory.tensor_spec(p + "convolution", GEOM.ple_conv, BF16),
        inventory.tensor_spec(p + "key_projection", GEOM.ple_key_proj, BF16),
        inventory.tensor_spec(p + "value_projection", GEOM.ple_value_proj, BF16),
        inventory.tensor_spec(p + "norm_conv", GEOM.ple_norm_shape, BF16),
        inventory.tensor_spec(p + "norm_key", GEOM.ple_norm_shape, BF16),
        inventory.tensor_spec(p + "norm_query", GEOM.ple_norm_shape, BF16),
    )


def _text_layer_specs(layer: int) -> tuple[TensorSpec, ...]:
    obj = f"text/layers/{layer}/"
    specs: list[TensorSpec] = []
    specs.extend(_hyper_connection_specs(obj, "attn"))
    if layer in GEOM.full_attention_layers:
        specs.extend(_full_attention_specs(obj))
    else:
        specs.extend(_gdn_specs(obj))
    specs.extend(_hyper_connection_specs(obj, "mlp"))
    specs.extend(_moe_specs(obj))
    if layer == GEOM.ple_layer:
        specs.extend(_ple_specs(obj))
    return tuple(specs)


MINI_TENSOR_SPECS: tuple[TensorSpec, ...] = (
    inventory.tensor_spec(
        "text/token_embedding", (GEOM.vocab, GEOM.hidden), FP8
    ),
    inventory.tensor_spec(
        "text/hyper_connection_mixer/hc_norm", GEOM.hc_norm_shape, BF16
    ),
    inventory.tensor_spec(
        "text/hyper_connection_mixer/input_mix_weight_down", GEOM.hc_mix_down, BF16
    ),
    inventory.tensor_spec(
        "text/hyper_connection_mixer/input_mix_weight_up", GEOM.hc_mix_up, BF16
    ),
    *[spec for layer in range(GEOM.num_layers) for spec in _text_layer_specs(layer)],
    inventory.tensor_spec("text/output_head", (GEOM.vocab, GEOM.hidden), FP8),
)

# Family-convention resource set (the mini carries stub bytes; P10 refines the
# tokenizer/template semantics).  The mini has no vision objects, so the
# preprocessor stubs document the production pixel contract without claiming
# vision weights.
_RESOURCE_STUBS: Mapping[str, bytes] = {
    "frontend/tokenizer.json": json.dumps(
        {
            "version": "1.0",
            "model": {
                "type": "BPE",
                # byte-level stub: 256 byte tokens, no merges
                "vocab": {str(i): i for i in range(GEOM.vocab)},
                "merges": [],
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<bos>", "special": True},
                {"id": 2, "content": "<eos>", "special": True},
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
    "frontend/tokenizer_config.json": json.dumps(
        {
            "model_max_length": 65536,
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<unk>",
            "chat_template": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
    "frontend/chat_template.jinja": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
        "{{ eos_token }}"
    ).encode("utf-8"),
    "frontend/generation_config.json": json.dumps(
        {"do_sample": False, "max_new_tokens": 128, "temperature": None,
         "top_p": None, "top_k": None},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
    # Production vision contract (16 MP registered cap); the mini has no
    # vision weights (out of scope until P13+).
    "frontend/preprocessor_config.json": json.dumps(
        {
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16,
            "temporal_patch_size": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
    "frontend/video_preprocessor_config.json": json.dumps(
        {
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "num_frames": 4,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"),
}

# Canonical family resource set (the six frontend objects, raw-bytes-v1);
# object order mirrors the production OBJECT_SPECS = RESOURCE_SPECS + TENSOR_SPECS.
MINI_RESOURCE_SPECS: tuple[ResourceSpec, ...] = inventory.RESOURCE_SPECS
assert tuple(spec.name for spec in MINI_RESOURCE_SPECS) == tuple(_RESOURCE_STUBS)


# --- deterministic payloads ---------------------------------------------------

def _seed_for(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def _bf16_source(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_for(name))
    return (torch.randn(shape, dtype=torch.float32, generator=gen) * 0.05).bfloat16()


def _encode_tensor(spec: TensorSpec) -> bytes:
    if spec.format == NVFP4:
        qz = quantize_nvfp4(_bf16_source(spec.name, spec.shape))
        return encode_nvfp4(
            qz.packed_codes, qz.natural_scales, qz.weight_scale_divisor, spec.shape
        )
    if spec.format == FP8:
        qz = quantize_fp8_row(_bf16_source(spec.name, spec.shape))
        return encode_fp8_row_scaled(qz.code_words, qz.row_scales, spec.shape)
    if spec.format == BF16:
        return encode_direct(_bf16_source(spec.name, spec.shape), BF16)
    if spec.format == FP32:
        if spec.shape == ():
            # placeholder activation scale (same convention as the real
            # converter: the true activation peak is not measurable pre-runtime)
            value = torch.tensor(cfg.INPUT_SCALE_DIVISOR_PLACEHOLDER, dtype=torch.float32)
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(_seed_for(spec.name))
            value = torch.randn(tuple(spec.shape), dtype=torch.float32, generator=gen) * 0.01
        return encode_direct(value, FP32)
    raise ValueError(f"unsupported mini format {spec.format} for {spec.name}")


def _validate_payload(spec: TensorSpec, payload: bytes) -> None:
    expected = encoded_size(spec.layout, spec.format, spec.shape)
    if len(payload) != expected:
        raise ValueError(
            f"{spec.name}: payload {len(payload)} bytes != encoded_size {expected}"
        )


# --- mini .ngram PLE table ----------------------------------------------------

def _mini_ngram_config_tensors() -> tuple[ngram_store.ConfigTensor, ...]:
    heads = GEOM.ple_heads
    rph = GEOM.ple_rows_per_head
    return (
        # stubs (P4 defines the gather semantics; values are small int64):
        # trigram layer multipliers, per-head ID offsets / vocab sizes.
        ngram_store.ConfigTensor("layer_multipliers", (3,), (1, 1, 1)),
        ngram_store.ConfigTensor(
            "ngram_heads_offsets", (heads,), tuple(h * rph for h in range(heads))
        ),
        ngram_store.ConfigTensor(
            "ngram_heads_vocab_sizes", (heads,), tuple(rph for _ in range(heads))
        ),
    )


def _mini_ngram_directory() -> bytes:
    """Mirror of ngram_store.build_directory at the mini PLE geometry.

    ``build_directory``/``validate_config_tensors`` validate against the
    PRODUCTION config shapes ((16,) head tables), so the mini builds its
    directory directly on the same JSON schema + shared row-scale geometry.
    """

    rows, cols, block_rows, n_shards = (
        GEOM.ple_rows, GEOM.ple_cols, ngram_store.BLOCK_ROWS, 1,
    )
    ngram_store._validate_geometry(rows, cols, block_rows)
    configs = _mini_ngram_config_tensors()
    geo = ngram_store.shard_geometry(rows, cols)

    payload_offset = 0
    directory = None
    for _ in range(16):
        block_codes_bytes = block_rows * cols
        shard = {
            "id": 0,
            "offset": payload_offset,
            "scale_offset": payload_offset + geo.scale_plane_offset,
            "bytes": geo.payload_bytes,
        }
        obj = {
            "version": ngram_store.NGRAM_VERSION,
            "model_id": IDENTITY.model_id,
            "quant_id": ngram_store.QUANT_ID,
            "n_shards": n_shards,
            "rows_per_shard": rows,
            "cols_per_shard": cols,
            "block_rows": block_rows,
            "payload_offset": payload_offset,
            "shard_stride": align_up(geo.payload_bytes, ngram_store.NGRAM_ALIGN),
            "planes": {
                "code_plane_bytes": geo.code_plane_bytes,
                "scale_plane_offset": geo.scale_plane_offset,
                "scale_plane_bytes": geo.scale_plane_bytes,
                "payload_bytes": geo.payload_bytes,
                "plane_alignment": PLANE_ALIGNMENT,
            },
            "block": {
                "rows": block_rows,
                "cols": cols,
                "code_row_bytes": cols,
                "block_codes_bytes": block_codes_bytes,
                "scale_row_bytes": 2,
                "block_scale_bytes": block_rows * 2,
                "blocks_per_shard": -(-rows // block_rows),
                "note": (
                    "block b of shard s: codes at shards[s].offset + b*"
                    f"{block_codes_bytes} ({block_codes_bytes}="
                    f"{block_codes_bytes // ngram_store.NGRAM_ALIGN}*"
                    f"{ngram_store.NGRAM_ALIGN}, aligned); scales at "
                    "shards[s].scale_offset + b*" f"{block_rows * 2}"
                ),
            },
            # mini ngram metadata: 4 heads, 8 values/head, trigram size 3,
            # 256 ngram IDs per head (rows = head*256 + id)
            "ngram": {
                "size": 3,
                "heads": GEOM.ple_heads,
                "heads_per_ngram": 8,
                "vocab_base": GEOM.ple_rows_per_head,
            },
            "config_tensors": [
                {"name": c.name, "dtype": "I64", "shape": list(c.shape),
                 "values": list(c.values)}
                for c in configs
            ],
            "shards": [shard],
        }
        directory = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        candidate = align_up(
            ngram_store.PREFIX_BYTES + len(directory), ngram_store.NGRAM_ALIGN
        )
        if candidate == payload_offset:
            return directory
        payload_offset = candidate
    raise RuntimeError("mini ngram directory did not reach a fixed point")


def _mini_ple_shard() -> bytes:
    """The PLE table shard: rows = head*rows_per_head + ngram_id, cols =
    head*8 + value.  Encoded with the production FP8 row-scale codec."""

    return ngram_store.encode_shard(_bf16_source("ngram/shard_0", (GEOM.ple_rows, GEOM.ple_cols)))


def _mini_ple_golden() -> bytes:
    """Exact dequantized block 0 (f32), for the C++ reader's bit-for-bit check."""

    rows, cols = GEOM.ple_rows, GEOM.ple_cols
    qz = quantize_fp8_row(_bf16_source("ngram/shard_0", (rows, cols)))
    assert qz.code_words.numel() == rows * cols
    codes = qz.code_words.view(torch.float8_e4m3fn).float()
    dequant = codes * qz.row_scales.float().unsqueeze(1)
    block = dequant[: ngram_store.BLOCK_ROWS].contiguous()
    body = block.reshape(-1).view(torch.int32).numpy().tobytes()
    return struct.pack("<8sii", PLE_GOLDEN_MAGIC, ngram_store.BLOCK_ROWS, cols) + body


# --- build --------------------------------------------------------------------

def build(out_dir: Path) -> dict[str, object]:
    """Build the mini .ninfer + .ngram (+ sidecars) into *out_dir* (idempotent)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    resources = {spec.name: _RESOURCE_STUBS[spec.name] for spec in MINI_RESOURCE_SPECS}
    tensor_payloads: dict[str, bytes] = {}
    for spec in MINI_TENSOR_SPECS:
        payload = _encode_tensor(spec)
        _validate_payload(spec, payload)
        tensor_payloads[spec.name] = payload
    object_specs = MINI_RESOURCE_SPECS + MINI_TENSOR_SPECS
    plan = family_conversion.build_object_plan(object_specs, resources)

    ninfer_path = out_dir / MINI_NINFER_BASENAME
    with ResumableArtifactWriter(ninfer_path, IDENTITY, plan.specs, overwrite=True) as writer:
        for spec in object_specs:
            payload = (
                resources[spec.name]
                if isinstance(spec, ResourceSpec)
                else tensor_payloads[spec.name]
            )
            writer.write(spec.name, payload)
    objects = plan.objects

    ngram_path = out_dir / MINI_NGRAM_BASENAME
    directory = _mini_ngram_directory()
    shard = _mini_ple_shard()
    with ngram_store.NgramWriter(ngram_path, directory, n_shards=1, overwrite=True) as writer:
        writer.write_shard(0, shard)
    golden = _mini_ple_golden()
    (out_dir / MINI_PLE_GOLDEN_BASENAME).write_bytes(golden)
    ngram_store.write_checksums_sidecar(ngram_path, directory, {0: ngram_store._sha256(shard)})

    inventory_doc = {
        "note": (
            "Mini Flash-Next inventory (scaled shapes, production codecs). "
            "Identity is the production identity; shapes come from this file / "
            "the artifact directory, not from config.py."
        ),
        "identity": {"model_id": IDENTITY.model_id, "weights_id": IDENTITY.weights_id},
        "target_key": inventory.TARGET_KEY,
        "geometry": {
            "hidden": GEOM.hidden,
            "vocab": GEOM.vocab,
            "layers": GEOM.num_layers,
            "full_attention_layers": sorted(GEOM.full_attention_layers),
            "gdn_layers": sorted(GEOM.gdn_layers),
            "ple_layer": GEOM.ple_layer,
            "experts": GEOM.experts,
            "top_experts": GEOM.top_experts,
            "moe_intermediate": GEOM.moe_intermediate,
            "hc_count": GEOM.hc_count,
            "hc_lowrank": GEOM.hc_lowrank,
            "q_heads": GEOM.q_heads,
            "kv_heads": GEOM.kv_heads,
            "head_dim": GEOM.head_dim,
            "gdn_key_heads": GEOM.gdn_key_heads,
            "gdn_value_heads": GEOM.gdn_value_heads,
            "ple_rows": GEOM.ple_rows,
            "ple_heads": GEOM.ple_heads,
            "ple_cols": GEOM.ple_cols,
        },
        "resources": [
            {"name": spec.name, "encoding": spec.encoding,
             "bytes": len(resources[spec.name])}
            for spec in MINI_RESOURCE_SPECS
        ],
        "tensors": [
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "format": spec.format,
                "layout": spec.layout,
                "bytes": encoded_size(spec.layout, spec.format, spec.shape),
            }
            for spec in MINI_TENSOR_SPECS
        ],
        "ngram": {
            "file": MINI_NGRAM_BASENAME,
            "quant_id": ngram_store.QUANT_ID,
            "n_shards": 1,
            "rows_per_shard": GEOM.ple_rows,
            "cols_per_shard": GEOM.ple_cols,
            "block_rows": ngram_store.BLOCK_ROWS,
            "row_addressing": "row = head * rows_per_head + ngram_id; col = head * 8 + value",
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (out_dir / MINI_INVENTORY_BASENAME).write_text(
        json.dumps(inventory_doc, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "ninfer": str(ninfer_path),
        "ngram": str(ngram_path),
        "objects": len(objects),
        "file_bytes": ninfer_path.stat().st_size,
        "elapsed_seconds": inventory_doc["elapsed_seconds"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/flash_next_dev",
                        help="output directory for the mini fixture files")
    args = parser.parse_args(argv)
    result = build(Path(args.out))
    print(
        f"mini artifact: {result['ninfer']} ({result['file_bytes']:,} B, "
        f"{result['objects']} objects, {result['elapsed_seconds']:.2f} s)"
    )


if __name__ == "__main__":
    main()

"""Persistent-object contract for the Qwen3.8-Flash-Next NVFP4 artifact.

This is the *main* ``.ninfer`` container inventory.  It holds the text core
(embedding / 48 layers / output head), the MTP module, and the Vision tower.

The 128 PLE n-gram embedding shards (51.2B params) are intentionally NOT here;
they live in a separate paging-oriented ``.ngram`` store (see ``ngram_store.py``)
so the main artifact stays a single mmap-friendly file.

Format allocation (matches the Qwen3.8-27B convention: big GEMMs quantized,
norms / control / routing kept high-precision):

* MoE expert + shared-expert gate_up / down  -> ``NVFP4`` (the dominant GEMMs)
* attention projections + GDN qkv_z / output  -> ``FP8_E4M3FN_ROW_BF16S``
* token embedding + output head               -> ``FP8_E4M3FN_ROW_BF16S``
* GDN a_log / dt_bias, every ``input_scale_divisor`` -> ``FP32``
* PLE n-gram config (I64, 63-bit) + 128 embedding shards -> the separate
  ``.ngram`` store (not representable in the ``.ninfer``)
* everything else (norms, hyper-connections, router, indexers, PLE projections,
  MTP projections, GDN conv / a-b projection)  -> ``BF16``
* Vision tower                                -> the shared Qwen3.6 inventory
  (``Q4G64_F16S`` / ``Q5G64_F16S`` / ``Q6G64_F16S`` / ``W8G32_F16S`` + BF16 biases)
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
)
from tools.convert.qwen3_8_flash_next import config as cfg

MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_8_flash_next"

NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

FORMAT_NAMES = (BF16, FP32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    BLOCK_SCALE_LAYOUT,
    ROW_SCALE_LAYOUT,
)

FULL_ATTENTION_LAYERS = cfg.FULL_ATTENTION_LAYERS
GDN_LAYERS = cfg.GDN_LAYERS
PLE_LAYER = cfg.PLE_LAYER


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Map a numeric format to its canonical layout for this target."""

    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif numeric_format == FP8:
        layout = ROW_SCALE_LAYOUT
    else:
        raise ValueError(f"unsupported Flash-Next NVFP4 format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


# --- per-block builders ------------------------------------------------------

def _hyper_connection_specs(prefix: str, block: str) -> tuple[TensorSpec, ...]:
    """One per-block hyper-connection (``attn`` or ``mlp``)."""

    p = f"{prefix}{block}_hyper_connection/"
    return (
        tensor_spec(p + "block_inject_weight", cfg.HC_BLOCK_INJECT, BF16),
        tensor_spec(p + "hc_norm", cfg.HC_NORM, BF16),
        tensor_spec(p + "input_mix_weight_down", cfg.HC_MIX_DOWN, BF16),
        tensor_spec(p + "input_mix_weight_up", cfg.HC_MIX_UP, BF16),
    )


def _gdn_specs(prefix: str) -> tuple[TensorSpec, ...]:
    """Gated-DeltaNet linear-attention block (36 layers)."""

    return (
        tensor_spec(prefix + "gdn/a_log", (cfg.GDN_A,), FP32),
        tensor_spec(prefix + "gdn/dt_bias", (cfg.GDN_A,), FP32),
        tensor_spec(prefix + "gdn/convolution", cfg.GDN_CONV, BF16),
        tensor_spec(
            prefix + "gdn/a_b_projection",
            (cfg.GDN_AB_PROJ_OUT, cfg.HIDDEN),
            BF16,
        ),
        tensor_spec(
            prefix + "gdn/query_key_value_z",
            (cfg.GDN_QKVZ_OUT, cfg.HIDDEN),
            FP8,
        ),
        tensor_spec(prefix + "gdn/norm", (cfg.GDN_NORM,), BF16),
        tensor_spec(prefix + "gdn/output", cfg.GDN_OUT_PROJ, FP8),
    )


def _full_attention_specs(prefix: str) -> tuple[TensorSpec, ...]:
    """Full (mRoPE) attention block with its QSA indexer (12 layers).

    Stored as separate query / key / value projections (unlike the 27B fused
    ``query_key_gate_value``) because Flash-Next uses a different head layout
    (mRoPE partial-rotary, mrope_interleaved) -- plain separate storage encodes
    no head-interleaving assumption.
    """

    p = prefix + "attention/"
    return (
        tensor_spec(p + "query", (cfg.Q_PROJ_OUT, cfg.HIDDEN), FP8),
        tensor_spec(p + "key", (cfg.KV_PROJ_OUT, cfg.HIDDEN), FP8),
        tensor_spec(p + "value", (cfg.KV_PROJ_OUT, cfg.HIDDEN), FP8),
        tensor_spec(p + "output", (cfg.HIDDEN, cfg.O_PROJ_OUT), FP8),
        tensor_spec(p + "query_norm", (cfg.QK_NORM,), BF16),
        tensor_spec(p + "key_norm", (cfg.QK_NORM,), BF16),
        tensor_spec(
            p + "indexer/qk_proj", (cfg.INDEXER_QK_PROJ_OUT, cfg.HIDDEN), BF16
        ),
        tensor_spec(p + "indexer/k_norm", (cfg.INDEXER_NORM,), BF16),
        tensor_spec(p + "indexer/q_norm", (cfg.INDEXER_NORM,), BF16),
    )


def _moe_specs(prefix: str) -> tuple[TensorSpec, ...]:
    """MoE MLP: 512 routed experts + 1 shared expert + router + shared gate.

    The expert weight matrices are the single largest tensors in the model and
    are NVFP4.  Each NVFP4 weight is paired with a 4-byte ``input_scale_divisor``
    scalar object (the activation's global scale the W4A4 kernel needs); a
    documented placeholder is stored because the MoE runtime is a BLOCKER and
    the true activation peak is not measurable pre-runtime.
    """

    return (
        tensor_spec(prefix + "mlp/experts/gate_up", cfg.EXPERTS_GATE_UP, NVFP4),
        tensor_spec(
            prefix + "mlp/experts/gate_up_projection/input_scale_divisor", (), FP32
        ),
        tensor_spec(prefix + "mlp/experts/down", cfg.EXPERTS_DOWN, NVFP4),
        tensor_spec(
            prefix + "mlp/experts/down_projection/input_scale_divisor", (), FP32
        ),
        tensor_spec(prefix + "mlp/experts/routing", (cfg.ROUTER_OUT, cfg.HIDDEN), BF16),
        tensor_spec(prefix + "mlp/shared_expert/gate_up", cfg.SHARED_GATE_UP, NVFP4),
        tensor_spec(
            prefix + "mlp/shared_expert/gate_up_projection/input_scale_divisor", (), FP32
        ),
        tensor_spec(prefix + "mlp/shared_expert/down", cfg.SHARED_DOWN, NVFP4),
        tensor_spec(
            prefix + "mlp/shared_expert/down_projection/input_scale_divisor", (), FP32
        ),
        tensor_spec(prefix + "mlp/shared_expert_gate", cfg.SHARED_GATE, BF16),
    )


def _ple_specs(prefix: str) -> tuple[TensorSpec, ...]:
    """PLE projection/control tensors (layer 1 only).

    The 128 n-gram embedding shards AND the 3 I64 n-gram-index config tables
    (``layer_multipliers``, ``ngram_heads_offsets``, ``ngram_heads_vocab_sizes``)
    are deliberately excluded from the ``.ninfer``; they go to the separate
    ``.ngram`` store.  The config tables are I64 in the source and hold 63-bit
    values (verified up to ~8.8e18), which the ``.ninfer`` contiguous layout
    (BF16/FP32/I32 only) cannot represent -- so they live in the ``.ngram``
    header as native int64 alongside the data they describe.
    """

    p = prefix + "ple/"
    return (
        tensor_spec(p + "convolution", cfg.PLE_CONV, BF16),
        tensor_spec(p + "key_projection", cfg.PLE_KEY_PROJ, BF16),
        tensor_spec(p + "value_projection", cfg.PLE_VALUE_PROJ, BF16),
        tensor_spec(p + "norm_conv", cfg.PLE_NORM, BF16),
        tensor_spec(p + "norm_key", cfg.PLE_NORM, BF16),
        tensor_spec(p + "norm_query", cfg.PLE_NORM, BF16),
    )


# --- section builders --------------------------------------------------------

def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (cfg.VOCAB, cfg.HIDDEN), FP8),
    ]
    # Global hyper-connection mixer (Flash-Next has no per-layer or final norm).
    specs.extend(
        (
            tensor_spec("text/hyper_connection_mixer/hc_norm", cfg.HC_NORM, BF16),
            tensor_spec(
                "text/hyper_connection_mixer/input_mix_weight_down",
                cfg.HC_MIX_DOWN,
                BF16,
            ),
            tensor_spec(
                "text/hyper_connection_mixer/input_mix_weight_up",
                cfg.HC_MIX_UP,
                BF16,
            ),
        )
    )

    for layer in range(cfg.NUM_LAYERS):
        prefix = f"text/layers/{layer}/"
        specs.extend(_hyper_connection_specs(prefix, "attn"))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(_full_attention_specs(prefix))
        else:
            specs.extend(_gdn_specs(prefix))
        specs.extend(_hyper_connection_specs(prefix, "mlp"))
        specs.extend(_moe_specs(prefix))
        if layer == PLE_LAYER:
            specs.extend(_ple_specs(prefix))

    specs.append(tensor_spec("text/output_head", (cfg.VOCAB, cfg.HIDDEN), FP8))
    return tuple(specs)


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("mtp/fc_embedding", cfg.MTP_FC, BF16),
        tensor_spec("mtp/fc_hidden", cfg.MTP_FC, BF16),
        tensor_spec("mtp/pre_fc_norm_embedding", cfg.MTP_PRE_NORM_EMBED, BF16),
        tensor_spec("mtp/pre_fc_norm_hidden", cfg.MTP_PRE_NORM_HIDDEN, BF16),
        tensor_spec("mtp/hyper_connection_mixer/hc_norm", cfg.HC_NORM, BF16),
        tensor_spec(
            "mtp/hyper_connection_mixer/input_mix_weight_down", cfg.HC_MIX_DOWN, BF16
        ),
        tensor_spec(
            "mtp/hyper_connection_mixer/input_mix_weight_up", cfg.HC_MIX_UP, BF16
        ),
    ]
    # The single MTP layer is a full-attention + MoE layer (mirrors a main full-attn layer).
    prefix = "mtp/layers/0/"
    specs.extend(_hyper_connection_specs(prefix, "attn"))
    specs.extend(_full_attention_specs(prefix))
    specs.extend(_hyper_connection_specs(prefix, "mlp"))
    specs.extend(_moe_specs(prefix))
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs(cfg.VISION_TEXT_WIDTH)

TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

NVFP4_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == NVFP4)
FP8_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == FP8)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("Flash-Next NVFP4 inventory contains duplicate names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(FP8_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (1235, 34, 333, 1602, 1608, 196, 126, 196):
        raise ValueError(
            "registered Flash-Next NVFP4 inventory is incomplete: "
            f"got (text={len(TEXT_CORE_TENSOR_SPECS)}, mtp={len(MTP_TENSOR_SPECS)}, "
            f"vision={len(VISION_TENSOR_SPECS)}, tensor={len(TENSOR_SPECS)}, "
            f"object={len(OBJECT_SPECS)}, nvfp4={len(NVFP4_TENSOR_SPECS)}, "
            f"fp8={len(FP8_TENSOR_SPECS)}, isd={len(INPUT_SCALE_DIVISOR_SPECS)})"
        )
    if FORMAT_COUNTS != {
        BF16: 901,
        FP32: 268,
        Q4: 54,
        Q5: 54,
        Q6: 1,
        W8: 2,
        NVFP4: 196,
        FP8: 126,
    }:
        raise ValueError(f"unexpected numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 1169,
        ROW_SPLIT_LAYOUT: 111,
        BLOCK_SCALE_LAYOUT: 196,
        ROW_SCALE_LAYOUT: 126,
    }:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FP8",
    "FP8_TENSOR_SPECS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "PLE_LAYER",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SCALE_LAYOUT",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
    "W8",
    "tensor_spec",
    "validate_inventory",
]

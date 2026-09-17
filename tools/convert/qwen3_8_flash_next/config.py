"""Model constants and checkpoint-config contract for Qwen3.8-Flash-Next.

Every number here is taken from the official checkpoint's ``config.json`` and
the safetensors header inventory (``.logs/research_qfn/``).  The converter, the
artifact validator, and the size accounting all import from this module so a
geometry change is made in exactly one place and fails loudly everywhere else.

This module is deliberately weight-free: it never touches a checkpoint file, so
it can be imported and unit-tested before the 360 GB download completes.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- core text geometry -----------------------------------------------------
HIDDEN = 2560
VOCAB = 248320
NUM_LAYERS = 48

# Full-attention every 4th layer (0-indexed): L % 4 == 3 -> [3,7,...,47].
FULL_ATTENTION_INTERVAL = 4
FULL_ATTENTION_LAYERS = tuple(l for l in range(NUM_LAYERS) if l % 4 == 3)
GDN_LAYERS = tuple(l for l in range(NUM_LAYERS) if l not in FULL_ATTENTION_LAYERS)

# --- full-attention (self_attn) geometry ------------------------------------
# q_proj [12288, 2560] = 24 heads x 512; k/v_proj [512, 2560] = 2 kv-heads x 256.
# o_proj [2560, 6144] = 24 heads x 256 head_dim.  q/k_norm are per-head (256).
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
Q_PROJ_OUT = 12288  # NUM_ATTENTION_HEADS * 512
KV_PROJ_OUT = 512  # NUM_KV_HEADS * HEAD_DIM
O_PROJ_OUT = 6144  # NUM_ATTENTION_HEADS * HEAD_DIM
QK_NORM = 256  # HEAD_DIM

# Indexer / QSA (sparse-attention selector) geometry.  Kept BF16: small and
# control-plane.  qk_proj [640, 2560]; q/k layernorm width 128 (indexer head_dim).
INDEXER_QK_PROJ_OUT = 640
INDEXER_NORM = 128

# --- GDN (linear_attn) geometry ---------------------------------------------
# in_proj_qkv [10240, 2560] splits as q 2048 / k 2048 / v 6144; in_proj_z
# [6144, 2560].  The fused query_key_value_z is the vertical stack [16384, 2560].
# a/b projections [48, 2560] each -> fused a_b_projection [96, 2560].
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 48
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
GDN_Q = LINEAR_KEY_HEADS * LINEAR_KEY_DIM  # 2048
GDN_K = LINEAR_KEY_HEADS * LINEAR_KEY_DIM  # 2048
GDN_V = LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM  # 6144
GDN_Z = LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM  # 6144
GDN_QKV_OUT = GDN_Q + GDN_K + GDN_V  # 10240
GDN_QKVZ_OUT = GDN_QKV_OUT + GDN_Z  # 16384
GDN_A = 48
GDN_AB_PROJ_OUT = 96  # GDN_A * 2
GDN_NORM = 128
GDN_CONV = (4, GDN_QKV_OUT)  # conv1d stored channel-major [10240,1,4] -> [4,10240]
GDN_OUT_PROJ = (HIDDEN, GDN_V)  # out_proj [2560, 6144]

# --- MoE geometry ------------------------------------------------------------
NUM_EXPERTS = 512
NUM_EXPERTS_PER_TOK = 10
MOE_INTERMEDIATE = 640  # per-expert gate/up width; shared expert width
SHARED_INTERMEDIATE = 640
ROUTER_OUT = NUM_EXPERTS  # [512, 2560]
# Fused expert matrices (512 experts x per-expert rows, flat expert-major):
EXPERTS_GATE_UP = (NUM_EXPERTS * (2 * MOE_INTERMEDIATE), HIDDEN)  # [655360, 2560]
EXPERTS_DOWN = (NUM_EXPERTS * HIDDEN, MOE_INTERMEDIATE)  # [1310720, 640]
SHARED_GATE_UP = (2 * SHARED_INTERMEDIATE, HIDDEN)  # [1280, 2560]
SHARED_DOWN = (HIDDEN, MOE_INTERMEDIATE)  # [2560, 640]
SHARED_GATE = (1, HIDDEN)  # [1, 2560]

# --- hyper-connection geometry ----------------------------------------------
# The residual stream is hc_count (4) copies of the hidden state -> width 10240.
HC_COUNT = 4
HC_STREAM = HC_COUNT * HIDDEN  # 10240
HC_LOWRANK = 320
HC_BLOCK_INJECT = (HC_COUNT, HC_STREAM)  # [4, 10240]
HC_NORM = (HC_STREAM,)  # [10240]
HC_MIX_DOWN = (HC_LOWRANK, HC_STREAM)  # [320, 10240]
HC_MIX_UP = (HC_STREAM, HC_LOWRANK)  # [10240, 320]

# --- PLE / n-gram geometry ---------------------------------------------------
# The PLE block lives in 0-indexed layer 1 (config ple_layer_ids [2] is 1-based).
PLE_LAYER = 1
PLE_EMBED_DIM = 2560
PLE_KEY_PROJ = (HC_STREAM, HIDDEN)  # key_proj [10240, 2560]
PLE_VALUE_PROJ = (HIDDEN, HIDDEN)  # value_proj [2560, 2560]
PLE_CONV = (4, HC_STREAM)  # conv1d [10240,1,4] -> [4,10240]
PLE_NORM = (HC_STREAM,)  # [10240] x3 (conv/key/query)
NGRAM_SPLIT = 128  # number of embedding shards (= ngram heads x heads_per_ngram)
NGRAM_SHARD = (2500012, 160)  # per-shard shape (rows, cols)
NGRAM_HEADS = 16  # 2560 embed-dim split across 16 heads -> 160 cols
HEADS_PER_NGRAM = 8  # NGRAM_HEADS * HEADS_PER_NGRAM == NGRAM_SPLIT
NGRAM_VOCAB_BASE = 20_000_000
NGRAM_SIZE = 3  # n-gram order (trigram)
NGRAM_CONFIG_SHAPES = {
    "layer_multipliers": (3,),
    "ngram_heads_offsets": (16,),
    "ngram_heads_vocab_sizes": (16,),
}

# --- MTP geometry ------------------------------------------------------------
MTP_LAYERS = 1  # mtp_num_hidden_layers
# MTP layer 0 is a full-attention + MoE layer (no linear_attn).  fc_* are
# [hidden, hidden]; the pre-fc hidden norm operates on the HC stream width.
MTP_FC = (HIDDEN, HIDDEN)
MTP_PRE_NORM_EMBED = (HIDDEN,)
MTP_PRE_NORM_HIDDEN = (HC_STREAM,)

# --- NVFP4 / FP8 profile -----------------------------------------------------
# NVFP4 is reserved for the large weight matrices whose Blackwell GEMM kernels
# it targets (the MoE experts and shared experts).  FP8 row-scale is used for
# the large dense projections (attention q/k/v/o, GDN qkv_z/output, the
# embedding and output head).  Norms, control-plane matrices, the indexer, the
# router, the hyper-connections and the PLE projections stay BF16; GDN a_log /
# dt_bias stay FP32.
NVFP4_WEIGHTS = 1
FP8_E4M3_MAX = 448.0
# Placeholder activation scale for NVFP4 objects.  The runtime computes
# alpha = 1 / (input_scale_divisor * weight_scale_divisor); the activation is
# quantized at runtime with input_scale_divisor.  The Flash-Next MoE/attention
# runtime does not exist yet (BLOCKER), so activation peaks cannot be measured
# and this is a documented placeholder that MUST be recalibrated.  It mirrors
# the canonical NVFP4 numerator and implies an O(1) activation peak.
INPUT_SCALE_DIVISOR_PLACEHOLDER = 2688.0

# Vision: identical ViT to Qwen3.8-27B, so the shared vision inventory builder
# applies with text_width = HIDDEN (fc2 -> [2560, 4608]).
VISION_TEXT_WIDTH = HIDDEN


@dataclass(frozen=True, slots=True)
class FlashNextConfig:
    """The full, validated constant set (single import point for consumers)."""

    hidden: int = HIDDEN
    vocab: int = VOCAB
    num_layers: int = NUM_LAYERS


def is_full_attention(layer: int) -> bool:
    return layer % FULL_ATTENTION_INTERVAL == 3


def validate_config(cfg: dict) -> None:
    """Assert a checkpoint ``config.json`` matches the geometry we convert.

    Accepts the full top-level config and reads the nested ``text_config``.
    Raises ``ValueError`` naming every mismatch so a wrong/different checkpoint
    is rejected before any tensor is read (do not convert an unexpected model).
    """

    if not isinstance(cfg, dict) or not isinstance(cfg.get("text_config"), dict):
        raise ValueError("config.json must contain a text_config object")
    tc = cfg["text_config"]

    def expect(key: str, value: object) -> None:
        actual = tc.get(key)
        if actual != value:
            raise ValueError(
                f"text_config mismatch: {key} expected {value!r}, got {actual!r}"
            )

    expect("hidden_size", HIDDEN)
    expect("vocab_size", VOCAB)
    expect("num_hidden_layers", NUM_LAYERS)
    expect("full_attention_interval", FULL_ATTENTION_INTERVAL)
    expect("num_attention_heads", NUM_ATTENTION_HEADS)
    expect("num_key_value_heads", NUM_KV_HEADS)
    expect("head_dim", HEAD_DIM)
    expect("linear_num_key_heads", LINEAR_KEY_HEADS)
    expect("linear_num_value_heads", LINEAR_VALUE_HEADS)
    expect("linear_key_head_dim", LINEAR_KEY_DIM)
    expect("linear_value_head_dim", LINEAR_VALUE_DIM)
    expect("num_experts", NUM_EXPERTS)
    expect("num_experts_per_tok", NUM_EXPERTS_PER_TOK)
    expect("moe_intermediate_size", MOE_INTERMEDIATE)
    expect("shared_expert_intermediate_size", SHARED_INTERMEDIATE)
    expect("ple_embed_dim", PLE_EMBED_DIM)
    expect("mtp_num_hidden_layers", MTP_LAYERS)
    expect("tie_word_embeddings", False)
    expect("max_position_embeddings", 262144)
    expect("split_ngram_parts", NGRAM_SPLIT)
    expect("heads_per_ngram", HEADS_PER_NGRAM)
    expect("ple_layer_ids", [PLE_LAYER + 1])  # config is 1-based

    # Cross-check the explicit layer_types list against the interval rule.
    layer_types = tc.get("layer_types")
    if isinstance(layer_types, list) and len(layer_types) == NUM_LAYERS:
        for layer in range(NUM_LAYERS):
            expect_full = "full_attention" if is_full_attention(layer) else "linear_attention"
            if layer_types[layer] != expect_full:
                raise ValueError(
                    f"layer_types[{layer}] = {layer_types[layer]!r}, "
                    f"expected {expect_full!r}"
                )

    # internal-consistency invariants (catch a corrupted config)
    if Q_PROJ_OUT != NUM_ATTENTION_HEADS * (2 * HEAD_DIM):
        raise ValueError("Q_PROJ_OUT is inconsistent with head geometry")
    if KV_PROJ_OUT != NUM_KV_HEADS * HEAD_DIM:
        raise ValueError("KV_PROJ_OUT is inconsistent with head geometry")
    if O_PROJ_OUT != NUM_ATTENTION_HEADS * HEAD_DIM:
        raise ValueError("O_PROJ_OUT is inconsistent with head geometry")
    if GDN_QKV_OUT != 10240 or GDN_QKVZ_OUT != 16384:
        raise ValueError("GDN projection geometry is inconsistent")
    if HC_STREAM != HC_COUNT * HIDDEN:
        raise ValueError("hyper-connection stream width is inconsistent")
    if NGRAM_SPLIT != NGRAM_HEADS * HEADS_PER_NGRAM:
        raise ValueError("ngram split is inconsistent with head counts")
    if EXPERTS_GATE_UP != (NUM_EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN):
        raise ValueError("fused expert gate_up geometry is inconsistent")
    if len(FULL_ATTENTION_LAYERS) != NUM_LAYERS // FULL_ATTENTION_INTERVAL:
        raise ValueError("full-attention layer count is inconsistent")
    # NVFP4 geometry constraints on the fused expert matrices.
    for name, (n, k) in (("experts_gate_up", EXPERTS_GATE_UP),
                         ("experts_down", EXPERTS_DOWN),
                         ("shared_gate_up", SHARED_GATE_UP),
                         ("shared_down", SHARED_DOWN)):
        if n % 128 or k % 64:
            raise ValueError(f"{name} {name} is not NVFP4-shaped (N%128/K%64)")


__all__ = [
    "FlashNextConfig",
    "INPUT_SCALE_DIVISOR_PLACEHOLDER",
    "NGRAM_HEADS",
    "NGRAM_SHARD",
    "NGRAM_SPLIT",
    "is_full_attention",
    "validate_config",
]

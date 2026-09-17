"""P0: the canonical expected inventory lists every logical parent of Flash-Next.

Label: flash_next_P0.
"""

from __future__ import annotations

from tools.convert.qwen3_8_flash_next import config as cfg
from tools.convert.qwen3_8_flash_next.inventory_nvfp4 import (
    BF16,
    FP32,
    MODEL_ID,
    NVFP4,
    OBJECT_SPECS,
    TARGET_KEY,
    TENSOR_SPECS,
    WEIGHTS_ID,
    validate_inventory,
)

BY_NAME = {spec.name: spec for spec in TENSOR_SPECS}
NAMES = frozenset(BY_NAME)
TEXT_NAMES = frozenset(name for name in NAMES if name.startswith("text/"))


def test_identity_constants():
    assert MODEL_ID == "qwen3.8-flash-next"
    assert WEIGHTS_ID == "nvfp4"
    assert TARGET_KEY == "qwen3_8_flash_next"
    # STOP guard: the artifact identity must never be a registered 27B/35B key.
    assert not MODEL_ID.startswith(("qwen3.8-27b", "qwen3.6"))


def test_validate_inventory_passes():
    validate_inventory()


def test_every_logical_parent_present():
    assert "text/token_embedding" in NAMES
    assert "text/output_head" in NAMES
    assert "text/hyper_connection_mixer/hc_norm" in NAMES
    for layer in range(cfg.NUM_LAYERS):
        p = f"text/layers/{layer}/"
        # per-layer norms / hyper-connections (separate attn + mlp HC blocks)
        assert p + "attn_hyper_connection/block_inject_weight" in NAMES
        assert p + "attn_hyper_connection/hc_norm" in NAMES
        assert p + "mlp_hyper_connection/hc_norm" in NAMES
        # MoE: router, shared expert, 512 fused experts
        assert p + "mlp/experts/routing" in NAMES
        assert p + "mlp/shared_expert_gate" in NAMES
        assert p + "mlp/experts/gate_up" in NAMES
        assert p + "mlp/experts/down" in NAMES
        if layer in cfg.FULL_ATTENTION_LAYERS:
            assert p + "attention/query" in NAMES
            assert p + "attention/query_norm" in NAMES
            assert p + "attention/indexer/qk_proj" in NAMES
        else:
            assert p + "gdn/query_key_value_z" in NAMES
            assert p + "gdn/a_log" in NAMES
            assert p + "gdn/dt_bias" in NAMES


def test_attention_layer_split_is_36_gdn_12_full():
    gdn = sum(1 for n in TEXT_NAMES if n.endswith("/gdn/query_key_value_z"))
    full = sum(1 for n in TEXT_NAMES if n.endswith("/attention/query"))
    assert gdn == 36
    assert full == 12


def test_ple_lives_only_in_layer_1():
    assert f"text/layers/{cfg.PLE_LAYER}/ple/key_projection" in NAMES
    assert f"text/layers/{cfg.PLE_LAYER}/ple/convolution" in NAMES
    for layer in range(cfg.NUM_LAYERS):
        if layer == cfg.PLE_LAYER:
            continue
        assert not any(
            name.startswith(f"text/layers/{layer}/ple/") for name in NAMES
        )


def test_experts_are_512_fused_nvfp4():
    for layer in range(cfg.NUM_LAYERS):
        p = f"text/layers/{layer}/mlp/"
        assert BY_NAME[p + "experts/gate_up"].shape == cfg.EXPERTS_GATE_UP
        assert BY_NAME[p + "experts/gate_up"].format == NVFP4
        assert BY_NAME[p + "experts/down"].shape == cfg.EXPERTS_DOWN
        assert BY_NAME[p + "experts/down"].format == NVFP4
        assert BY_NAME[p + "experts/routing"].shape == (cfg.NUM_EXPERTS, cfg.HIDDEN)
        assert BY_NAME[p + "experts/routing"].format == BF16
        # each NVFP4 expert weight carries its FP32 input_scale_divisor
        assert BY_NAME[p + "experts/gate_up_projection/input_scale_divisor"].format == FP32
        assert BY_NAME[p + "experts/down_projection/input_scale_divisor"].format == FP32


def test_counts_are_locked():
    assert len(TENSOR_SPECS) == 1602
    assert len(OBJECT_SPECS) == 1608
    text = sum(1 for s in TENSOR_SPECS if s.name.startswith("text/"))
    mtp = sum(1 for s in TENSOR_SPECS if s.name.startswith("mtp/"))
    vision = sum(1 for s in TENSOR_SPECS if not s.name.startswith(("text/", "mtp/")))
    assert (text, mtp, vision) == (1235, 34, 333)

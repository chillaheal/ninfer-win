"""Closed single-source recipe for the Qwen3.8-Flash-Next NVFP4 artifact.

Unlike ``qwen3_8_27b`` (which repacks an already-quantized compressed-tensors
checkpoint), Qwen3.8-Flash-Next ships only a BF16 checkpoint -- there is no
official NVFP4 artifact to repack.  Every quantized object is therefore built
here by quantizing the BF16 source on the fly:

  * NVFP4 weight matrices   -> ``quantize_nvfp4``    -> ``encode_nvfp4``
  * FP8 row-scale matrices  -> ``quantize_fp8_row``  -> ``encode_fp8_row_scaled``
  * direct matrices (BF16 /
    FP32) and row-split
    vision (Q4/Q5/Q6/W8)    -> ``encode_direct`` /
                               ``encode_tensor_payload``

The PLE n-gram embedding (128 shards) and the three 63-bit I64 PLE config
tensors are *not* representable in the ``.ninfer`` container and live in the
separate ``.ngram`` store (see ``ngram_store``), so they are excluded from this
recipe.

Every non-divisor artifact object maps to exactly one ``TensorRecipe`` whose
expression produces the logical BF16 matrix (or the direct tensor for norm /
conv / cast objects).  For a quantized object the expression yields the
``[N, K]`` BF16 weight that the encoder then quantizes, so the expression shape
equals the inventory ``TensorSpec.shape`` and ``validate_recipe_coverage``
applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe

from . import config as cfg
from . import inventory_nvfp4 as inventory
from . import quantize_fp8_row
from . import quantize_nvfp4


# Local aliases keep the recipe readable without the ``family_recipe.`` prefix.
source = family_recipe.source
Slice = family_recipe.Slice
Reshape = family_recipe.Reshape
Transpose = family_recipe.Transpose
Concat = family_recipe.Concat
Cast = family_recipe.Cast
TensorRecipe = family_recipe.TensorRecipe
Expression = family_recipe.Expression

DIVISOR_NAMES = frozenset(spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS)


def _source(name: str, shape: tuple[int, ...]) -> Expression:
    return source(name, shape)


def _conv_expression(src_name: str, channels: int) -> Expression:
    """Squeeze a ``[C, 1, 4]`` causal conv1d weight to the ``[4, C]`` layout.

    The checkpoint stores the depthwise causal conv as ``[C, 1, 4]`` (a
    size-1 inner dimension); the artifact wants ``[4, C]`` (kernel width
    first).  No ``Slice`` is needed -- the size-1 axis is dropped by the
    reshape, then the two axes are swapped.
    """

    return Transpose(Reshape(_source(src_name, (channels, 1, 4)), (channels, 4)), (1, 0))


def _hyper_recipes(obj_prefix: str, src_prefix: str) -> list[TensorRecipe]:
    """The 8 hyper-connection objects shared by attention and MLP blocks."""

    recipes: list[TensorRecipe] = []
    for block in ("attn_hyper_connection", "mlp_hyper_connection"):
        recipes.append(
            TensorRecipe(
                obj_prefix + block + "/block_inject_weight",
                _source(
                    src_prefix + block + ".block_inject_weight.weight",
                    cfg.HC_BLOCK_INJECT,
                ),
            )
        )
        recipes.append(
            TensorRecipe(
                obj_prefix + block + "/hc_norm",
                _source(src_prefix + block + ".hc_norm.weight", cfg.HC_NORM),
            )
        )
        recipes.append(
            TensorRecipe(
                obj_prefix + block + "/input_mix_weight_down",
                _source(
                    src_prefix + block + ".input_mix_weight_down.weight",
                    cfg.HC_MIX_DOWN,
                ),
            )
        )
        recipes.append(
            TensorRecipe(
                obj_prefix + block + "/input_mix_weight_up",
                _source(
                    src_prefix + block + ".input_mix_weight_up.weight",
                    cfg.HC_MIX_UP,
                ),
            )
        )
    return recipes


def _gdn_recipes(obj_prefix: str, src_prefix: str) -> list[TensorRecipe]:
    """The 7 GDN (gated DeltaNet linear-attention) objects of a GDN layer."""

    s = src_prefix + "linear_attn."
    return [
        TensorRecipe(
            obj_prefix + "gdn/a_log",
            Cast(_source(s + "A_log", (cfg.GDN_A,)), inventory.FP32),
        ),
        TensorRecipe(
            obj_prefix + "gdn/dt_bias",
            Cast(_source(s + "dt_bias", (cfg.GDN_A,)), inventory.FP32),
        ),
        TensorRecipe(
            obj_prefix + "gdn/convolution",
            _conv_expression(s + "conv1d.weight", cfg.GDN_QKV_OUT),
        ),
        TensorRecipe(
            obj_prefix + "gdn/a_b_projection",
            Concat(
                (
                    _source(s + "in_proj_a.weight", (cfg.GDN_A, cfg.HIDDEN)),
                    _source(s + "in_proj_b.weight", (cfg.GDN_A, cfg.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            obj_prefix + "gdn/query_key_value_z",
            Concat(
                (
                    _source(
                        s + "in_proj_qkv.weight", (cfg.GDN_QKV_OUT, cfg.HIDDEN)
                    ),
                    _source(s + "in_proj_z.weight", (cfg.GDN_Z, cfg.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            obj_prefix + "gdn/norm",
            _source(s + "norm.weight", (cfg.GDN_NORM,)),
        ),
        TensorRecipe(
            obj_prefix + "gdn/output",
            _source(s + "out_proj.weight", cfg.GDN_OUT_PROJ),
        ),
    ]


def _full_attention_recipes(obj_prefix: str, src_prefix: str) -> list[TensorRecipe]:
    """The 9 full-attention objects (Q/K/V/O + norms + indexer) of a full-attn layer."""

    s = src_prefix + "self_attn."
    return [
        TensorRecipe(
            obj_prefix + "attention/query",
            _source(s + "q_proj.weight", (cfg.Q_PROJ_OUT, cfg.HIDDEN)),
        ),
        TensorRecipe(
            obj_prefix + "attention/key",
            _source(s + "k_proj.weight", (cfg.KV_PROJ_OUT, cfg.HIDDEN)),
        ),
        TensorRecipe(
            obj_prefix + "attention/value",
            _source(s + "v_proj.weight", (cfg.KV_PROJ_OUT, cfg.HIDDEN)),
        ),
        TensorRecipe(
            obj_prefix + "attention/output",
            _source(s + "o_proj.weight", cfg.GDN_OUT_PROJ),
        ),
        TensorRecipe(
            obj_prefix + "attention/query_norm",
            _source(s + "q_norm.weight", (cfg.QK_NORM,)),
        ),
        TensorRecipe(
            obj_prefix + "attention/key_norm",
            _source(s + "k_norm.weight", (cfg.QK_NORM,)),
        ),
        TensorRecipe(
            obj_prefix + "attention/indexer/qk_proj",
            _source(s + "indexer.index_qk_proj.weight", (cfg.INDEXER_QK_PROJ_OUT, cfg.HIDDEN)),
        ),
        TensorRecipe(
            obj_prefix + "attention/indexer/k_norm",
            _source(s + "indexer.k_layernorm.weight", (cfg.INDEXER_NORM,)),
        ),
        TensorRecipe(
            obj_prefix + "attention/indexer/q_norm",
            _source(s + "indexer.q_layernorm.weight", (cfg.INDEXER_NORM,)),
        ),
    ]


def _moe_recipes(obj_prefix: str, src_prefix: str) -> list[TensorRecipe]:
    """The 6 routed-MoE objects (experts + shared expert + router) of every layer."""

    s = src_prefix + "mlp."
    return [
        TensorRecipe(
            obj_prefix + "mlp/experts/gate_up",
            Reshape(
                _source(
                    s + "experts.gate_up_proj",
                    (
                        cfg.NUM_EXPERTS,
                        2 * cfg.MOE_INTERMEDIATE,
                        cfg.HIDDEN,
                    ),
                ),
                cfg.EXPERTS_GATE_UP,
            ),
        ),
        TensorRecipe(
            obj_prefix + "mlp/experts/down",
            Reshape(
                _source(
                    s + "experts.down_proj",
                    (cfg.NUM_EXPERTS, cfg.HIDDEN, cfg.MOE_INTERMEDIATE),
                ),
                cfg.EXPERTS_DOWN,
            ),
        ),
        TensorRecipe(
            obj_prefix + "mlp/experts/routing",
            _source(s + "gate.weight", (cfg.ROUTER_OUT, cfg.HIDDEN)),
        ),
        TensorRecipe(
            obj_prefix + "mlp/shared_expert/gate_up",
            Concat(
                (
                    _source(
                        s + "shared_expert.gate_proj.weight",
                        (cfg.SHARED_INTERMEDIATE, cfg.HIDDEN),
                    ),
                    _source(
                        s + "shared_expert.up_proj.weight",
                        (cfg.SHARED_INTERMEDIATE, cfg.HIDDEN),
                    ),
                ),
                0,
            ),
        ),
        TensorRecipe(
            obj_prefix + "mlp/shared_expert/down",
            _source(s + "shared_expert.down_proj.weight", cfg.SHARED_DOWN),
        ),
        TensorRecipe(
            obj_prefix + "mlp/shared_expert_gate",
            _source(s + "shared_expert_gate.weight", cfg.SHARED_GATE),
        ),
    ]


def _ple_recipes(obj_prefix: str, src_prefix: str) -> list[TensorRecipe]:
    """The 6 BF16 PLE objects of the PLE layer (its embedding goes to ``.ngram``)."""

    s = src_prefix + "ple."
    return [
        TensorRecipe(
            obj_prefix + "ple/convolution",
            _conv_expression(s + "conv1d.weight", cfg.PLE_NORM[0]),
        ),
        TensorRecipe(
            obj_prefix + "ple/key_projection",
            _source(s + "key_proj.weight", cfg.PLE_KEY_PROJ),
        ),
        TensorRecipe(
            obj_prefix + "ple/value_projection",
            _source(s + "value_proj.weight", cfg.PLE_VALUE_PROJ),
        ),
        TensorRecipe(
            obj_prefix + "ple/norm_conv",
            _source(s + "norm_conv.weight", cfg.PLE_NORM),
        ),
        TensorRecipe(
            obj_prefix + "ple/norm_key",
            _source(s + "norm_key.weight", cfg.PLE_NORM),
        ),
        TensorRecipe(
            obj_prefix + "ple/norm_query",
            _source(s + "norm_query.weight", cfg.PLE_NORM),
        ),
    ]


def _text_top_recipes() -> list[TensorRecipe]:
    return [
        TensorRecipe(
            "text/token_embedding",
            _source(
                "model.language_model.embed_tokens.weight", (cfg.VOCAB, cfg.HIDDEN)
            ),
        ),
        TensorRecipe(
            "text/hyper_connection_mixer/hc_norm",
            _source(
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                cfg.HC_NORM,
            ),
        ),
        TensorRecipe(
            "text/hyper_connection_mixer/input_mix_weight_down",
            _source(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                cfg.HC_MIX_DOWN,
            ),
        ),
        TensorRecipe(
            "text/hyper_connection_mixer/input_mix_weight_up",
            _source(
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                cfg.HC_MIX_UP,
            ),
        ),
        TensorRecipe(
            "text/output_head",
            _source("lm_head.weight", (cfg.VOCAB, cfg.HIDDEN)),
        ),
    ]


def _text_layer_recipes(layer: int) -> list[TensorRecipe]:
    obj = f"text/layers/{layer}/"
    src = f"model.language_model.layers.{layer}."
    recipes = _hyper_recipes(obj, src)
    if cfg.is_full_attention(layer):
        recipes.extend(_full_attention_recipes(obj, src))
    else:
        recipes.extend(_gdn_recipes(obj, src))
    recipes.extend(_moe_recipes(obj, src))
    if layer == cfg.PLE_LAYER:
        recipes.extend(_ple_recipes(obj, src))
    return recipes


def _mtp_top_recipes() -> list[TensorRecipe]:
    s = "mtp."
    recipes = [
        TensorRecipe("mtp/fc_embedding", _source(s + "fc_embedding.weight", cfg.MTP_FC)),
        TensorRecipe("mtp/fc_hidden", _source(s + "fc_hidden.weight", cfg.MTP_FC)),
        TensorRecipe(
            "mtp/pre_fc_norm_embedding",
            _source(s + "pre_fc_norm_embedding.weight", cfg.MTP_PRE_NORM_EMBED),
        ),
        TensorRecipe(
            "mtp/pre_fc_norm_hidden",
            _source(s + "pre_fc_norm_hidden.weight", cfg.MTP_PRE_NORM_HIDDEN),
        ),
    ]
    for suffix, shape in (
        ("hc_norm", cfg.HC_NORM),
        ("input_mix_weight_down", cfg.HC_MIX_DOWN),
        ("input_mix_weight_up", cfg.HC_MIX_UP),
    ):
        recipes.append(
            TensorRecipe(
                f"mtp/hyper_connection_mixer/{suffix}",
                _source(f"mtp.hyper_connection_mixer.{suffix}.weight", shape),
            )
        )
    return recipes


def _mtp_layer_recipes() -> list[TensorRecipe]:
    """MTP layer 0 is a full-attention layer with a routed MoE (no GDN, no PLE)."""

    obj = "mtp/layers/0/"
    src = "mtp.layers.0."
    recipes = _hyper_recipes(obj, src)
    recipes.extend(_full_attention_recipes(obj, src))
    recipes.extend(_moe_recipes(obj, src))
    return recipes


def _build_all() -> dict[str, TensorRecipe]:
    recipes: dict[str, TensorRecipe] = {}

    def add(items: list[TensorRecipe]) -> None:
        for recipe in items:
            if recipe.object_name in recipes:
                raise ValueError(f"duplicate recipe for {recipe.object_name}")
            recipes[recipe.object_name] = recipe

    add(_text_top_recipes())
    for layer in range(cfg.NUM_LAYERS):
        add(_text_layer_recipes(layer))
    add(_mtp_top_recipes())
    add(_mtp_layer_recipes())
    add(list(family_recipe.build_vision_recipes(cfg.HIDDEN)))
    return recipes


RECIPES_BY_NAME = _build_all()
NON_DIVISOR_SPECS = tuple(
    spec for spec in inventory.TENSOR_SPECS if spec.name not in DIVISOR_NAMES
)
# Inventory order is the artifact order; reindex the recipes to match exactly.
RECIPES = tuple(RECIPES_BY_NAME[spec.name] for spec in NON_DIVISOR_SPECS)
RECIPE_COUNT = len(RECIPES)


@dataclass(frozen=True, slots=True)
class RecipePreflight:
    recipe_count: int
    source_tensor_count: int
    source_shard_count: int
    source_dtype_counts: Mapping[str, int]


def validate_recipe() -> None:
    """Order / shape / uniqueness coverage plus source-name closure."""

    family_recipe.validate_recipe_coverage(RECIPES, NON_DIVISOR_SPECS)

    # Every non-divisor object must be owned exactly once; divisors have no recipe.
    if set(RECIPES_BY_NAME) != {spec.name for spec in NON_DIVISOR_SPECS}:
        raise ValueError("recipe set does not match the non-divisor inventory")

    # The recipe count must be the full inventory minus the placeholder divisors.
    if RECIPE_COUNT != len(inventory.TENSOR_SPECS) - len(DIVISOR_NAMES):
        raise ValueError("recipe count does not cover the non-divisor inventory")

    # Vision is the checkpoint-invariant family block, untouched here.
    if len(family_recipe.build_vision_recipes(cfg.HIDDEN)) != sum(
        1 for spec in NON_DIVISOR_SPECS if spec.name.startswith("vision/")
    ):
        raise ValueError("vision recipe count does not match the inventory")

    # Source names must be closed: every expression leaf names a real checkpoint
    # tensor (verified against the header table by preflight_source_reader).


SOURCE_REQUIREMENTS = family_recipe.source_requirements(RECIPES)


def preflight_source_reader(reader) -> family_recipe.SourcePreflight:
    """Verify every recipe source exists with the declared shape and dtype.

    Works against a real ``ShardReader`` (conversion) or a
    ``ShapeTableReader`` (dry preflight over the safetensors headers).
    """

    return family_recipe.preflight_source_reader(reader, RECIPES)


def materialize(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    device: str | torch.device,
) -> torch.Tensor:
    """Materialize one non-divisor object as a BF16/FP32 tensor on *device*.

    Returns the logical tensor only (not yet encoded).  For a quantized object
    this is the ``[N, K]`` BF16 weight that ``encode`` then quantizes.
    """

    recipe = RECIPES_BY_NAME[spec.name]
    tensor = family_recipe.materialize_recipe(recipe, reader)
    expected = family_recipe.expression_shape(recipe.expression)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {expected}"
        )
    return tensor.to(device)


def quantize_nvfp4_object(spec: inventory.TensorSpec, bf16: torch.Tensor):
    """Quantize a materialized ``[N, K]`` BF16 matrix to the NVFP4 triple."""

    return quantize_nvfp4.quantize_nvfp4(bf16)


def quantize_fp8_object(spec: inventory.TensorSpec, bf16: torch.Tensor):
    """Quantize a materialized ``[N, K]`` BF16 matrix to the FP8 row-scale pair."""

    return quantize_fp8_row.quantize_fp8_row(bf16)


validate_recipe()


__all__ = [
    "DIVISOR_NAMES",
    "NON_DIVISOR_SPECS",
    "RECIPES",
    "RECIPES_BY_NAME",
    "RECIPE_COUNT",
    "SOURCE_REQUIREMENTS",
    "RecipePreflight",
    "materialize",
    "preflight_source_reader",
    "quantize_fp8_object",
    "quantize_nvfp4_object",
    "validate_recipe",
]

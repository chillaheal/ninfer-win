"""BF16/float -> FP8_E4M3FN row-scale quantizer for NInfer row-scale kernels.

The bit-exact contract comes from the runtime layout (``tools/artifact/
layouts.py::dequantize_fp8_row_scaled``), which reconstructs a matrix as

      W[i, j] ~= decode_e4m3fn(code[i, j]) * row_scale[i]

where ``code`` is an E4M3FN byte and ``row_scale`` is a binary16 (BF16)
multiplier, one per row.  We therefore store, per row, a scale chosen so the
row's peak magnitude maps to the top of the E4M3FN range (448.0) and integer
E4M3FN codes for the rest::

      scale   = round_bf16(row_absmax / 448.0)
      codes   = e4m3fn_satfinite(row / scale)        (scale = 0 -> all-zero row)

This is a pure weight transform; the stored bytes are consumed directly, the
runtime never re-quantizes the weight.  This module produces the logical
``(code_words[N,K] uint8, row_scales[N] bf16)`` pair that
``layouts.encode_fp8_row_scaled`` packs, plus a matching dequantizer for
round-trip tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.convert.qwen3_8_flash_next.quantize_nvfp4 import (
    _CHUNK_ELEMENTS,
    _row_chunks,
)

# E4M3FN max finite magnitude (0x7E) and NaN word (0x7F / 0xFF).
_E4M3_MAX = 448.0
_NAN_WORD = 0x7F


@dataclass(frozen=True, slots=True)
class Fp8RowQuantized:
    """The logical FP8 row-scale pair that layouts.encode_fp8_row_scaled packs."""

    code_words: torch.Tensor  # uint8 [N, K], E4M3FN words
    row_scales: torch.Tensor  # bfloat16 [N]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.code_words.shape)


def quantize_fp8_row(weight: torch.Tensor) -> Fp8RowQuantized:
    """Quantize a logical ``[N, K]`` float/BF16 matrix to row-scaled E4M3FN.

    No geometry constraint (row-scale layout accepts any positive ``N, K``).

    The quantization runs in memory-bounded row chunks: the scale is per-row,
    so each row's scale and codes depend only on that row -- chunking is
    bit-exact and keeps a wide tensor's f32 temporaries bounded.
    """

    if weight.dim() != 2:
        raise ValueError(
            f"FP8 row-scale quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    n, k = weight.shape

    src = weight.detach()
    code_words = torch.empty((n, k), dtype=torch.uint8, device=src.device)
    row_scales = torch.empty((n,), dtype=torch.bfloat16, device=src.device)

    for r0, r1 in _row_chunks(n, k, _CHUNK_ELEMENTS):
        w_c = src[r0:r1].to(torch.float32)
        if not torch.isfinite(w_c).all():
            raise ValueError("FP8 source contains NaN or infinity")
        rows = r1 - r0
        row_absmax_c = w_c.abs().amax(dim=1)  # [rows]
        scale_f32_c = row_absmax_c / _E4M3_MAX
        row_scales[r0:r1] = scale_f32_c.to(torch.bfloat16)
        scale_c = row_scales[r0:r1].float()  # [rows]
        # Per-row division; zero-scale rows (all-zero input) map to zero codes.
        inv_c = torch.where(scale_c > 0, 1.0 / scale_c, 0.0).unsqueeze(1)  # [rows, 1]
        norm_c = w_c * inv_c  # [rows, K]; zero where scale == 0
        codes_c = norm_c.clamp_min(-_E4M3_MAX).clamp_max(_E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        code_words[r0:r1] = codes_c.view(torch.uint8)

    # Sanity: every code must be a finite E4M3FN word (no NaN 0x7F/0xFF).
    if bool(((code_words & 0x7F) == _NAN_WORD).any()):
        raise ValueError("produced an invalid E4M3FN code word")
    # A zero row scale must carry only signed-zero codes (the layout enforces).
    zero_scale = (row_scales.view(torch.int16).to(torch.int32) == 0)
    if bool((zero_scale.unsqueeze(1) & (code_words & 0x7F) != 0).any()):
        raise ValueError("a zero row scale requires only signed-zero codes")

    return Fp8RowQuantized(code_words=code_words, row_scales=row_scales)


def dequantize_fp8_row(qz: Fp8RowQuantized) -> torch.Tensor:
    """Reconstruct the logical ``[N, K]`` float32 matrix a kernel would dequantize."""

    code_f = qz.code_words.view(torch.float8_e4m3fn).float()  # [N, K]
    scale = qz.row_scales.float().unsqueeze(1)  # [N, 1]
    return code_f * scale


__all__ = [
    "Fp8RowQuantized",
    "dequantize_fp8_row",
    "quantize_fp8_row",
]

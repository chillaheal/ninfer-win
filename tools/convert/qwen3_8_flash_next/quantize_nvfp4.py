"""BF16/float -> NVFP4 quantizer for NInfer Blackwell kernels.

This is the *from-scratch* quantization path.  The 27B converter repacks an
already-quantized compressed-tensors checkpoint; Qwen3.8-Flash-Next has no such
official checkpoint, so we quantize the BF16 source ourselves.

The bit-exact contract comes from the runtime, not from us:

  src/ops/linear/nvfp4/nvfp4_codec.cuh::quantize_nvfp4_k16
      scale_unencoded = input_scale_divisor * max_abs / 6.0
      scale_byte      = cvt.satfinite.e4m3(scale_unencoded)
      decoded         = decode(scale_byte)
      code            = cvt.rn.satfinite.e2m1(value * input_scale_divisor / decoded)

  src/ops/linear/nvfp4/nvfp4_w4a4.cu
      alpha = 1.0f / (input_scale_divisor * weight_scale_divisor)

So a weight tensor is quantized with ``weight_scale_divisor`` (the FP32 word
stored at the tail of the NVFP4 payload, see layouts.encode_nvfp4).  The
dequantization the kernel reconstructs is

      W[i,j] ~= decode_e2m1(code[i,j]) * decoded_scale[i, j/16] / weight_scale_divisor

``weight_scale_divisor`` is a pure normalization constant -- it cancels in the
ideal dequant -- but it fixes the range of the E4M3FN block scales, so we pick
the canonical NVFP4 value  ``2688.0 / global_max(|W|)`` ( = 448*6/max ), which
keeps every block scale finite and uses the full E2M1 dynamic range (max 6.0).

The E2M1 weights and E4M3FN scales are packed by layouts.encode_nvfp4; this
module only produces the logical (packed-codes, natural-scales, divisor) triple
that encode_nvfp4 consumes, plus a matching dequantizer for round-trip tests.
"""

from __future__ import annotations

import math
import struct
import typing
from dataclasses import dataclass

import torch

from tools.artifact.numeric import (
    decode_e2m1_word,
    valid_positive_fp32_word,
)

# E4M3FN max finite magnitude (0x7E) and E2M1 max magnitude (6.0).
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
# 448.0 * 6.0 -- the numerator of the canonical weight divisor.
_GLOBAL_SCALE_NUM = _E4M3_MAX * _E2M1_MAX
# Bounded row-chunk size for quantization (elements = rows x K). Keeps the
# live f32 temporaries to a few hundred MB so a 1.7B-element expert weight
# does not need multi-GiB contiguous blocks. Bit-exact (see quantize_nvfp4).
_CHUNK_ELEMENTS = 16_000_000


@dataclass(frozen=True, slots=True)
class Nvfp4Quantized:
    """The logical NVFP4 triple that layouts.encode_nvfp4 packs to bytes."""

    packed_codes: torch.Tensor  # uint8 [N, K // 2], even index -> low nibble
    natural_scales: torch.Tensor  # uint8 [N, K // 16] E4M3FN words
    weight_scale_divisor: bytes  # one little-endian FP32 word (4 bytes)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.packed_codes.shape[0], self.packed_codes.shape[1] * 2)


def weight_scale_divisor_word(global_max_abs: float) -> int:
    """Canonical NVFP4 weight divisor for a tensor whose |W| peaks at *global_max_abs*."""

    if not math.isfinite(global_max_abs) or global_max_abs <= 0.0:
        raise ValueError("global_max_abs must be finite and positive")
    divisor = _GLOBAL_SCALE_NUM / global_max_abs
    word = struct.unpack("<I", struct.pack("<f", divisor))[0]
    if not valid_positive_fp32_word(word):
        raise ValueError("computed weight divisor is not a finite positive FP32 word")
    return word


def _e2m1_codes(norm: torch.Tensor) -> torch.Tensor:
    """Round *norm* (finite float, any range) to E2M1 code words 0..15.

    Mirrors ``cvt.rn.satfinite.e2m1x2.f32``: saturate to [-6, 6] then round to
    the nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6}, tie toward the lower code.
    """

    mag = norm.abs().clamp_max(_E2M1_MAX)
    # Nearest-level boundaries; torch.bucketize (right=False) is rightmost
    # boundary < value, so an exact boundary rounds to the lower level.
    boundaries = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=norm.dtype, device=norm.device
    )
    idx = torch.bucketize(mag, boundaries)  # 0..7, int64
    sign = (norm < 0).to(torch.int64)
    words = (idx | (sign << 3)).to(torch.uint8)  # sign is E2M1 bit 3, magnitude bits 0..2
    return words


def _pack_nibbles(codes: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Pack ``codes`` [N, K] (0..15) to uint8 [N, K // 2]; even -> low nibble."""

    if codes.dtype != torch.uint8:
        raise TypeError("codes must be uint8 E2M1 words")
    flat = codes.reshape(n, k)
    low = flat[:, 0::2].to(torch.int16)
    high = flat[:, 1::2].to(torch.int16)
    return (low | (high << 4)).to(torch.uint8)


def _row_chunks(
    n: int, k: int, budget_elements: int
) -> typing.Generator[tuple[int, int], None, None]:
    """Yield whole-row ranges ``(r0, r1)`` whose f32 footprint stays bounded.

    NVFP4 quantization is per-row and per-16-group with a single global
    divisor, so quantizing whole rows in bounded chunks is bit-exact: every
    emitted scale/code word depends only on its own row plus the divisor.
    """

    rows = max(1, budget_elements // k)
    r0 = 0
    while r0 < n:
        r1 = min(r0 + rows, n)
        yield r0, r1
        r0 = r1


def quantize_nvfp4(
    weight: torch.Tensor,
    *,
    weight_scale_divisor: float | int | bytes | None = None,
) -> Nvfp4Quantized:
    """Quantize a logical ``[N, K]`` float/BF16 matrix to NVFP4.

    *weight_scale_divisor* may be given as a float/int/4-byte word; if omitted
    the canonical ``2688.0 / global_max(|W|)`` is used.  Requires ``N % 128 == 0``
    and ``K % 64 == 0`` (the blockscale layout constraint).

    The quantization runs in memory-bounded row chunks: only one chunk's f32
    temporaries are live at a time, so a 1.7B-element expert weight peaks at a
    few hundred MB instead of several multi-GiB f32 copies.  Bit-exact with the
    whole-tensor path (per-group scales + a single global divisor).
    """

    if weight.dim() != 2:
        raise ValueError(f"NVFP4 quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    n, k = weight.shape
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            f"NVFP4 requires N%128==0 and K%64==0, got N={n}, K={k}"
        )

    src = weight.detach()

    # Pass 1 (bounded): validate finiteness and take the global max(|W|).
    # bf16/f16/f32 promote to f32 exactly, so the running max over chunks equals
    # the whole-tensor max(|W|) bit-for-bit.
    gmax = 0.0
    need_gmax = weight_scale_divisor is None
    for r0, r1 in _row_chunks(n, k, _CHUNK_ELEMENTS):
        chunk = src[r0:r1].to(torch.float32)
        if not torch.isfinite(chunk).all():
            raise ValueError("NVFP4 source contains NaN or infinity")
        if need_gmax:
            gmax = max(gmax, float(chunk.abs().max().item()))

    if weight_scale_divisor is None:
        if gmax <= 0.0:
            # All-zero matrix: any positive divisor is fine.
            word = struct.unpack("<I", struct.pack("<f", 1.0))[0]
        else:
            word = weight_scale_divisor_word(gmax)
        divisor = float(struct.unpack("<f", struct.pack("<I", word))[0])
        divisor_bytes = struct.pack("<I", word)
    else:
        if isinstance(weight_scale_divisor, (bytes, bytearray, memoryview)):
            raw = bytes(weight_scale_divisor)
            if len(raw) != 4:
                raise ValueError("weight_scale_divisor bytes must be 4 long")
            dword = struct.unpack("<I", raw)[0]
            if not valid_positive_fp32_word(dword):
                raise ValueError("weight_scale_divisor is not a finite positive word")
            divisor = struct.unpack("<f", raw)[0]
            divisor_bytes = raw
        else:
            divisor = float(weight_scale_divisor)
            dword = struct.unpack("<I", struct.pack("<f", divisor))[0]
            if not valid_positive_fp32_word(dword):
                raise ValueError("weight_scale_divisor is not a finite positive value")
            divisor_bytes = struct.pack("<I", dword)

    groups = k // 16
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=src.device)
    scale_word = torch.empty((n, groups), dtype=torch.uint8, device=src.device)

    # Pass 2 (bounded): per-group scales + per-element codes, one row-chunk live.
    for r0, r1 in _row_chunks(n, k, _CHUNK_ELEMENTS):
        w_c = src[r0:r1].to(torch.float32)
        rows = r1 - r0
        g_c = w_c.reshape(rows, groups, 16)
        max_abs_c = g_c.abs().amax(dim=2)  # [rows, groups]
        scale_unencoded_c = divisor * max_abs_c / _E2M1_MAX
        scale_bf8_c = scale_unencoded_c.clamp_max(_E4M3_MAX).to(torch.float8_e4m3fn)
        scale_word[r0:r1] = scale_bf8_c.view(torch.uint8)
        decoded_c = scale_bf8_c.float()
        nonzero_c = decoded_c > 0
        norm_c = torch.where(nonzero_c.unsqueeze(-1), g_c / decoded_c.unsqueeze(-1), 0.0)
        norm_c = norm_c * divisor  # matches kernel: value * divisor / decoded
        codes_c = _e2m1_codes(norm_c)  # [rows, groups, 16] uint8
        packed[r0:r1] = _pack_nibbles(codes_c, rows, k)  # [rows, k // 2]

    # Sanity: every scale word must be an admitted nonnegative finite E4M3FN.
    sw = scale_word.reshape(-1)
    bad = ((sw & 0x80) != 0) | (sw == 0x7F)
    if bool(bad.any()):
        raise ValueError("produced an invalid E4M3FN scale word")

    return Nvfp4Quantized(
        packed_codes=packed,
        natural_scales=scale_word,
        weight_scale_divisor=divisor_bytes,
    )


def dequantize_nvfp4(qz: Nvfp4Quantized) -> torch.Tensor:
    """Reconstruct the logical ``[N, K]`` float32 matrix a kernel would dequantize."""

    n, k = qz.shape
    packed = qz.packed_codes
    low = packed.to(torch.int16) & 0x0F
    high = (packed.to(torch.int16) >> 4) & 0x0F
    flat = torch.empty((n, k), dtype=torch.int16, device=packed.device)
    flat[:, 0::2] = low
    flat[:, 1::2] = high
    code = flat.to(torch.uint8)
    # decode_e2m1_word is scalar; vectorize over the 16 code values.
    lut = torch.tensor(
        [decode_e2m1_word(i) for i in range(16)], dtype=torch.float32, device=code.device
    )
    weight = lut[code.reshape(-1).to(torch.int64)]  # [n*k]
    weight = weight.reshape(n, k // 16, 16)
    # Vectorized E4M3FN decode via torch.
    scale_f = qz.natural_scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)  # [n, k//16, 1]
    divisor = float(struct.unpack("<f", qz.weight_scale_divisor)[0])
    recon = weight * scale_f / divisor
    return recon.reshape(n, k)


__all__ = [
    "Nvfp4Quantized",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "weight_scale_divisor_word",
]

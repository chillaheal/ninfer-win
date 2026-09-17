"""P1: production-shape packed expert host buffers (spec section 4.3).

* allocation-free geometry queries: the fused (512 expert, hidden 2560,
  inter 640) NVFP4 payload layout, per-expert plane offsets and alignment;
* the PRODUCTION packed gate_up buffer (944 MB, one layer's expert block):
  RAM-guarded, generated on demand, verified at the file tail and against
  the per-expert bit-exact quantization contract;
* a small-geometry end-to-end write proving the two-pass streaming
  quantizer is bit-exact with the one-shot reference.

One layer only -- never 48 layers x 512 experts in a unit test.
Label: flash_next_P1.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import pytest
import torch

from tools.flash_next_dev.make_shape_tensors import (
    PRODUCTION_EXPERTS,
    ExpertGeometry,
    _expert_source,
    geometry_report,
    production_expert_geometry,
    write_packed_experts,
)
from tools.convert.qwen3_8_flash_next.quantize_nvfp4 import quantize_nvfp4

ROOT = Path(__file__).resolve().parents[3]
PROD_GATE_UP = ROOT / "out" / "flash_next_dev" / "shape" / "experts_gate_up.nvfp4"

MIN_AVAILABLE_RAM = 4 * 1024**3  # peak = output file + one expert chunk


def _available_ram_bytes() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return 0


def _ram_guarded() -> bool:
    return _available_ram_bytes() > MIN_AVAILABLE_RAM


# --- allocation-free geometry queries -----------------------------------------


def test_production_geometry_exact_numbers():
    geo = PRODUCTION_EXPERTS
    geo.validate()
    assert (geo.experts, geo.hidden, geo.inter) == (512, 2560, 640)
    assert geo.gate_up_shape == (655_360, 2560)
    assert geo.down_shape == (1_310_720, 640)
    assert geo.rows_per_expert("gate_up") == 1280
    assert geo.rows_per_expert("down") == 2560
    gu = geo.geometry("gate_up")
    assert gu.code_plane_bytes == 838_860_800
    assert gu.scale_plane_offset == 838_860_800
    assert gu.scale_plane_bytes == 104_857_600
    assert gu.payload_bytes == 943_718_404
    assert gu.weight_divisor_offset == gu.payload_bytes - 4


def test_expert_offsets_aligned_and_covered():
    geo = PRODUCTION_EXPERTS
    for role, k in (("gate_up", 2560), ("down", 640)):
        offsets = geo.expert_offsets(role, 0)
        last = geo.expert_offsets(role, geo.experts - 1)
        rpe = geo.rows_per_expert(role)
        assert offsets["code_offset"] == 0
        assert offsets["scale_offset"] == geo.geometry(role).scale_plane_offset
        assert offsets["code_bytes"] == rpe * (k // 2)
        assert offsets["scale_bytes"] == rpe * (k // 16)
        assert offsets["code_bytes"] % 256 == 0 and offsets["scale_bytes"] % 256 == 0
        assert last["code_offset"] + offsets["code_bytes"] == geo.geometry(role).code_plane_bytes
        assert last["scale_offset"] + offsets["scale_bytes"] == (
            geo.geometry(role).scale_plane_offset + geo.geometry(role).scale_plane_bytes
        )
    with pytest.raises(ValueError):
        geo.expert_offsets("gate_up", -1)
    with pytest.raises(ValueError):
        geo.expert_offsets("gate_up", 512)
    with pytest.raises(ValueError):
        geo.role_shape("bogus")


def test_geometry_report_is_allocation_free_and_consistent():
    report = geometry_report(production_expert_geometry())
    assert report["experts"] == 512
    for role in ("gate_up", "down"):
        entry = report["roles"][role]
        assert entry["payload_bytes"] == entry["code_plane_bytes"] + entry["scale_plane_bytes"] + 4
        assert entry["expert0"]["code_offset"] == 0
        assert entry["expert_last"]["code_offset"] % 256 == 0


# --- production packed buffer (RAM-guarded) ------------------------------------


def test_production_gate_up_buffer():
    if not _ram_guarded():
        pytest.skip(f"available RAM <= {MIN_AVAILABLE_RAM // 1024 // 1024} MiB")
    if not PROD_GATE_UP.exists():
        write_packed_experts(PROD_GATE_UP, role="gate_up")
    geo = PRODUCTION_EXPERTS
    gu = geo.geometry("gate_up")
    assert PROD_GATE_UP.stat().st_size == gu.payload_bytes  # ONE layer, not 48x
    rpe = geo.rows_per_expert("gate_up")
    with open(PROD_GATE_UP, "rb") as fh:
        fh.seek(gu.weight_divisor_offset)
        word = struct.unpack("<I", fh.read(4))[0]
        divisor = struct.unpack("<f", struct.pack("<I", word))[0]
        assert divisor > 0.0 and divisor == divisor
        # tail + middle experts: per-expert rows bit-exact vs regenerated source
        for expert in (0, 255, 511):
            src = _expert_source(geo, "gate_up", expert, "flash_next_P1")
            qz = quantize_nvfp4(src, weight_scale_divisor=divisor)
            r0 = expert * rpe
            fh.seek(r0 * 1280)
            codes = fh.read(1280)
            fh.seek(gu.scale_plane_offset + r0 * 160)
            scales = fh.read(160)
            assert codes == qz.packed_codes[0].contiguous().numpy().tobytes()
            assert scales == qz.natural_scales[0].contiguous().numpy().tobytes()
        # every scale word of the last expert is a valid E4M3FN word
        fh.seek(gu.scale_plane_offset + (geo.experts - 1) * rpe * 160)
        s = torch.frombuffer(fh.read(rpe * 160), dtype=torch.uint8)
        assert not bool(((s & 0x80) != 0).any())
        assert not bool((s == 0x7F).any())


# --- small-geometry end-to-end two-pass proof ----------------------------------


def test_small_geometry_two_pass_bit_exact(tmp_path):
    geo = ExpertGeometry(experts=8, hidden=256, inter=64)
    geo.validate()
    out = tmp_path / "gate_up.nvfp4"
    write_packed_experts(out, geometry=geo, role="gate_up", seed_name="p1_small")
    gu = geo.geometry("gate_up")
    assert out.stat().st_size == gu.payload_bytes
    blob = out.read_bytes()
    word = struct.unpack_from("<I", blob, gu.weight_divisor_offset)[0]
    divisor = struct.unpack("<f", struct.pack("<I", word))[0]
    assert divisor > 0.0
    for expert in range(geo.experts):
        src = _expert_source(geo, "gate_up", expert, "p1_small")
        qz = quantize_nvfp4(src, weight_scale_divisor=divisor)
        r0 = expert * geo.rows_per_expert("gate_up")
        r1 = r0 + geo.rows_per_expert("gate_up")
        assert blob[r0 * 128 : r1 * 128] == qz.packed_codes.contiguous().numpy().tobytes()
        assert blob[
            gu.scale_plane_offset + r0 * 16 : gu.scale_plane_offset + r1 * 16
        ] == qz.natural_scales.contiguous().numpy().tobytes()


def test_deterministic_reproducible_seeds():
    a = _expert_source(PRODUCTION_EXPERTS, "down", 7, "seed_x").abs().sum().item()
    b = _expert_source(PRODUCTION_EXPERTS, "down", 7, "seed_x").abs().sum().item()
    c = _expert_source(PRODUCTION_EXPERTS, "down", 7, "seed_y").abs().sum().item()
    assert a == b
    assert a != c

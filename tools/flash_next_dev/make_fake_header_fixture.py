"""Generate the checked-in P0 fixture: a 4-tensor fake ``.ninfer`` header.

Deterministic (zeroed payloads), so the fixture bytes are stable across runs.
Mirrors four real Flash-Next storage shapes (FP8 row-scale dense projection,
NVFP4 block-scale expert, FP32 contiguous GDN control, one more FP8 object) so
the inspect parser exercises the v2 prefix, the JSON directory, and per-object
offset/alignment validation without touching the 75 GB real artifact.

Usage:  python -m tools.flash_next_dev.make_fake_header_fixture
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.container import (  # noqa: E402
    ArtifactIdentity,
    TensorSpec,
    write_artifact,
)
from tools.artifact.layouts import encoded_size  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "targets" / "qwen3_8_flash_next" / "fixtures" / "fake_header_4t.ninfer"

IDENTITY = ArtifactIdentity("fake", "flash-next-p0")

SPECS = (
    TensorSpec(
        "text/token_embedding", (256, 256), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"
    ),
    TensorSpec(
        "text/layers/0/gdn/query_key_value_z",
        (128, 256),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
    ),
    TensorSpec(
        "text/layers/0/mlp/experts/gate_up",
        (128, 256),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
    ),
    TensorSpec("text/layers/0/gdn/a_log", (16,), "FP32", "contiguous-le-v1"),
)


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    entries = [(spec, bytes(encoded_size(spec.layout, spec.format, spec.shape))) for spec in SPECS]
    write_artifact(FIXTURE, IDENTITY, entries)
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

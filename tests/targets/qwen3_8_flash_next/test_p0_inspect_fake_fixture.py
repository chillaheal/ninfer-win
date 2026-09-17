"""P0: the existing inspect tool parses the checked-in 4-tensor fake header.

Label: flash_next_P0 (pytest until CMake wiring lands at P1).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactIdentity
from tools.artifact.inspect import artifact_summary

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
FIXTURE = HERE / "fixtures" / "fake_header_4t.ninfer"

EXPECTED_IDENTITY = ArtifactIdentity("fake", "flash-next-p0")
EXPECTED_OBJECTS = {
    "text/token_embedding": ((256, 256), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"),
    "text/layers/0/gdn/query_key_value_z": (
        (128, 256),
        "FP8_E4M3FN_ROW_BF16S",
        "row-scale-v1",
    ),
    "text/layers/0/mlp/experts/gate_up": (
        (128, 256),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
    ),
    "text/layers/0/gdn/a_log": ((16,), "FP32", "contiguous-le-v1"),
}


def test_fixture_round_trips_through_artifact_open():
    assert FIXTURE.is_file(), (
        "fixture missing -- run: python -m tools.flash_next_dev.make_fake_header_fixture"
    )
    with Artifact.open(FIXTURE) as artifact:
        assert artifact.identity == EXPECTED_IDENTITY
        assert len(artifact.objects) == len(EXPECTED_OBJECTS)
        for obj in artifact.objects:
            shape, fmt, layout = EXPECTED_OBJECTS[obj.name]
            assert tuple(obj.shape) == shape, obj.name
            assert obj.format == fmt, obj.name
            assert obj.layout == layout, obj.name
        summary = artifact_summary(artifact)
        assert summary["model_id"] == "fake"
        assert summary["weights_id"] == "flash-next-p0"
        assert summary["objects"] == 4
        assert summary["tensors"] == 4
        assert summary["resources"] == 0
        assert summary["formats"] == {
            "FP8_E4M3FN_ROW_BF16S": 2,
            "NVFP4": 1,
            "FP32": 1,
        }


def test_inspect_cli_json_on_fixture():
    proc = subprocess.run(
        [sys.executable, "-m", "tools.artifact.inspect", str(FIXTURE), "--json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["model_id"] == "fake"
    assert data["weights_id"] == "flash-next-p0"
    assert data["objects"] == 4
    assert data["tensors"] == 4


def test_inspect_cli_objects_listing_on_fixture():
    proc = subprocess.run(
        [sys.executable, "-m", "tools.artifact.inspect", str(FIXTURE), "--objects"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    # --objects prints a summary header first, then one row per object whose
    # line starts with the (zero-padded) payload offset; the row ends in the name.
    object_lines = [
        line for line in proc.stdout.splitlines() if line.lstrip()[:1].isdigit()
    ]
    assert len(object_lines) == 4, proc.stdout
    assert {line.split()[-1] for line in object_lines} == set(EXPECTED_OBJECTS)

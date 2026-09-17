"""P0: header-only audit of the REAL artifact. No H2D, no payload decode.

``Artifact.open`` mmaps the file and validates metadata + object ranges only;
``artifact_summary`` never touches payload bytes, so this test is safe to run
with the GPU serve resident.

Skipped unless NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS is set.
Label: flash_next_P0 (real-weight comparison is also tagged flash_next_real).
"""

from __future__ import annotations

import os

import pytest

from tools.artifact.container import Artifact
from tools.artifact.inspect import artifact_summary
from tools.convert.qwen3_8_flash_next.inventory_nvfp4 import (
    BF16,
    FP32,
    FP8,
    MODEL_ID,
    NVFP4,
    Q4,
    Q5,
    Q6,
    W8,
    WEIGHTS_ID,
)

ENV = "NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS"
pytestmark = pytest.mark.skipif(
    not os.environ.get(ENV),
    reason=f"{ENV} not set (real-artifact comparison skipped)",
)

EXPECTED_FORMATS = {
    BF16: 901,
    FP32: 268,
    Q4: 54,
    Q5: 54,
    Q6: 1,
    W8: 2,
    NVFP4: 196,
    FP8: 126,
}


def _open() -> Artifact:
    return Artifact.open(os.environ[ENV])


def test_identity_is_flash_next_not_27b_or_35b():
    with _open() as artifact:
        model_id = artifact.identity.model_id
        if model_id.startswith(("qwen3.8-27b", "qwen3.6")):
            pytest.fail(
                f"STOP: artifact identity {model_id!r} is a registered 27B/35B key "
                "(converter bug) -- do not patch the 27B registry"
            )
        assert (artifact.identity.model_id, artifact.identity.weights_id) == (
            MODEL_ID,
            WEIGHTS_ID,
        )


def test_object_counts_match_the_audit():
    with _open() as artifact:
        summary = artifact_summary(artifact)
        assert summary["objects"] == 1608
        assert summary["tensors"] == 1602
        assert summary["resources"] == 6


def test_format_counts_match_the_audit():
    with _open() as artifact:
        summary = artifact_summary(artifact)
        assert summary["formats"] == EXPECTED_FORMATS

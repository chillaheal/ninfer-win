"""P1: the mini Flash-Next fixture -- production codecs, scaled shapes.

Checks the generator's on-disk artifacts:

* mini ``.ninfer`` identity is the production identity, object count matches
  the mini inventory (order: resources first), every payload is readable at
  its planned encoded size;
* NVFP4 tensors use the blockscale payload geometry and the per-expert row
  stride is scale-tile aligned (the (layer, expert_id) slicing contract);
* the mini ``.ngram`` PLE store parses via the production ``parse_directory``
  and its block 0 decodes bit-for-bit to the checked-in golden
  (independent decode straight from file bytes).

The fixture is generated on demand (``make_mini_artifact.build``) when
missing, so ctest is self-contained. Label: flash_next_P1.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from tools.artifact.container import Artifact
from tools.artifact.layouts import block_scale_geometry, encoded_size
from tools.convert.qwen3_8_flash_next import make_mini_artifact, ngram_store
from tools.convert.qwen3_8_flash_next.inventory_nvfp4 import MODEL_ID, NVFP4, WEIGHTS_ID

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "flash_next_dev"
MINI_NINFER = OUT / make_mini_artifact.MINI_NINFER_BASENAME
MINI_NGRAM = OUT / make_mini_artifact.MINI_NGRAM_BASENAME
MINI_GOLDEN = OUT / make_mini_artifact.MINI_PLE_GOLDEN_BASENAME
MINI_INVENTORY = OUT / make_mini_artifact.MINI_INVENTORY_BASENAME
MINI_CHECKSUMS = Path(str(MINI_NGRAM) + ".checksums.json")

EXPECTED_OBJECTS = 119  # 6 resources + 113 tensors


def _ensure_fixture() -> None:
    if all(p.exists() for p in (MINI_NINFER, MINI_NGRAM, MINI_GOLDEN, MINI_INVENTORY)):
        return
    make_mini_artifact.build(OUT)


@pytest.fixture(scope="module")
def fixture():
    _ensure_fixture()
    inventory = json.loads(MINI_INVENTORY.read_text(encoding="utf-8"))
    with Artifact.open(MINI_NINFER) as artifact:
        yield artifact, inventory


def test_mini_files_exist_and_inventory_matches_identity(fixture):
    artifact, inventory = fixture
    assert (artifact.identity.model_id, artifact.identity.weights_id) == (
        MODEL_ID,
        WEIGHTS_ID,
    )
    assert inventory["identity"] == {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID}
    assert inventory["target_key"] == "qwen3_8_flash_next"


def test_object_count_matches_mini_inventory(fixture):
    artifact, inventory = fixture
    objects = list(artifact.objects)
    assert len(objects) == EXPECTED_OBJECTS
    doc_names = [r["name"] for r in inventory["resources"]] + [
        t["name"] for t in inventory["tensors"]
    ]
    assert [o.name for o in objects] == doc_names


def test_every_payload_readable_at_planned_size(fixture):
    artifact, inventory = fixture
    tensor_doc = {t["name"]: t for t in inventory["tensors"]}
    resource_doc = {r["name"] for r in inventory["resources"]}
    for obj in artifact.objects:
        span = artifact.payload(obj.name)
        assert len(span) == obj.bytes
        if obj.name in tensor_doc:
            doc = tensor_doc[obj.name]
            assert (tuple(obj.shape), obj.format, obj.layout) == (
                tuple(doc["shape"]),
                doc["format"],
                doc["layout"],
            )
            assert obj.bytes == encoded_size(obj.layout, obj.format, tuple(obj.shape))
        else:
            assert obj.name in resource_doc


def test_nvfp4_payload_geometry_and_expert_stride_alignment(fixture):
    artifact, _ = fixture
    for name in (
        "text/layers/0/mlp/experts/gate_up",
        "text/layers/0/mlp/experts/down",
    ):
        obj = artifact.find(name)
        assert obj is not None, name
        geo = block_scale_geometry(NVFP4, tuple(obj.shape))
        assert obj.bytes == geo.payload_bytes, f"{name}: payload size != geometry"
        # (layer, expert_id) slicing: per-expert stride is a 128-row N-tile multiple
        rows_per_expert = obj.shape[0] // 8  # mini: 8 experts
        assert rows_per_expert % 128 == 0, f"{name}: expert stride not tile-aligned"
        # the 4-byte weight divisor word is present and is a positive finite f32
        span = artifact.payload(name)
        word = struct.unpack_from("<I", span, geo.weight_divisor_offset)[0]
        divisor = struct.unpack("<f", struct.pack("<I", word))[0]
        assert divisor > 0.0 and divisor == divisor and divisor != float("inf")


def test_ngram_structure_via_production_parser(fixture):
    _, inventory = fixture
    directory, shards = ngram_store.parse_directory(MINI_NGRAM)
    assert directory["version"] == ngram_store.NGRAM_VERSION
    assert directory["quant_id"] == ngram_store.QUANT_ID
    assert directory["n_shards"] == 1
    assert (directory["rows_per_shard"], directory["cols_per_shard"]) == (
        inventory["ngram"]["rows_per_shard"],
        inventory["ngram"]["cols_per_shard"],
    )
    assert directory["block_rows"] == ngram_store.BLOCK_ROWS
    assert directory["block"]["block_codes_bytes"] == ngram_store.NGRAM_ALIGN
    shard = shards[0]
    assert shard.offset == directory["payload_offset"] == ngram_store.NGRAM_ALIGN
    assert shard.scale_offset == shard.offset + 1024 * 32  # codes 1024 x 32 B
    assert shard.bytes == 34816  # codes 32768 + scales 2048
    # every block code offset is 4096-aligned (the C++ mmap gather contract)
    for block in range(directory["block"]["blocks_per_shard"]):
        assert (shard.offset + block * 4096) % ngram_store.NGRAM_ALIGN == 0
    # single shard: the file ends exactly at payload + shard bytes (no
    # trailing pad; inter-shard alignment is the stride's job, not the file's)
    assert MINI_NGRAM.stat().st_size == directory["payload_offset"] + shard.bytes


def test_ple_block0_golden_crosscheck():
    """Decode block 0 straight from .ngram file bytes (no store reader) and
    compare bit-for-bit against the golden produced by the generator."""

    _ensure_fixture()
    directory, shards = ngram_store.parse_directory(MINI_NGRAM)
    block_rows = directory["block_rows"]
    cols = directory["cols_per_shard"]
    shard = shards[0]
    blob = MINI_NGRAM.read_bytes()
    codes = torch.frombuffer(
        blob[shard.offset : shard.offset + block_rows * cols], dtype=torch.uint8
    ).view(torch.float8_e4m3fn).view(block_rows, cols)
    scales = torch.frombuffer(
        blob[shard.scale_offset : shard.scale_offset + block_rows * 2],
        dtype=torch.uint8,
    ).view(torch.bfloat16)
    dequant = codes.float() * scales.float().unsqueeze(1)

    magic, rows, cols_g, *body = struct.unpack_from(
        "<8sii", MINI_GOLDEN.read_bytes(), 0
    )
    assert magic == make_mini_artifact.PLE_GOLDEN_MAGIC
    assert (rows, cols_g) == (block_rows, cols)
    golden = torch.frombuffer(
        MINI_GOLDEN.read_bytes()[16:], dtype=torch.int32
    ).view(torch.float32).view(rows, cols_g)
    assert torch.equal(dequant.view(torch.int32).contiguous(), golden.view(torch.int32).contiguous())


def test_checksums_sidecar_matches_file():
    _ensure_fixture()
    sidecar = json.loads(MINI_CHECKSUMS.read_text(encoding="utf-8"))
    assert (
        __import__("hashlib").sha256(MINI_NGRAM.read_bytes()).hexdigest()
        == sidecar["whole_file_sha256"]
    )
    assert sidecar["file"] == MINI_NGRAM.name

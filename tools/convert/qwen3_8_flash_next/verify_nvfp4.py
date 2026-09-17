"""STEP 9 -- fail-loudly structural validator for the Flash-Next ``.ninfer``.

This is the main-artifact half of the STEP 9 validator.  It mirrors the
Qwen3.8-27B ``verify_nvfp4.py`` convention (raise ``VerificationError`` on the
first contract violation, return a frozen dataclass summary, emit JSON from
``main``), adapted to the Flash-Next inventory and to the honesty constraint:

* a structurally valid artifact is NOT a working model.  The current runtime
  has no kernel for the Flash-Next architecture (GDN + MoE + PLE + hyper-
  connections + QSA + mRoPE + MTP), so every PASS report also carries
  ``runtime_ready = false`` and the full **BLOCKER: RUNTIME KERNEL REQUIRED**
  list.  A validated artifact can therefore never be mistaken for a runnable
  one.

What is checked (the *structure* contract, no source / no payload values):

* the container magic, directory member sets, per-object alignment, ordering,
  non-overlap and in-file ranges -- enforced by ``Artifact`` itself;
* the artifact identity (model_id / weights_id) matches the inventory;
* the object set is EXACTLY the inventory object set, in plan order, with a
  per-object offset recomputed by the same ``align_up`` cursor-walk the writer
  used, a matching tensor signature (shape / format / layout) + byte size
  (``encoded_size``), or a matching resource encoding;
* the aggregate numeric-format and layout distributions equal the registered
  ``FORMAT_COUNTS`` / ``LAYOUT_COUNTS``;
* the final object extent equals the payload extent (no trailing hole).

This half needs no source checkpoint, so it is exercisable (synthetic
artifact + synthetic contract) before the multi-terabyte BF16 source is
present.  The *numerical* reconstruction check (re-encode every object from
the source and compare byte-for-byte) is STEP 10 and is a separate workstream.

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.verify_nvfp4 \
        models/qwen3_8_flash_next.ninfer [--json]
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import align_up, encoded_size

from . import inventory_nvfp4 as inventory
from .mapping_report import BLOCKERS

__all__ = [
    "DEFAULT_CONTRACT",
    "InventoryContract",
    "StructureSummary",
    "VerificationError",
    "VerificationSummary",
    "runtime_support",
    "validate_structure",
    "verify_artifact",
    "main",
]


class VerificationError(ValueError):
    """The artifact does not satisfy the registered Flash-Next structure contract."""


@dataclass(frozen=True, slots=True)
class InventoryContract:
    """The structural facts ``validate_structure`` checks against.

    Parameterised (rather than read straight off ``inventory``) so the
    cursor-walk can be tested against a small synthetic artifact + synthetic
    inventory instead of the 1608-object / multi-hundred-gigabyte real one.
    ``DEFAULT_CONTRACT`` binds the real inventory for production use.
    """

    model_id: str
    weights_id: str
    object_specs: tuple
    format_counts: Mapping[str, int]
    layout_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int
    formats: Mapping[str, int]
    layouts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    structure: StructureSummary
    runtime_ready: bool
    blocker_count: int
    blockers: tuple


DEFAULT_CONTRACT = InventoryContract(
    model_id=inventory.MODEL_ID,
    weights_id=inventory.WEIGHTS_ID,
    object_specs=inventory.OBJECT_SPECS,
    format_counts=inventory.FORMAT_COUNTS,
    layout_counts=inventory.LAYOUT_COUNTS,
)


def _error(message: str) -> None:
    raise VerificationError(message)


def runtime_support() -> dict[str, object]:
    """The honesty report: structural validity != the model runs.

    The artifact is a STATIC deliverable.  Whether the model actually runs is
    a property of the runtime, which today has no kernel for the Flash-Next
    architecture.  So this is always ``runtime_ready = false`` with the blocked
    operations listed verbatim (``BLOCKER: RUNTIME KERNEL REQUIRED``).
    """

    return {
        "runtime_ready": False,
        "note": (
            "Structural + plan validity does NOT mean the model runs. The "
            "current runtime has no kernel for the Flash-Next architecture "
            "(GDN + MoE + PLE + hyper-connections + QSA + mRoPE + MTP). Each "
            "operation below is BLOCKER: RUNTIME KERNEL REQUIRED; this report "
            "exists so a validated artifact is never mistaken for a working model."
        ),
        "blockers": list(BLOCKERS),
        "blocker_count": len(BLOCKERS),
    }


def validate_structure(
    artifact: Artifact,
    contract: InventoryContract = DEFAULT_CONTRACT,
) -> StructureSummary:
    """Validate the directory against the contract without reading payload values.

    The container layer (magic, member sets, alignment, ordering, non-overlap,
    in-file ranges) is enforced by ``Artifact`` before this runs; this adds the
    *plan* layer -- identity, object identity/offset/signature/size, the
    aggregate format + layout distributions, and the no-trailing-hole extent.
    """

    expected_identity = ArtifactIdentity(contract.model_id, contract.weights_id)
    if artifact.identity != expected_identity:
        _error(
            f"artifact identity is {artifact.identity!r}, expected {expected_identity!r}"
        )
    if len(artifact.objects) != len(contract.object_specs):
        _error(
            f"artifact has {len(artifact.objects)} objects, expected "
            f"{len(contract.object_specs)}"
        )

    cursor = 0
    tensor_count = 0
    resource_count = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, contract.object_specs)
    ):
        if actual.name != expected.name:
            _error(f"object {position} is {actual.name!r}, expected {expected.name!r}")
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _error(f"{actual.name}: offset {actual.offset}, expected {expected_offset}")

        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _error(f"{actual.name}: expected a tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _error(f"{actual.name}: signature {signature} != {registered}")
            required = encoded_size(actual.layout, actual.format, actual.shape)
            if actual.bytes != required:
                _error(f"{actual.name}: stores {actual.bytes} bytes, expected {required}")
            tensor_count += 1
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _error(f"{actual.name}: expected a resource descriptor")
            if actual.encoding != expected.encoding:
                _error(
                    f"{actual.name}: encoding {actual.encoding!r}, expected {expected.encoding!r}"
                )
            resource_count += 1

        cursor = actual.offset + actual.bytes

    if dict(formats) != dict(contract.format_counts):
        _error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != dict(contract.layout_counts):
        _error(f"layout counts are {dict(layouts)}")

    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")

    return StructureSummary(
        objects=len(artifact.objects),
        tensors=tensor_count,
        resources=resource_count,
        payload_bytes=payload_bytes,
        formats=dict(formats),
        layouts=dict(layouts),
    )


def verify_artifact(
    artifact: Artifact,
    contract: InventoryContract = DEFAULT_CONTRACT,
) -> VerificationSummary:
    """Structure contract + the runtime-support honesty report (STEP 9)."""

    support = runtime_support()
    return VerificationSummary(
        structure=validate_structure(artifact, contract),
        runtime_ready=bool(support["runtime_ready"]),
        blocker_count=int(support["blocker_count"]),
        blockers=tuple(support["blockers"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = parser.parse_args(argv)

    try:
        with Artifact.open(args.artifact) as artifact:
            summary = verify_artifact(artifact)
    except VerificationError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if args.json:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    else:
        s = summary.structure
        print(
            f"[PASS] {args.artifact}: objects={s.objects} tensors={s.tensors} "
            f"resources={s.resources} payload_bytes={s.payload_bytes}"
        )
        print(f"       formats={dict(s.formats)}")
        print(f"       layouts={dict(s.layouts)}")
        print(
            f"       runtime_ready={summary.runtime_ready} "
            f"({summary.blocker_count} operations BLOCKED: RUNTIME KERNEL REQUIRED)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resumable single-pass writer for the NInfer v2 object directory.

The shared ``tools.artifact.container.ArtifactWriter`` is a strict single-pass
streaming writer: one crash anywhere loses the whole run.  The Flash-Next
artifact is ~75 GB assembled from a 360 GB BF16 checkpoint on a network drive,
so an interrupted run would cost hours to redo.  This writer produces a
*byte-identical* file to the shared writer but tracks per-object progress in a
sidecar so a run can resume from the first incomplete object.

The file layout is identical to ``ArtifactWriter``::

    [ PREFIX: <8s magic, Q json_bytes> ][ json directory ][ zero pad ]
    [ payload: objects in plan order, each at payload_offset + offset ]

``plan_objects`` and ``encode_directory`` are deterministic in ``(identity,
specs)``, so the on-disk directory (and therefore every object's absolute
offset) is fixed for a given inventory.  Resume = verify the on-disk directory
still equals the recomputed one, then seek to the first incomplete object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tools.artifact.container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    PREFIX_BYTES,
    ArtifactIdentity,
    ArtifactObject,
    ObjectSpec,
    encode_directory,
    plan_objects,
)
from tools.artifact.layouts import align_up


class ArtifactWriterError(RuntimeError):
    """The writer state is inconsistent with the on-disk artifact."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class _Progress:
    directory_sha256: str
    completed: int

    @classmethod
    def read(cls, path: Path) -> "_Progress | None":
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            directory_sha256=str(value["directory_sha256"]),
            completed=int(value["completed"]),
        )

    def write(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"directory_sha256": self.directory_sha256, "completed": self.completed}),
            encoding="utf-8",
        )
        tmp.replace(path)


class ResumableArtifactWriter:
    """Write a preplanned artifact, resuming from the last complete object."""

    def __init__(
        self,
        path: str | Path,
        identity: ArtifactIdentity,
        specs: Sequence[ObjectSpec],
        *,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.identity = identity
        self.objects = plan_objects(specs)
        self.directory = encode_directory(self.identity, self.objects)
        self.directory_sha256 = _sha256(self.directory)
        self.payload_offset = align_up(PREFIX_BYTES + len(self.directory), PAYLOAD_ALIGNMENT)
        self.progress_path = self.path.with_name(self.path.name + ".progress.json")

        started = self._resolve_start(overwrite)
        self._file = self.path.open("r+b")
        self._next = started
        self._cursor = self._object_file_offset(started) if self.objects else self.payload_offset
        self._finish_progress(started)

    @property
    def completed(self) -> int:
        """Number of objects already written and recorded complete (0-based).

        A fresh file reports 0; a resumed file reports the count it validated.
        The converter skips the first ``completed`` planned objects.
        """

        return self._next

    # -- lifecycle -----------------------------------------------------------

    def _resolve_start(self, overwrite: bool) -> int:
        if not self.path.exists():
            self._write_header()
            return 0
        if overwrite:
            self._write_header()
            return 0
        # Validate the existing file carries the *same* directory.
        with self.path.open("rb") as handle:
            head = handle.read(PREFIX_BYTES)
            if len(head) < PREFIX_BYTES:
                raise ArtifactWriterError("existing artifact is shorter than the v2 prefix")
            magic, json_bytes = PREFIX.unpack(head)
            if magic != MAGIC:
                raise ArtifactWriterError("existing artifact magic is not NInfer v2")
            stored_directory = handle.read(json_bytes)
        if len(stored_directory) != len(self.directory) or stored_directory != self.directory:
            raise ArtifactWriterError(
                "existing artifact has a different object plan; re-run with --overwrite"
            )
        progress = _Progress.read(self.progress_path)
        if progress is None or progress.directory_sha256 != self.directory_sha256:
            raise ArtifactWriterError(
                "artifact exists but has no matching progress record; re-run with --overwrite"
            )
        completed = progress.completed
        if completed > len(self.objects):
            raise ArtifactWriterError("progress record exceeds the object count")
        return completed

    def _write_header(self) -> None:
        with self.path.open("wb") as handle:
            handle.write(PREFIX.pack(MAGIC, len(self.directory)))
            handle.write(self.directory)
            handle.write(b"\x00" * (self.payload_offset - PREFIX_BYTES - len(self.directory)))
        self.progress_path.unlink(missing_ok=True)

    def _object_file_offset(self, index: int) -> int:
        if index >= len(self.objects):
            last = self.objects[-1]
            return self.payload_offset + last.offset + last.bytes
        return self.payload_offset + self.objects[index].offset

    def _finish_progress(self, completed: int) -> None:
        _Progress(self.directory_sha256, completed).write(self.progress_path)

    # -- writing -------------------------------------------------------------

    def write(self, name: str, payload: bytes) -> None:
        if self._next >= len(self.objects):
            raise ArtifactWriterError("artifact already has every planned payload")
        obj = self.objects[self._next]
        if name != obj.name:
            raise ArtifactWriterError(f"expected payload {obj.name}, got {name}")
        if len(payload) != obj.bytes:
            raise ArtifactWriterError(
                f"payload {name} has {len(payload)} bytes; expected {obj.bytes}"
            )
        target = self.payload_offset + obj.offset
        if target > self._cursor:
            # Zero-fill the alignment gap before this object.
            self._file.seek(self._cursor)
            self._file.write(b"\x00" * (target - self._cursor))
        self._file.seek(target)
        self._file.write(payload)
        self._file.flush()
        self._cursor = target + len(payload)
        self._next += 1
        self._finish_progress(self._next)

    def finish(self) -> None:
        if self._next != len(self.objects):
            missing = self.objects[self._next].name
            raise ArtifactWriterError(f"artifact is missing payload {missing}")
        # _cursor is an absolute file offset (payload_offset-inclusive).
        self._file.truncate(self._cursor)
        self._file.flush()
        self._file.close()
        self.progress_path.unlink(missing_ok=True)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "ResumableArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.finish()
            finally:
                self.close()
        else:
            self.close()


def validate_artifact_file(path: str | Path) -> tuple[ArtifactIdentity, tuple[ArtifactObject, ...]]:
    """Structurally validate a finished artifact and return its directory.

    Mirrors ``container.Artifact`` construction (magic, directory, per-object
    alignment/range checks) but is a stateless helper the validator reuses.
    """

    from tools.artifact.container import (
        _validate_ranges,
        parse_directory,
    )
    from tools.artifact.layouts import align_up

    p = Path(path)
    with p.open("rb") as handle:
        file_bytes = handle.seek(0, 2)
        handle.seek(0)
        head = handle.read(PREFIX_BYTES)
        if len(head) < PREFIX_BYTES:
            raise ArtifactWriterError("artifact is shorter than the v2 prefix")
        magic, json_bytes = PREFIX.unpack(head)
        if magic != MAGIC:
            raise ArtifactWriterError("artifact magic is not NInfer v2")
        metadata_end = PREFIX_BYTES + json_bytes
        payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
        if metadata_end > file_bytes or payload_offset > file_bytes:
            raise ArtifactWriterError("declared JSON or payload extends beyond the file")
        directory = handle.read(json_bytes)
    identity, objects = parse_directory(directory)
    payload_bytes = file_bytes - payload_offset
    index = _validate_ranges(objects, payload_bytes)
    # Every object must actually be backed by file bytes (range check already
    # enforces end <= payload_bytes); also confirm no trailing gap beyond the
    # last object was left as a hole that finish() should have truncated.
    expected_total = payload_offset + (objects[-1].offset + objects[-1].bytes)
    if file_bytes < expected_total:
        raise ArtifactWriterError("artifact payload is truncated")
    return identity, objects


__all__ = [
    "ArtifactWriterError",
    "ResumableArtifactWriter",
    "validate_artifact_file",
]

"""Shape/dtype table reader for recipe preflight before full weights land.

The official checkpoint is a 360 GB / 131-shard download.  Its per-tensor
shape and dtype are already available in the safetensors *headers* (captured
to ``tensor_shapes.json`` during research), so a recipe can be prefaulted
against the real geometry without streaming the weights.

This reader exposes the same surface ``family_recipe.preflight_source_reader``
needs from a ``ShardReader`` (``metadata(names)`` entries carrying ``.shape``
and ``.dtype``, plus ``.names`` / ``.has``) over that header table.  It is a
preflight aid only -- it never materializes tensor bytes; the real conversion
uses ``tools.convert.common.safetensors.ShardReader``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ShapeTableEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str


class ShapeTableReader:
    """Read-only shape/dtype view over a safetensors-headers table."""

    def __init__(self, source: str | Path | Mapping[str, Mapping[str, object]]):
        if isinstance(source, Mapping):
            table = source
        else:
            with Path(source).open("r", encoding="utf-8") as handle:
                table = json.load(handle)
        self._table: dict[str, ShapeTableEntry] = {}
        for shard, header in table.items():
            for name, meta in header.items():
                self._table[name] = ShapeTableEntry(
                    name=name,
                    shape=tuple(int(dim) for dim in meta["shape"]),
                    dtype=str(meta["dtype"]),
                    shard=str(shard),
                )

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._table)

    def has(self, name: str) -> bool:
        return name in self._table

    def metadata(self, names: Iterable[str]) -> dict[str, ShapeTableEntry]:
        missing = set(names).difference(self._table)
        if missing:
            raise KeyError(f"shape table is missing {sorted(missing)[0]!r}")
        return {name: self._table[name] for name in names}


def load(shape_table_path: str | Path) -> ShapeTableReader:
    return ShapeTableReader(shape_table_path)


__all__ = ["ShapeTableEntry", "ShapeTableReader", "load"]

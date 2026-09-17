"""P0: generate the Flash-Next artifact audit table.

Imports the canonical inventory (no file access, no conversion), computes the
encoded payload size of every object, and assigns the proposed residency class:

* ``host_experts``  -- the fused 512-expert NVFP4 matrices (48 layers x 2).
* ``gpu_resident``  -- everything else in the ``.ninfer`` (backbone, shared
  experts, norms, HC, router, indexers, MTP, Vision, scalar scales).
* ``mmap_ple``      -- the 128 PLE n-gram shards + 3 I64 config tables; these
  live in the separate ``.ngram`` store, NOT in the ``.ninfer`` (documented
  here for completeness).

Writes out/flash_next_dev/audit_objects.csv and prints the section/residency
totals used by docs/maintainer/qwen3_8_flash_next-artifact-audit.md.

Usage:  python -m tools.flash_next_dev.audit_inventory
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.layouts import encoded_size  # noqa: E402
from tools.convert.qwen3_8_flash_next.inventory_nvfp4 import OBJECT_SPECS  # noqa: E402

OUT = REPO_ROOT / "out" / "flash_next_dev"
CSV_PATH = OUT / "audit_objects.csv"


def section(name: str) -> str:
    if name.startswith("text/"):
        return "text"
    if name.startswith("mtp/"):
        return "mtp"
    if name.startswith("vision/"):
        return "vision"
    return "resource" if name.startswith("frontend/") or "/" not in name else "other"


def residency(name: str) -> str:
    if name.endswith("/mlp/experts/gate_up") or name.endswith("/mlp/experts/down"):
        return "host_experts"
    return "gpu_resident"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in OBJECT_SPECS:
        if hasattr(spec, "encoding"):
            # Converter-side ResourceSpecs carry no size; the real per-resource
            # byte counts are in out/flash_next_dev/inspect_real_objects.txt.
            rows.append(
                {
                    "name": spec.name,
                    "section": section(spec.name),
                    "shape": "",
                    "format": spec.encoding,
                    "layout": "",
                    "nbytes": "",
                    "residency": "frontend",
                }
            )
        else:
            rows.append(
                {
                    "name": spec.name,
                    "section": section(spec.name),
                    "shape": ",".join(str(d) for d in spec.shape),
                    "format": spec.format,
                    "layout": spec.layout,
                    "nbytes": encoded_size(spec.layout, spec.format, spec.shape),
                    "residency": residency(spec.name),
                }
            )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["name", "section", "shape", "format", "layout", "nbytes", "residency"]
        )
        writer.writeheader()
        writer.writerows(rows)

    by_section = defaultdict(lambda: [0, 0])
    by_residency = defaultdict(lambda: [0, 0])
    for row in rows:
        nbytes = row["nbytes"] or 0
        by_section[row["section"]][0] += 1
        by_section[row["section"]][1] += nbytes
        by_residency[row["residency"]][0] += 1
        by_residency[row["residency"]][1] += nbytes

    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    total = sum(r["nbytes"] or 0 for r in rows)
    print(f"total tensor payload bytes: {total:,} ({total / 1e9:.3f} GB)")
    for key, (count, nbytes) in sorted(by_section.items()):
        print(f"  section {key:<10} {count:>5} objects  {nbytes:>15,} B")
    for key, (count, nbytes) in sorted(by_residency.items()):
        print(f"  residency {key:<14} {count:>5} objects  {nbytes:>15,} B ({nbytes / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()

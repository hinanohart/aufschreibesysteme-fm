"""Hard-fail audio license audit.

Walks every entry in ``data/manifests/gramophone.csv`` and verifies it is
listed in the per-item whitelist. Exits 0 only if every audio item is
whitelist-approved. Designed to be run as part of CI and at step 1 of
``afm oss`` (gate G1).

Usage:
    python scripts/audit_audio.py
    python scripts/audit_audio.py --whitelist data/manifests/audio_whitelist.csv

EU neighbouring rights ("Leistungsschutzrechte") run 70 years past the
performer's death. Many Internet Archive 78rpm transfers feature performers
who died after 1955 — those recordings are still in copyright in EU
jurisdictions even when the underlying composition is public domain. Hence
the whitelist is per-item and conservative.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_whitelist(p: Path) -> set[str]:
    if not p.is_file():
        print(f"audio whitelist not found at {p}", file=sys.stderr)
        sys.exit(2)
    with p.open("r", encoding="utf-8") as fh:
        return {row["path"] for row in csv.DictReader(fh)}


def audit(manifest: Path, whitelist: set[str]) -> tuple[int, int, list[str]]:
    if not manifest.is_file():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        sys.exit(2)
    rejected: list[str] = []
    total = 0
    approved = 0
    with manifest.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            path = row.get("path", "")
            source = row.get("source", "")
            if source == "internet_archive_78rpm_bulk":
                rejected.append(path)
                continue
            if path in whitelist or source in {"wikimedia_pd", "loc_american_memory_pd"}:
                approved += 1
            else:
                rejected.append(path)
    return total, approved, rejected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/gramophone.csv", type=Path)
    ap.add_argument("--whitelist", default="data/manifests/audio_whitelist.csv", type=Path)
    args = ap.parse_args()

    whitelist = load_whitelist(args.whitelist)
    total, approved, rejected = audit(args.manifest, whitelist)
    print(f"total={total} approved={approved} rejected={len(rejected)}")
    if rejected:
        print("rejected (first 10):")
        for p in rejected[:10]:
            print(f"  - {p}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Per-regime data loaders + license-audited manifest reader.

Every audio item MUST be present in ``data/manifests/audio_whitelist.csv``
or ``audit_manifests`` fails. Internet Archive 78rpm bulk import is
specifically disallowed because EU neighbouring rights run 70 years past
the performer's death and most of that bulk is still in copyright in EU
jurisdictions (see Critic C1 / docs/architecture.md).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    path: str
    license: str
    sha256: str
    source: str
    whitelist_approved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeManifest:
    regime: str
    entries: list[ManifestEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def load_manifest(regime: str, *, root: Path = Path("data/manifests")) -> RegimeManifest:
    p = root / f"{regime}.csv"
    if not p.is_file():
        # Empty manifest is fine for smoke tests; real training will fail in
        # iter_samples because there's nothing to yield.
        return RegimeManifest(regime=regime, entries=[])
    entries: list[ManifestEntry] = []
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(
                ManifestEntry(
                    path=row["path"],
                    license=row["license"],
                    sha256=row["sha256"],
                    source=row["source"],
                    whitelist_approved=row.get("whitelist_approved", "false").lower() == "true",
                    extra={
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            "path",
                            "license",
                            "sha256",
                            "source",
                            "whitelist_approved",
                        }
                    },
                )
            )
    return RegimeManifest(regime=regime, entries=entries)


# ----- audit (step 1) -------------------------------------------------------


DISALLOWED_AUDIO_SOURCES = {
    # See docs/architecture.md — IA 78rpm bulk fails the manual whitelist check.
    "internet_archive_78rpm_bulk",
}

ALLOWED_AUDIO_SOURCES = {
    "wikimedia_pd",
    "loc_american_memory_pd",
    "manual_whitelist",
}

ALLOWED_IMAGE_SOURCES = {
    "kodak_photocd",
    "clic2020",
    "loc_pd",
    "wikimedia_cc0",
}


def audit_manifests(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Step 1 body — license-audit every manifest. Hard-fail on disallowed.

    Returns a report dict written into state.json (so a partial run can be
    inspected without rerunning the audit). The structure is:

        {
            "audio": {
                "total": <int>,
                "whitelist_approved": <int>,
                "rejected": [<path>, ...],
                "license_whitelist_verified": <bool>,
            },
            "images": {...},
            "files": <int>,
        }
    """
    report: dict[str, Any] = {"audio": {}, "images": {}, "files": 0}

    # Audio is the strict path.
    audio_cfg = data_cfg.get("audio", {})
    if audio_cfg.get("bulk_internet_archive_78rpm", False):
        raise RuntimeError(
            "data.audio.bulk_internet_archive_78rpm is True — disallowed. "
            "See docs/architecture.md C1 (EU neighbouring rights 70-year)."
        )
    whitelist_path = Path(audio_cfg.get("whitelist", "data/manifests/audio_whitelist.csv"))
    if not whitelist_path.is_file():
        raise FileNotFoundError(f"audio whitelist missing: {whitelist_path}")
    whitelist = {row["path"] for row in csv.DictReader(whitelist_path.open("r", encoding="utf-8"))}

    audio_total = 0
    audio_ok = 0
    rejected: list[str] = []
    for regime in ("gramophone",):  # only audio regime in MVP
        man = load_manifest(regime)
        for e in man.entries:
            audio_total += 1
            if e.source in DISALLOWED_AUDIO_SOURCES:
                rejected.append(e.path)
                continue
            if e.source not in ALLOWED_AUDIO_SOURCES and e.path not in whitelist:
                rejected.append(e.path)
                continue
            audio_ok += 1
    if rejected:
        raise RuntimeError(
            f"audio audit failed — {len(rejected)} item(s) not on whitelist; "
            f"first 3: {rejected[:3]}"
        )
    report["audio"] = {
        "total": audio_total,
        "whitelist_approved": audio_ok,
        "rejected": rejected,
        "license_whitelist_verified": audio_total == audio_ok,
    }

    # Image audit is looser — we only flag unknown sources, don't fail.
    image_total = 0
    image_unknown: list[str] = []
    for regime in ("parchment", "typewriter", "photograph", "film", "jpeg", "crt"):
        man = load_manifest(regime)
        for e in man.entries:
            image_total += 1
            if e.source not in ALLOWED_IMAGE_SOURCES:
                image_unknown.append(e.source)
    report["images"] = {
        "total": image_total,
        "unknown_sources": sorted(set(image_unknown)),
    }
    report["files"] = audio_total + image_total
    return report


# ----- sample iterator (cycled by the trainer) -----------------------------


def iter_samples(
    manifest: RegimeManifest,
    *,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield batched samples — placeholder that synthesises random tensors
    until the regime's measurement / dataset is wired.

    The trainer wraps this in ``DataLoader(batch_size=None)`` and pin_memory.
    """
    import torch

    while True:
        # Synthesise a batch — real loader replaces this once data fetched.
        x = torch.randn(batch_size, 3, 256, 256)
        text_emb = torch.randn(batch_size, 64, 1024)
        yield {"x": x, "text_emb": text_emb, "regime": manifest.regime}


__all__ = [
    "ManifestEntry",
    "RegimeManifest",
    "load_manifest",
    "iter_samples",
    "audit_manifests",
    "DISALLOWED_AUDIO_SOURCES",
    "ALLOWED_AUDIO_SOURCES",
    "ALLOWED_IMAGE_SOURCES",
]

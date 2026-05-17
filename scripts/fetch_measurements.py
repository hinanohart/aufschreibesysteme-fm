"""Async measurement fetcher for the 7 regimes.

Pulls public-domain audio/image samples (rate-limited to 1 req/s per host)
to ``data/datasets/<regime>/`` and writes SHA256SUMS alongside. Each row in
the input manifest carries an explicit ``license`` column so the audit step
can validate everything.

Usage:
    python scripts/fetch_measurements.py --regime gramophone
    python scripts/fetch_measurements.py --regime all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

import httpx

REGIME_NAMES = (
    "parchment",
    "typewriter",
    "gramophone",
    "photograph",
    "film",
    "jpeg",
    "crt",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    sem: asyncio.Semaphore,
) -> None:
    if dest.is_file():
        return
    async with sem:
        async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                async for chunk in r.aiter_bytes(1 << 20):
                    fh.write(chunk)
        await asyncio.sleep(1.0)  # rate limit: 1 req/s


async def _fetch_manifest(regime: str, *, manifest_root: Path, out_root: Path) -> dict:
    src = manifest_root / f"{regime}_sources.csv"
    if not src.is_file():
        print(f"[{regime}] no source manifest at {src}; skipping", file=sys.stderr)
        return {"regime": regime, "fetched": 0, "skipped": True}

    sem = asyncio.Semaphore(1)  # serialise to honour rate limit
    rows: list[dict] = []
    with src.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out_dir = out_root / regime
    out_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(headers={"User-Agent": "afm-fetch/0.1"}) as client:
        tasks = []
        for row in rows:
            dest = out_dir / row["filename"]
            tasks.append(_fetch_one(client, row["url"], dest, sem=sem))
        await asyncio.gather(*tasks)

    # Write SHA256SUMS
    sha_path = out_dir / "SHA256SUMS"
    with sha_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            dest = out_dir / row["filename"]
            if dest.is_file():
                fh.write(f"{_sha256(dest)}  {row['filename']}\n")

    return {"regime": regime, "fetched": len(rows), "sha_file": str(sha_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="all", help="regime name or 'all'")
    parser.add_argument("--manifest-root", default="data/manifests", type=Path)
    parser.add_argument("--out-root", default="data/datasets", type=Path)
    args = parser.parse_args()

    regimes: Iterable[str]
    if args.regime == "all":
        regimes = REGIME_NAMES
    elif args.regime in REGIME_NAMES:
        regimes = (args.regime,)
    else:
        parser.error(f"unknown regime {args.regime!r}; valid: {REGIME_NAMES}")
        return

    async def _run() -> list[dict]:
        return [
            await _fetch_manifest(r, manifest_root=args.manifest_root, out_root=args.out_root)
            for r in regimes
        ]

    out = asyncio.run(_run())
    for row in out:
        print(row)


if __name__ == "__main__":
    main()

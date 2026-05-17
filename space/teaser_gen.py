"""Render the 7-regime teaser gallery used in README / HF Space / X post.

The teaser is the broad-impact hook — same prompt across all 7 regimes,
horizontal layout, no labels in the image itself (they go in the caption).

Usage:
    python space/teaser_gen.py --config configs/mvp.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mvp.yaml", type=Path)
    ap.add_argument(
        "--out", default=None, type=Path, help="output path; defaults to cfg.space.teaser.out"
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.out is not None:
        cfg["space"]["teaser"]["out"] = str(args.out)

    from afm.cli.space_deploy import build_teaser

    out = build_teaser(cfg)
    print(f"teaser written to {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()

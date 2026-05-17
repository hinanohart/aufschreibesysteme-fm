"""Step 4 body — orchestrates M1-M4 against the trained pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import print as rprint


def run_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run M1-M4 and write per-regime JSON under ``eval/{regime}/...``.

    Returns a small dict for state.json. We deliberately do not store the
    full reports in state.json — they live in eval/{regime}/ on disk.
    """
    eval_dir = Path(cfg.get("eval", {}).get("out_root", "eval"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Each metric is a thin wrapper; the heavy lifting (data + model load)
    # is owned by the trainer / pipeline. For MVP we emit placeholder
    # reports so the orchestrator can advance — real numbers populate when
    # the GPU run finishes.
    summary: dict[str, Any] = {}
    for regime in cfg["regimes"]["enabled"]:
        target_dir = eval_dir / regime
        target_dir.mkdir(parents=True, exist_ok=True)
        for metric in ("classifier", "fid", "morph", "inscription"):
            report_path = target_dir / f"{metric}.json"
            if not report_path.is_file():
                report_path.write_text(json.dumps({"pending": True}, indent=2))
        summary[regime] = {"dir": str(target_dir), "metrics_written": 4}
    rprint(f"[green]eval scaffolded[/green] under {eval_dir}/")
    return summary


__all__ = ["run_eval"]

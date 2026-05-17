"""``afm morph`` — cross-regime morphing via DSBM (inference-time only)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def morph_command(
    prompt: str = typer.Argument(..., help="text prompt"),
    from_regime: str = typer.Option(..., "--from", help="source regime"),
    to_regime: str = typer.Option(..., "--to", help="target regime"),
    steps: int = typer.Option(50, "--steps", "-s"),
    out_dir: Path = typer.Option(Path("morph_out"), "--out", "-o"),
    seed: int | None = typer.Option(None, "--seed"),
    pretrained: str = typer.Option(
        "hinanohart/aufschreibesysteme-fm",
        "--pretrained",
        help="HF Hub repo id to load weights from",
    ),
) -> None:
    """Morph between two regimes for the same prompt (DSBM)."""
    from afm.core.pipeline import AufschreibePipeline

    pipe = AufschreibePipeline.from_pretrained(pretrained)
    frames = pipe.morph(
        prompt=prompt,
        regime_a=from_regime,
        regime_b=to_regime,
        steps=steps,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_frames(frames, out_dir)
    rprint(f"[green]saved {len(frames)} frames to[/green] {out_dir}")


def _save_frames(frames, out_dir: Path) -> None:
    from PIL import Image

    for i, x in enumerate(frames):
        x = x.detach().cpu().float().clamp(0, 1)
        if x.ndim == 4:
            x = x[0]
        arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"frame_{i:04d}.png")


__all__ = ["morph_command"]

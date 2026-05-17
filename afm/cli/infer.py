"""``afm infer`` — sample a single image from one regime."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def infer_command(
    prompt: str = typer.Argument(..., help="text prompt"),
    regime: str = typer.Option("jpeg", "--regime", "-r"),
    steps: int = typer.Option(50, "--steps", "-s"),
    seed: int | None = typer.Option(None, "--seed"),
    out: Path = typer.Option(Path("infer_out.png"), "--out", "-o"),
    pretrained: str = typer.Option(
        "hinanohart/aufschreibesysteme-fm",
        "--pretrained",
        help="HF Hub repo id to load weights from",
    ),
) -> None:
    """Generate a single sample from a single regime."""
    from afm.core.pipeline import AufschreibePipeline, InferenceConfig

    pipe = AufschreibePipeline.from_pretrained(pretrained)
    cfg = InferenceConfig(num_inference_steps=steps, seed=seed)
    x = pipe(prompt=prompt, regime=regime, config=cfg)
    _save_image(x, out)
    rprint(f"[green]saved[/green] {out}")


def _save_image(x, out: Path) -> None:
    from PIL import Image

    x = x.detach().cpu().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(out)


__all__ = ["infer_command"]

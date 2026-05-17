"""``afm demo`` — single-command reproduction entry.

    pipx install afm
    afm demo --regime jpeg

Pulls pre-trained weights, generates a sample, opens it. The whole point
is that someone reading the paper can verify the gallery in 30 seconds.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def demo_command(
    regime: str = typer.Option("jpeg", "--regime", "-r"),
    prompt: str = typer.Option("a lighthouse at the edge of a stormy sea", "--prompt"),
    out: Path = typer.Option(Path("demo.png"), "--out", "-o"),
    pretrained: str = typer.Option(
        "hinanohart/aufschreibesysteme-fm",
        "--pretrained",
        help="HF Hub repo id with pre-trained adapters",
    ),
) -> None:
    """Run the 1-command reproduction described in README."""
    from afm.core.pipeline import AufschreibePipeline, InferenceConfig

    rprint(f"[bold]demo[/bold] regime=[cyan]{regime}[/cyan] prompt={prompt!r}")
    pipe = AufschreibePipeline.from_pretrained(pretrained)
    x = pipe(prompt=prompt, regime=regime, config=InferenceConfig(seed=17))
    _save(x, out)
    rprint(f"[green]saved[/green] {out}")


def _save(x, out: Path) -> None:
    from PIL import Image

    x = x.detach().cpu().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(out)


__all__ = ["demo_command"]

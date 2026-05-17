"""``afm space-deploy`` + release/teaser helpers used by ``afm oss``.

Step 5 (release) and 5.5 (HF Space + teaser gallery) live here. The HF Space
publish gate (G3) is enforced by the orchestrator — this module only renders
and pushes artefacts; the actual push commands need ``huggingface-cli login``
to have run beforehand (token is NEVER read via this code path, see R11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich import print as rprint


def space_deploy_command(
    config: Path = typer.Option(Path("configs/mvp.yaml"), "--config", "-c"),
) -> None:
    """Build the Space app + teaser, then push (requires hub auth)."""
    import yaml

    cfg = yaml.safe_load(config.read_text())
    teaser = build_teaser(cfg)
    rprint(f"[green]teaser[/green] {teaser}")
    info = build_space(cfg)
    rprint(info)


# --- step 5 (release) -------------------------------------------------------


def release_to_hub(cfg: dict[str, Any]) -> dict[str, Any]:
    """Push the trained LoRAs + router + MuLAN to HF Hub.

    No token is read from the environment by this code path. We rely on
    ``huggingface_hub``'s default auth (``HUGGINGFACE_TOKEN`` env var OR
    a previous ``huggingface-cli login``). This keeps R11 — Claude never
    touches the token directly.
    """
    repo = cfg["release"]["hub_repo"]
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise ImportError("huggingface_hub is required for release") from e

    api = HfApi()
    ckpt_root = Path(cfg.get("release", {}).get("ckpt_root", "ckpt"))
    rprint(f"[bold]release[/bold] → {repo} (from {ckpt_root})")
    if not ckpt_root.is_dir():
        return {"skipped": True, "reason": f"{ckpt_root} not found"}
    # Upload folder — auth is implicit through huggingface_hub.
    api.upload_folder(folder_path=str(ckpt_root), repo_id=repo, repo_type="model")
    return {"hub_repo": repo, "uploaded": True}


# --- step 5.5 (teaser + Space) ----------------------------------------------


def build_teaser(cfg: dict[str, Any]) -> str:
    """Generate the 7-regime gallery image from a single prompt.

    The result is written to ``cfg.space.teaser.out`` (default
    ``space/assets/teaser.png``). Returns the output path string.

    This function lazily imports the pipeline so the CLI stays fast.
    """
    teaser_cfg = cfg["space"]["teaser"]
    out_path = Path(teaser_cfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = teaser_cfg["prompt"]
    regimes = teaser_cfg["regimes"]

    from PIL import Image

    from afm.core.pipeline import AufschreibePipeline, InferenceConfig

    pipe = AufschreibePipeline.from_pretrained(cfg["release"]["hub_repo"])
    panels = []
    for r in regimes:
        x = pipe(prompt=prompt, regime=r, config=InferenceConfig(seed=17))
        x = x.detach().cpu().float().clamp(0, 1)
        if x.ndim == 4:
            x = x[0]
        arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
        panels.append(Image.fromarray(arr))
    # Horizontal concat — labelled in the README, not in the image itself.
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * len(panels), h), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * w, 0))
    canvas.save(out_path)
    return str(out_path)


def build_space(cfg: dict[str, Any]) -> dict[str, Any]:
    """Push the Space directory to HF Spaces.

    The actual ``space/app.py`` lives in the repo and is unchanged at push
    time. We only upload, never edit secrets in-place.
    """
    repo = cfg["space"]["hub_repo"]
    space_dir = Path("space")
    if not space_dir.is_dir():
        return {"skipped": True, "reason": "space/ not found"}
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise ImportError("huggingface_hub is required for space-deploy") from e
    api = HfApi()
    api.upload_folder(folder_path=str(space_dir), repo_id=repo, repo_type="space")
    return {"hub_repo": repo, "uploaded": True}


__all__ = [
    "space_deploy_command",
    "release_to_hub",
    "build_teaser",
    "build_space",
]

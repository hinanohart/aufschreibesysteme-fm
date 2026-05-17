"""``afm train`` + step-internal helpers (``train_all_regimes``, ``train_router``).

Sequential per-regime LoRA training (24 GB budget) + router head training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich import print as rprint


def train_command(
    config: Path = typer.Option(Path("configs/mvp.yaml"), "--config", "-c"),
    regime: str = typer.Option(..., "--regime", help="regime name (e.g. jpeg)"),
) -> None:
    """Train one regime's LoRA. For the full pipeline, use ``afm oss``."""
    cfg = yaml.safe_load(config.read_text())
    out = train_one(cfg, regime=regime)
    rprint(out)


def train_one(cfg: dict[str, Any], *, regime: str) -> dict[str, Any]:
    """Train a single regime LoRA. Heavy imports happen here, not at import-time."""
    from torch.utils.data import DataLoader

    from afm.core.expert_lora import ExpertLoRAManager
    from afm.core.mulan_sigma import MuLANSigmaHead
    from afm.core.physical_noise_prior import PhysicalNoisePrior
    from afm.core.schedulers import FlowMatchingScheduler
    from afm.core.trainer import TrainConfig, train_one_regime
    from afm.data import iter_samples, load_manifest
    from afm.regimes import get

    spec = get(regime)
    train_cfg = TrainConfig(
        regime=regime,
        out_dir=cfg.get("release", {}).get("ckpt_root", "ckpt"),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        batch_size=cfg["train"]["batch_size"],
        grad_accum=cfg["train"]["grad_accum"],
        steps=cfg["train"]["steps_per_regime"],
        save_steps=cfg["train"]["save_steps"],
        warmup_steps=cfg["train"]["warmup_steps"],
        precision=cfg["train"]["precision"],
        seed=cfg["train"]["seed"],
        temporal=bool(spec.temporal),
        fps=spec.codec_params.get("fps", 24),
    )

    # Base model load is intentionally inside this branch — `afm oss step env`
    # does not need to hold the model in memory.
    base, text_encoder = _load_base(cfg)
    manager = ExpertLoRAManager(base, regimes=[regime])
    mulan = MuLANSigmaHead(in_channels=base.config.in_channels)
    prior = PhysicalNoisePrior()
    scheduler = FlowMatchingScheduler(
        sigma_min=cfg["train"]["schedule"]["sigma_min"],
        rectify=cfg["train"]["schedule"].get("rectify", True),
    )

    manifest = load_manifest(regime, root=Path(cfg["data"].get("manifests_root", "data/manifests")))
    dataloader = DataLoader(
        iter_samples(manifest, batch_size=train_cfg.batch_size),
        batch_size=None,
        num_workers=cfg.get("data", {}).get("num_workers", 2),
        pin_memory=True,
    )

    return train_one_regime(
        base_model=base,
        manager=manager,
        mulan=mulan,
        prior=prior,
        scheduler=scheduler,
        dataloader=dataloader,
        cfg=train_cfg,
        text_encoder=text_encoder,
    )


def train_all_regimes(cfg: dict[str, Any], *, state) -> dict[str, Any]:
    """Step 2 body — train one LoRA per regime sequentially."""
    summary: dict[str, Any] = {}
    for regime_name in cfg["regimes"]["enabled"]:
        rprint(f"[bold]training regime[/bold] [cyan]{regime_name}[/cyan]")
        summary[regime_name] = train_one(cfg, regime=regime_name)
    return summary


def train_router(cfg: dict[str, Any], *, state) -> dict[str, Any]:
    """Step 3 body — train the router with the frozen LoRAs in place.

    The collapse monitor pauses the run when entropy < 0.3 nats for 1k steps
    — handled via ``RegimeRouter.step_monitor``. If the monitor flags a
    collapse, ``state`` will record it but not auto-pass G2; ``afm oss``
    pauses if G2 is in ``--pause-on``.
    """
    from afm.core.regime_router import RegimeRouter, RouterConfig

    n = len(cfg["regimes"]["enabled"])
    router_cfg = cfg.get("router", {})
    in_features = router_cfg.get("in_features", 2048)
    router = RegimeRouter(
        in_features=in_features,
        config=RouterConfig(
            n_regimes=n,
            hidden=router_cfg.get("hidden", 256),
            top_k=router_cfg.get("top_k", 2),
            lambda_init=router_cfg["load_balance"]["lambda_init"],
            auto_double_on_collapse=router_cfg["load_balance"]["auto_double_on_collapse"],
            collapse_entropy_threshold_nats=router_cfg["load_balance"][
                "collapse_entropy_threshold_nats"
            ],
            collapse_window_steps=router_cfg["load_balance"]["collapse_window_steps"],
        ),
    )
    # Router training body is an MVP stub: optimisation hooks into the same
    # dataloader as the LoRA step but only the gate is updated. The full
    # body is intentionally kept short — extension lives downstream.
    rprint(
        "[yellow]NOTE[/yellow]: train_router is an MVP STUB — no optimisation step is run. "
        "G2 collapse pause will only fire once the real loop is wired. The state output "
        "records this with placeholder=true so downstream tooling can distinguish a real "
        "router checkpoint from this scaffold output."
    )
    rprint(f"router initialised, n_regimes={n}, lambda_init={router.lambda_lb}")
    return {
        "n_regimes": n,
        "lambda_init": router.lambda_lb,
        "trained": False,
        "placeholder": True,
        "report": router.collapse_report(),
    }


def _load_base(cfg: dict[str, Any]):
    """Load (or stub) the base DiT + text encoder.

    Real model load happens here so ``afm --help`` is fast. If diffusers
    or the model weights aren't available, we raise a clear error rather
    than silently failing.
    """
    try:
        from diffusers import AutoPipelineForText2Image
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("training requires diffusers + transformers; install both") from e
    model_id = cfg["base_model"]["default"]
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype="auto")
    base = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if base is None:
        raise RuntimeError(f"could not locate a transformer/unet on pipeline {model_id!r}")
    tokenizer = getattr(pipe, "tokenizer", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("text encoder/tokenizer not found on the loaded pipeline")

    def encode(prompts):
        toks = tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=256
        ).to(base.device)
        return text_encoder(**toks).last_hidden_state

    return base, encode


__all__ = ["train_command", "train_one", "train_all_regimes", "train_router"]

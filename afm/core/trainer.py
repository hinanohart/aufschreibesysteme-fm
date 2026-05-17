"""Training orchestrator: bf16, gradient checkpointing, AdamW8bit, save_steps=500.

The trainer is intentionally minimal — it loops over one regime at a time
(parallel regime training at 24 GB is infeasible) and writes a LoRA per
regime to ``ckpt/{regime}/lora.safetensors``. ``afm oss step lora`` calls
``train_one_regime`` seven times.

Mid-step crash protection: ``save_steps=500`` (~30 min at ~1 s/step) is the
contract documented in README and configs/mvp.yaml. Lowering it is a
runtime decision and stays in the YAML.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from afm.core.expert_lora import ExpertLoRAManager
from afm.core.losses import fm_velocity_loss
from afm.core.mulan_sigma import MuLANSigmaHead
from afm.core.physical_noise_prior import PhysicalNoisePrior
from afm.core.schedulers import DiffusionForcingScheduler, FlowMatchingScheduler
from afm.regimes import get


@dataclass
class TrainConfig:
    regime: str
    out_dir: str
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    batch_size: int = 2
    grad_accum: int = 8
    steps: int = 30_000
    save_steps: int = 500  # ≤30 min crash bound at ~1 s/step
    warmup_steps: int = 500
    log_every: int = 50
    eval_every: int = 2_500
    seed: int = 1729
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    optimizer: str = "adamw8bit"
    # When True, use Diffusion Forcing per-frame σ (film only).
    temporal: bool = False
    fps: int = 24


def _make_optimizer(params: list[dict], cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "adamw8bit requires bitsandbytes; install with `pip install bitsandbytes`"
            ) from e
        return AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _autocast_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def _warmup_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps and cfg.warmup_steps > 0:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def _save_lora(manager: ExpertLoRAManager, mulan: MuLANSigmaHead, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manager.save(str(out_dir / "lora"))
    # Safetensors over torch.save — no pickle in the artifact stream.
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in mulan.state_dict().items()}
    save_file(state, str(out_dir / "mulan.safetensors"), metadata={"step": str(step)})


def train_one_regime(
    *,
    base_model: nn.Module,
    manager: ExpertLoRAManager,
    mulan: MuLANSigmaHead,
    prior: PhysicalNoisePrior,
    scheduler: FlowMatchingScheduler,
    dataloader: DataLoader,
    cfg: TrainConfig,
    text_encoder: Callable[[Any], torch.Tensor] | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    """Train a single regime LoRA + MuLAN head.

    Returns a small dict ``{"final_step": ..., "loss_ema": ..., "saved": ...}``
    that the oss orchestrator uses to update ``state.json``.

    The trainer never touches the router — that's a separate step (step 3).
    """
    spec = get(cfg.regime)
    device = next(base_model.parameters()).device
    autocast_dtype = _autocast_dtype(cfg.precision)
    torch.manual_seed(cfg.seed)

    if cfg.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    manager.set_active([cfg.regime])
    trainable = [
        {"params": [p for p in manager.peft_model.parameters() if p.requires_grad]},
        {"params": list(mulan.parameters())},
    ]
    opt = _make_optimizer(trainable, cfg)

    sched_temporal = DiffusionForcingScheduler(scheduler) if cfg.temporal else None

    loss_ema = 0.0
    save_dir = Path(cfg.out_dir) / cfg.regime
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)
    sigma_min = float(spec.sigma_min)
    sigma_min_t = torch.tensor(sigma_min)

    data_iter = _cycle(dataloader)
    while step < cfg.steps:
        batch = next(data_iter)
        x0 = batch["x"].to(device, non_blocking=True)
        text = batch.get("text_emb")
        if text is None and text_encoder is not None:
            text = text_encoder(batch["prompt"])
        if text is None:
            raise RuntimeError("dataloader returned no text_emb and no text_encoder supplied")
        text = text.to(device, non_blocking=True)

        noise = prior.sample(cfg.regime, x0.shape, device=device, dtype=x0.dtype)
        if cfg.temporal and sched_temporal is not None:
            frames = x0.shape[1] if x0.ndim == 5 else 1
            t = sched_temporal.sample_t(x0.shape[0], frames, device=device, dtype=x0.dtype)
            xt = sched_temporal.add_noise(x0, noise, t)
            target = sched_temporal.target(x0, noise, t)
        else:
            t = scheduler.sample_t(x0.shape[0], device=device, dtype=x0.dtype)
            xt = scheduler.add_noise(x0, noise, t)
            target = scheduler.target(x0, noise, t)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            v_out = base_model(
                hidden_states=xt,
                timestep=t,
                encoder_hidden_states=text,
            )
            v_pred = v_out.sample if hasattr(v_out, "sample") else v_out
            # MuLAN per-pixel σ, clipped from below by the regime physical floor.
            sigma = mulan(v_pred.detach().float(), sigma_min=sigma_min_t.to(device))
            loss = fm_velocity_loss(v_pred, target, sigma=sigma)

        (loss / cfg.grad_accum).backward()
        accum += 1
        if accum >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in manager.peft_model.parameters() if p.requires_grad]
                + list(mulan.parameters()),
                max_norm=1.0,
            )
            _set_lr(opt, _warmup_lr(step, cfg))
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            loss_ema = (
                0.98 * loss_ema + 0.02 * float(loss.detach().item())
                if step > 1
                else float(loss.detach().item())
            )

            if step % cfg.log_every == 0 and progress_cb:
                progress_cb({"step": step, "loss_ema": loss_ema, "regime": cfg.regime})

            # save_steps=500 → ≤30 min crash bound at ~1 s/step.
            if step % cfg.save_steps == 0 or step == cfg.steps:
                _save_lora(manager, mulan, save_dir, step)
                if progress_cb:
                    progress_cb({"saved": True, "step": step, "regime": cfg.regime})

    return {"final_step": step, "loss_ema": loss_ema, "saved": str(save_dir)}


def _cycle(loader: Iterable):
    while True:
        for x in loader:
            yield x


__all__ = ["TrainConfig", "train_one_regime"]

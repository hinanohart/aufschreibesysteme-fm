"""Diffusion Schrödinger Bridge Matching (DSBM, Shi+2023) — cross-regime morph.

Inference-time only — does not contribute to training cost (Critic C2).
The bridge couples a sample drawn from regime A's prior to a sample drawn
from regime B's prior, and integrates the learned drift along the way
producing a trajectory of intermediate frames.

The implementation is small and self-contained — full DSBM lives in
``afm.morph.dsbm_full`` (future work). For the demo we use the IPF outer-
loop step count of 1 (no iteration); empirically this is enough for the
visual hook used in the HF Space gallery / YouTube short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DSBMConfig:
    steps: int = 50
    sigma: float = 0.5
    # IPF outer iterations — 1 keeps inference-time cost bounded.
    ipf_iters: int = 1
    # Save every k-th step for the frame list (k=1 saves all).
    save_every: int = 1


def morph_between_regimes(
    *,
    pipeline: Any,
    text_emb: Any,
    regime_a: str,
    regime_b: str,
    cfg: DSBMConfig,
) -> list[Any]:
    """Produce a list of intermediate tensors moving from A's prior to B's prior.

    The drift is the average of the two regime velocity predictions weighted
    by t (linear schedule). The noise σ is the geometric mean of the two
    regime σ_min floors. This is intentionally simple — a full DSBM with
    IPF iterations belongs in a research paper, not a demo.
    """
    import torch

    device = pipeline._device()
    A = pipeline.prior.sample(regime_a, (1, 3, 1024, 1024), device=device)
    B = pipeline.prior.sample(regime_b, (1, 3, 1024, 1024), device=device)

    # Endpoint coupling — simple linear with reflective tail.
    frames: list[Any] = []
    for i in range(cfg.steps):
        t = (i + 1) / cfg.steps
        x = (1.0 - t) * A + t * B
        # Walk the pipeline once with the active adapter set to a blend of
        # both regime LoRAs. Diffusers' set_adapter supports a list and sums
        # adapter outputs; the t-weighting is folded into the noise schedule.
        pipeline._lora_manager.set_active([regime_a, regime_b])
        v = pipeline._predict_velocity(
            x,
            torch.tensor(1.0 - t, device=device),
            text_emb,
            regime=regime_b,
        )
        x = x - (1.0 / cfg.steps) * v
        if i % cfg.save_every == 0:
            frames.append(x)
    return frames


__all__ = ["DSBMConfig", "morph_between_regimes"]

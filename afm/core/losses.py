"""FM / CFM loss helpers + regime-conditioned weighting."""

from __future__ import annotations

import torch


def fm_velocity_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    *,
    sigma: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard FM velocity MSE with optional per-pixel σ weighting.

    If ``sigma`` is provided the loss is divided element-wise by σ² (the
    MuLAN per-pixel σ reweighting). σ must be broadcastable to ``v_pred``.
    """
    diff = (v_pred - v_target) ** 2
    if sigma is not None:
        diff = diff / (sigma**2 + 1.0e-8)
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    raise ValueError(f"unknown reduction {reduction!r}")


def cfm_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    cond_drop_mask: torch.Tensor,
    uncond_v_pred: torch.Tensor | None = None,
    *,
    cfg_weight: float = 1.0,
) -> torch.Tensor:
    """Classifier-free guidance training loss.

    With probability `cond_drop_mask`, the conditional prediction is dropped
    and the model sees the unconditional branch. We expect the trainer to
    have already produced both branches; we just blend losses.
    """
    cond_loss = fm_velocity_loss(v_pred, v_target)
    if uncond_v_pred is None:
        return cond_loss
    uncond_loss = fm_velocity_loss(uncond_v_pred, v_target)
    return cond_loss + cfg_weight * uncond_loss


__all__ = ["fm_velocity_loss", "cfm_loss"]

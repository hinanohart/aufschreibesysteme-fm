"""Flow Matching + Diffusion Forcing schedulers.

We use rectified Flow Matching as the default for all spatial regimes and
Diffusion Forcing (per-frame σ) for the single temporal regime (``film``).
The two scheduler shapes share a tiny interface so the trainer can call
``step(...)`` uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class FlowMatchingSchedulerProto(Protocol):
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...


@dataclass
class FlowMatchingScheduler:
    """Rectified Flow Matching (Liu+2022) — straight-line interpolant.

    x_t = (1 - t) * x0 + t * noise      (t ∈ [σ_min, 1])
    target velocity v = noise - x0
    """

    sigma_min: float = 1.0e-4
    rectify: bool = True

    def sample_t(self, batch: int, *, device, dtype=torch.float32) -> torch.Tensor:
        # Uniform in [sigma_min, 1] — Liu+2022 ablation finds uniform works
        # well after rectification.
        t = torch.rand(batch, device=device, dtype=dtype)
        t = self.sigma_min + (1.0 - self.sigma_min) * t
        return t

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # t shape (B,) → broadcast across remaining dims.
        view = (-1,) + (1,) * (x0.ndim - 1)
        t = t.view(*view).to(x0)
        return (1.0 - t) * x0 + t * noise

    def target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Velocity target — does not depend on t for rectified FM.
        return noise - x0


@dataclass
class DiffusionForcingScheduler:
    """Per-frame σ — Diffusion Forcing variant.

    Each frame of a temporal sample independently draws t. This decorrelates
    grain across frames the way film grain actually behaves (within the
    per-frame correlation budget of the regime).
    """

    base: FlowMatchingScheduler

    def sample_t(self, batch: int, frames: int, *, device, dtype=torch.float32) -> torch.Tensor:
        t = torch.rand(batch, frames, device=device, dtype=dtype)
        t = self.base.sigma_min + (1.0 - self.base.sigma_min) * t
        return t

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x0 shape (B, F, C, H, W), t shape (B, F)
        if t.ndim != 2:
            raise ValueError(f"expected per-frame t (B, F), got {t.shape}")
        t = t.view(t.shape[0], t.shape[1], 1, 1, 1).to(x0)
        return (1.0 - t) * x0 + t * noise

    def target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return noise - x0


__all__ = [
    "FlowMatchingScheduler",
    "DiffusionForcingScheduler",
    "FlowMatchingSchedulerProto",
]

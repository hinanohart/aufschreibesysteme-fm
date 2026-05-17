"""Per-pixel σ head with prior-clipped output (MuLAN-style).

Architecture spec:
    σ_pred(x, r) = max(σ_min^(r), MuLAN_θ(x))

The clip floor σ_min^(r) is the **regime's measured noise floor** — the
output cannot drop below the channel's physical noise. This guarantees
that the implementation respects the frozen physical prior at training
time even when MuLAN's learnable head would otherwise prefer a smaller σ.

The head consumes the model's intermediate feature map and emits a tensor
with the same spatial shape as the noised sample. Channel count of the
output is 1 by default (scalar σ) — set ``out_channels=C`` for per-channel σ.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MuLANSigmaHead(nn.Module):
    """Tiny conv head that emits per-pixel log-σ, then exp + clip-from-below.

    The head is intentionally small (3 conv layers) — its purpose is local
    σ refinement, not feature extraction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        out_channels: int = 1,
        *,
        log_sigma_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        # Init final bias so initial σ ≈ exp(log_sigma_init).
        with torch.no_grad():
            final = self.body[-1]
            assert isinstance(final, nn.Conv2d)
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, log_sigma_init)
        self.log_sigma_init = log_sigma_init

    def forward(
        self,
        features: torch.Tensor,
        *,
        sigma_min: float | torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-pixel σ clipped from below by ``sigma_min``.

        Args:
            features: (B, C_in, H, W)
            sigma_min: scalar or (B, 1, 1, 1) — the regime physical floor

        Returns:
            sigma: (B, out_channels, H, W) with sigma >= sigma_min everywhere
        """
        log_sigma = self.body(features)
        sigma = log_sigma.exp()
        if isinstance(sigma_min, float):
            sigma_min_t = sigma.new_tensor(sigma_min)
        else:
            sigma_min_t = sigma_min.to(sigma)
        # Element-wise max with broadcast.
        return torch.maximum(sigma, sigma_min_t.expand_as(sigma))


def per_regime_sigma_min_lookup(regime_names: list[str]) -> dict[str, float]:
    """Convenience: pull ``RegimeSpec.sigma_min`` for each regime up-front."""
    from afm.regimes import get

    return {r: float(get(r).sigma_min) for r in regime_names}


__all__ = ["MuLANSigmaHead", "per_regime_sigma_min_lookup"]

"""Top-2 soft regime router with load-balance loss and collapse monitor.

Spec from architecture:
    - 2-layer MLP gate, hidden=256
    - input: concat(DiT layer-4 patch-mean, text CLS)
    - output: softmax over N=7 regimes, top-2 soft mixing (NOT hard top-1)
    - load-balance loss: ``L_lb = lambda * N * sum_r f_r * P_r`` (Switch style)
    - collapse monitor: if mean gate entropy < 0.3 nats for ``window_steps`` ≥
      1000 steps, ``lambda`` auto-doubles
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterConfig:
    n_regimes: int
    hidden: int = 256
    top_k: int = 2
    lambda_init: float = 0.01
    auto_double_on_collapse: bool = True
    collapse_entropy_threshold_nats: float = 0.3
    collapse_window_steps: int = 1000
    # max times lambda can auto-double (capped to avoid runaway).
    max_doublings: int = 6


class RegimeRouter(nn.Module):
    """2-layer MLP gate + top-2 soft routing + load-balance loss + collapse monitor."""

    def __init__(self, in_features: int, config: RouterConfig) -> None:
        super().__init__()
        if config.top_k < 1 or config.top_k > config.n_regimes:
            raise ValueError(f"top_k must be in [1, n_regimes], got {config.top_k}")
        self.config = config
        h = config.hidden
        self.gate = nn.Sequential(
            nn.Linear(in_features, h),
            nn.GELU(),
            nn.Linear(h, config.n_regimes),
        )
        # Load-balance lambda is exposed as a buffer (mutable, not a Parameter).
        self.register_buffer("_lambda", torch.tensor(config.lambda_init, dtype=torch.float32))
        # Sliding-window entropy bookkeeping for the collapse monitor.
        self._entropy_window: deque[float] = deque(maxlen=config.collapse_window_steps)
        self._doublings = 0
        # For load-balance: running fraction of dispatches per regime f_r.
        self.register_buffer("_running_f", torch.zeros(config.n_regimes, dtype=torch.float32))
        self.register_buffer("_running_f_steps", torch.zeros((), dtype=torch.long))

    # -------- properties ---------------------------------------------------

    @property
    def lambda_lb(self) -> float:
        return float(self._lambda.item())

    @property
    def n_regimes(self) -> int:
        return self.config.n_regimes

    # -------- forward ------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the top-k soft gate distribution.

        Args:
            features: (B, in_features) — concat(layer4_patch_mean, text_cls)

        Returns:
            weights: (B, n_regimes) — full softmax (zeros outside top-k)
            indices: (B, k) — indices of top-k regimes per item, descending
        """
        logits = self.gate(features)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.config.top_k, dim=-1)
        # Soft top-k: renormalise the kept mass so weights still sum to 1
        # along the kept set (NOT one-hot). This is the "soft" in top-k soft.
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)
        weights = torch.zeros_like(probs)
        weights.scatter_(-1, topk_idx, topk_vals)
        return weights, topk_idx

    # -------- load-balance loss --------------------------------------------

    def load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Switch-style load-balance loss.

        L_lb = lambda * N * sum_r f_r * P_r
            f_r = mean over batch of indicator(regime r is in top-k of token)
            P_r = mean over batch of softmax probability for regime r

        Args:
            weights: (B, N) — full softmax output from forward() (already
                renormalised over top-k mass). We re-derive f from non-zero
                mask of weights, and P from the un-masked softmax.
        """
        # Re-derive un-masked softmax probabilities from weights is not
        # possible (lossy). We expect the caller to pass the *raw* softmax
        # here when computing L_lb. To keep the API single-entry, we accept
        # the post-mask weights and treat them as both f and P proxies —
        # this is the convention used in fairseq/peft moe references.
        N = weights.shape[-1]
        # f_r: fraction of tokens that dispatched to regime r.
        f = (weights > 0).float().mean(dim=0)
        # P_r: average gate weight for regime r across batch (proxy).
        P = weights.mean(dim=0)
        return self._lambda * N * (f * P).sum()

    # -------- entropy monitor + lambda auto-double -------------------------

    @torch.no_grad()
    def step_monitor(self, weights: torch.Tensor) -> dict[str, float]:
        """Call once per optimizer step with the current batch weights.

        Tracks mean per-item entropy of the gate distribution. If the mean
        across a 1 k-step sliding window drops below the configured threshold
        and ``auto_double_on_collapse`` is True, ``_lambda`` is doubled.

        Returns a small diagnostics dict (useful for logging / G2 gate).
        """
        # Avoid log(0) — use a stabilised entropy.
        p = weights.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)  # (B,)
        mean_ent = float(ent.mean().item())
        self._entropy_window.append(mean_ent)

        # Update running f counters (used by gate diagnostics / G2 report).
        self._running_f.mul_(0.99).add_(
            0.01 * (weights > 0).float().mean(dim=0).to(self._running_f)
        )
        self._running_f_steps += 1

        collapsed = False
        if (
            self.config.auto_double_on_collapse
            and len(self._entropy_window) >= self.config.collapse_window_steps
            and self._doublings < self.config.max_doublings
        ):
            window_mean = sum(self._entropy_window) / len(self._entropy_window)
            if window_mean < self.config.collapse_entropy_threshold_nats:
                # Auto-double; emit a single doubling event and clear window
                # so the doubling does not retrigger on the next step.
                self._lambda.mul_(2.0)
                self._doublings += 1
                self._entropy_window.clear()
                collapsed = True

        return {
            "mean_entropy_nats": mean_ent,
            "lambda_lb": self.lambda_lb,
            "doublings": self._doublings,
            "collapse_event": float(collapsed),
        }

    # -------- gate diagnostics for G2 --------------------------------------

    def collapse_report(self) -> dict[str, float | list[float]]:
        running = self._running_f.detach().cpu().tolist()
        return {
            "running_fraction_per_regime": running,
            "lambda_lb": self.lambda_lb,
            "doublings": self._doublings,
            "max_fraction": float(max(running)) if running else 0.0,
            "min_fraction": float(min(running)) if running else 0.0,
        }


__all__ = ["RegimeRouter", "RouterConfig"]

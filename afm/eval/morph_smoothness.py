"""M3 — morphing smoothness via LPIPS-trajectory monotonicity and PPL.

Given a list of frames produced by ``afm morph --from A --to B --steps N``,
we want:
    (a) consecutive-frame LPIPS to be monotonic in t
    (b) total Perceptual Path Length (PPL) to be bounded

Monotonicity is computed as the fraction of consecutive pairs whose LPIPS
distance respects the expected ordering. The MVP success criterion is
≥ 0.90.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MorphSmoothnessReport:
    monotonicity: float
    ppl_total: float
    target_monotonicity: float = 0.90

    def passes(self) -> bool:
        return self.monotonicity >= self.target_monotonicity

    def to_json(self) -> dict[str, Any]:
        return {
            "monotonicity": self.monotonicity,
            "ppl_total": self.ppl_total,
            "target_monotonicity": self.target_monotonicity,
            "passes": self.passes(),
        }


def _lpips_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    import lpips

    net = lpips.LPIPS(net="alex").eval()
    # LPIPS expects images in [-1, 1].
    x_ = x.clamp(0, 1) * 2 - 1
    y_ = y.clamp(0, 1) * 2 - 1
    with torch.no_grad():
        return net(x_, y_).flatten()


def evaluate_trajectory(frames: list[torch.Tensor]) -> MorphSmoothnessReport:
    if len(frames) < 3:
        raise ValueError("need >= 3 frames to compute monotonicity / PPL")
    # consecutive LPIPS distances
    dists: list[float] = []
    for i in range(len(frames) - 1):
        d = _lpips_pair(frames[i], frames[i + 1])
        dists.append(float(d.mean()))
    ppl_total = float(sum(dists))
    # Monotonicity = fraction of consecutive pairs that don't reverse the
    # expected direction; for a near-straight bridge we just check the
    # sliding-window second-difference sign.
    diffs = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
    n_monotone = (
        sum(1 for d in diffs if d >= 0) if sum(diffs) > 0 else sum(1 for d in diffs if d <= 0)
    )
    monotonicity = n_monotone / max(1, len(diffs))
    return MorphSmoothnessReport(monotonicity=monotonicity, ppl_total=ppl_total)


__all__ = ["MorphSmoothnessReport", "evaluate_trajectory"]

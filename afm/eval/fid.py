"""M2 — per-regime FID against a held-out real reference of 5 000 images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _inception_features(images: torch.Tensor) -> np.ndarray:
    """Run images through Inception-V3 pool3, return (N, 2048) numpy.

    Falls back to a random projection if torchvision Inception is not
    available — useful for smoke tests, not for real evaluation.
    """
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError:  # pragma: no cover
        return torch.randn(images.shape[0], 2048).numpy()
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True).eval()
    model.fc = torch.nn.Identity()
    with torch.no_grad():
        feats = model(images)
    return feats.cpu().numpy()


def _fid_from_features(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    """Standard FID = ||mu1-mu2||^2 + Tr(C1+C2-2sqrt(C1*C2))."""
    from scipy.linalg import sqrtm

    mu1, mu2 = f_real.mean(0), f_fake.mean(0)
    c1 = np.cov(f_real, rowvar=False)
    c2 = np.cov(f_fake, rowvar=False)
    diff = mu1 - mu2
    cov_mean, _ = sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * cov_mean))


@dataclass
class FIDReport:
    fid_per_regime: dict[str, float]
    baseline_fid_per_regime: dict[str, float]

    def passes(self) -> bool:
        return all(
            self.fid_per_regime[r] <= self.baseline_fid_per_regime.get(r, float("inf"))
            for r in self.fid_per_regime
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "fid_per_regime": self.fid_per_regime,
            "baseline_fid_per_regime": self.baseline_fid_per_regime,
            "passes": self.passes(),
        }


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    if real_images.ndim != 4 or fake_images.ndim != 4:
        raise ValueError("images must be (N, C, H, W)")
    f_real = _inception_features(real_images)
    f_fake = _inception_features(fake_images)
    return _fid_from_features(f_real, f_fake)


__all__ = ["compute_fid", "FIDReport"]

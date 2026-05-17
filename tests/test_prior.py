"""Physical noise prior is frozen. No nn.Parameter, no requires_grad."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

import torch.nn as nn

from afm.core.physical_noise_prior import PhysicalNoisePrior


def test_psd_buffers_are_not_parameters() -> None:
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    for name, _p in prior.named_parameters():
        # Anything that ends up as Parameter would land here.
        raise AssertionError(f"prior introduced a Parameter named {name!r}")


def test_psd_buffers_do_not_require_grad() -> None:
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    for n in prior.regime_names:
        buf = prior.psd(n)
        assert buf.requires_grad is False
        assert not isinstance(buf, nn.Parameter)


def test_assert_frozen_passes_on_clean_init() -> None:
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    prior._assert_frozen()  # should not raise


def test_sample_runs_under_no_grad() -> None:
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    eps = prior.sample("jpeg", (2, 3, 16, 16))
    assert eps.shape == (2, 3, 16, 16)
    assert eps.requires_grad is False
    # unit-ish variance after normalisation
    assert 0.5 < eps.std().item() < 2.0


def test_sample_has_zero_mean_per_sample() -> None:
    """Regression: 1/f PSDs left a ±15-50 DC offset on every sample, which
    inflated eps.std() far past the 0.5..2.0 bound and silently corrupted
    velocity-loss targets. After fix, per-sample mean must be ~0."""
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    eps = prior.sample("jpeg", (4, 3, 16, 16))
    per_sample_mean = eps.flatten(start_dim=-2).mean(dim=-1).abs()
    assert per_sample_mean.max().item() < 1e-4, (
        f"per-sample DC offset must be ~0; got max abs {per_sample_mean.max().item()}"
    )


def test_synthetic_warning_lists_missing_priors() -> None:
    prior = PhysicalNoisePrior(spatial_shape=(16, 16))
    # No measurements installed → all seven regimes are synthetic.
    synthetic = prior.warn_if_synthetic()
    assert isinstance(synthetic, list)
    assert len(synthetic) >= 1  # at least one regime lacks a measurement file

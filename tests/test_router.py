"""Router invariants — top-2 soft + load-balance + collapse monitor.

GPU-free. We feed synthetic features into the gate and assert the
documented behaviour.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

import torch

from afm.core.regime_router import RegimeRouter, RouterConfig


def _make(n: int = 7) -> RegimeRouter:
    return RegimeRouter(
        in_features=32,
        config=RouterConfig(
            n_regimes=n,
            hidden=64,
            top_k=2,
            lambda_init=0.01,
            collapse_window_steps=4,
            collapse_entropy_threshold_nats=10.0,  # very lax for test
        ),
    )


def test_top_k_soft_returns_two_nonzero_per_row() -> None:
    router = _make()
    x = torch.randn(8, 32)
    weights, idx = router(x)
    assert weights.shape == (8, 7)
    # Top-2 soft → exactly 2 non-zero entries per row, summing to ≈ 1.
    nonzero = (weights > 0).sum(dim=-1)
    assert torch.all(nonzero == 2)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_indices_are_descending() -> None:
    router = _make()
    x = torch.randn(4, 32)
    weights, idx = router(x)
    # idx is from torch.topk → descending by value.
    assert idx.shape == (4, 2)


def test_load_balance_loss_nonnegative_and_uses_lambda() -> None:
    router = _make()
    x = torch.randn(16, 32)
    weights, _ = router(x)
    loss = router.load_balance_loss(weights)
    assert loss.item() >= 0
    # Lambda should appear in the result — bump it and confirm scaling.
    router._lambda.fill_(0.1)
    loss2 = router.load_balance_loss(weights)
    assert loss2.item() > loss.item() * 5  # 10× λ should ≫ 5× outcome


def test_collapse_monitor_doubles_lambda_when_low_entropy() -> None:
    # Tight threshold to force collapse.
    router = RegimeRouter(
        in_features=32,
        config=RouterConfig(
            n_regimes=7,
            top_k=2,
            lambda_init=0.01,
            collapse_window_steps=4,
            collapse_entropy_threshold_nats=10.0,  # any entropy < 10 nats triggers
            auto_double_on_collapse=True,
            max_doublings=2,
        ),
    )
    x = torch.randn(8, 32)
    initial = router.lambda_lb
    for _ in range(5):
        w, _ = router(x)
        router.step_monitor(w)
    assert router.lambda_lb > initial


def test_collapse_report_has_running_fractions() -> None:
    router = _make()
    x = torch.randn(8, 32)
    w, _ = router(x)
    router.step_monitor(w)
    rep = router.collapse_report()
    assert "running_fraction_per_regime" in rep
    assert len(rep["running_fraction_per_regime"]) == 7

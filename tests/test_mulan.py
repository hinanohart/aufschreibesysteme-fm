"""MuLAN σ — must clip from below by regime physical floor."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

import torch

from afm.core.mulan_sigma import MuLANSigmaHead, per_regime_sigma_min_lookup


def test_sigma_never_drops_below_floor() -> None:
    head = MuLANSigmaHead(in_channels=4, log_sigma_init=-10.0)
    # log_sigma_init very small → exp ~ 0; clip must kick in.
    feat = torch.randn(2, 4, 16, 16)
    sigma_min = 0.01
    out = head(feat, sigma_min=sigma_min)
    assert out.min().item() >= sigma_min - 1e-6


def test_sigma_min_lookup_present_for_all_regimes() -> None:
    floors = per_regime_sigma_min_lookup(
        ["parchment", "typewriter", "gramophone", "photograph", "film", "jpeg", "crt"]
    )
    assert len(floors) == 7
    for _r, v in floors.items():
        assert v > 0.0


def test_sigma_min_is_per_regime_tensor() -> None:
    head = MuLANSigmaHead(in_channels=4)
    feat = torch.randn(2, 4, 16, 16)
    # Pass a (B, 1, 1, 1) tensor — should broadcast and clip element-wise.
    sm = torch.tensor([0.01, 0.05]).view(2, 1, 1, 1)
    out = head(feat, sigma_min=sm)
    # Each batch row clipped at its own floor.
    assert out[0].min().item() >= 0.01 - 1e-6
    assert out[1].min().item() >= 0.05 - 1e-6

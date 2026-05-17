"""Registry invariants — no GPU required."""

from __future__ import annotations

import pytest

from afm.regimes import (
    EPOCH_GROUPS,
    N_REGIMES,
    all_regimes,
    assert_registry_complete,
    get,
    regime_names,
)


def test_exactly_seven_regimes() -> None:
    assert N_REGIMES == 7
    assert len(all_regimes()) == 7
    assert len(regime_names()) == 7


def test_registry_complete_passes() -> None:
    assert_registry_complete()


def test_epoch_grouping_matches_seven() -> None:
    members = {n for group in EPOCH_GROUPS.values() for n in group}
    assert members == set(regime_names())
    assert set(EPOCH_GROUPS) == {"1800", "1900", "2000"}


def test_film_is_temporal_photograph_is_not() -> None:
    # Critic C4 — film and photograph must be functionally separated.
    assert get("film").temporal is True
    assert get("photograph").temporal is False


def test_jpeg_codec_params() -> None:
    p = get("jpeg").codec_params
    assert p["block"] == 8
    assert sorted(p["qualities"]) == [10, 50, 90]


def test_gramophone_is_mel_codec() -> None:
    assert get("gramophone").codec == "mel"


def test_lowercase_name_invariant() -> None:
    for spec in all_regimes():
        assert spec.name == spec.name.lower()


def test_unknown_regime_raises() -> None:
    with pytest.raises(KeyError):
        get("daguerreotype")

"""Regime registry public API.

Importing this module force-imports the 7 regime modules so each one's
``register(...)`` call fires. Downstream code can then do::

    from afm.regimes import all_regimes, get, assert_registry_complete

The invariant ``len(all_regimes()) == 7`` is checked at CLI startup.
"""

from afm.regimes.base import (
    EPOCH_GROUPS,
    N_REGIMES,
    RegimeSpec,
    _load_all,
    all_regimes,
    assert_registry_complete,
    epoch_of,
    get,
    regime_names,
    register,
)

_load_all()

__all__ = [
    "EPOCH_GROUPS",
    "N_REGIMES",
    "RegimeSpec",
    "all_regimes",
    "assert_registry_complete",
    "epoch_of",
    "get",
    "register",
    "regime_names",
]

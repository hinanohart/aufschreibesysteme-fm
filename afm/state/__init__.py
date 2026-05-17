"""state.json atomic manager — schema version "0.2"."""

from afm.state.manager import (
    SCHEMA_VERSION,
    StateError,
    StateManager,
    StepRecord,
)

__all__ = ["SCHEMA_VERSION", "StateError", "StateManager", "StepRecord"]

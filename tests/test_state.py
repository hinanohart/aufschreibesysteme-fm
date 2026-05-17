"""state.json — atomic write, v0.2 schema, 3-branch resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afm.state.manager import (
    GATE_IDS,
    SCHEMA_VERSION,
    STEP_IDS,
    StateError,
    StateManager,
)


def test_schema_version_is_exactly_0_2() -> None:
    assert SCHEMA_VERSION == "0.2"


def test_steps_include_5_5() -> None:
    assert "5.5" in STEP_IDS


def test_gates_are_g1_to_g4() -> None:
    assert set(GATE_IDS) == {"G1", "G2", "G3", "G4"}


def test_branch_fresh_when_no_state_file(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    assert mgr.branch() == "fresh"


def test_branch_partial_after_mark_running(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    mgr.mark_running("0")
    assert (tmp_path / "state.json").is_file()
    mgr2 = StateManager(tmp_path / "state.json")
    assert mgr2.branch() == "partial"


def test_mark_done_advances_current_step(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    mgr.mark_done("0")
    assert mgr.doc.current_step == 1.0
    mgr.mark_done("1")
    assert mgr.doc.current_step == 2.0


def test_branch_done_after_step_6(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    for sid in STEP_IDS:
        mgr.mark_done(sid)
    assert mgr.branch() == "done"


def test_branch_partial_after_5_5_done_but_6_not(tmp_path: Path) -> None:
    """Regression: mark_done('5.5') leaves current_step==6.0; old branch()
    used `current_step >= 6` and returned 'done' before step 6 ran. The
    fixed branch() must wait until steps['6'].status == 'done'."""
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    for sid in ("0", "1", "2", "3", "4", "5", "5.5"):
        mgr.mark_done(sid)
    assert mgr.branch() == "partial", (
        "branch() returned 'done' before step 6 ran — off-by-one regression"
    )


def test_pass_gate_appends_once(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    mgr.pass_gate("G1")
    mgr.pass_gate("G1")
    assert mgr.doc.gates_passed == ["G1"]


def test_unknown_step_raises(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    with pytest.raises(StateError):
        mgr.mark_running("99")


def test_version_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"version": "0.1", "current_step": 0, "steps": {}}))
    mgr = StateManager(p)
    with pytest.raises(StateError):
        mgr.load()


def test_atomic_write_leaves_no_tmp_file(tmp_path: Path) -> None:
    mgr = StateManager(tmp_path / "state.json")
    mgr.load()
    mgr.mark_running("0")
    leftovers = list(tmp_path.glob("state.json.*.tmp"))
    assert leftovers == []

"""``state.json`` manager — schema version "0.2".

Implements the 3-branch resume protocol documented in
``docs/architecture.md``:

    test -f state.json
      ├─ absent           → fresh         (start at step 0)
      ├─ current_step<6   → partial       (resume)
      └─ current_step==6  → done          (eval-only / ablation)

Writes are atomic (tmp + rename). The file MUST stay consistent through
crashes so ``afm oss --resume`` is deterministic. Failures are appended to
``failures`` rather than overwriting any state, and the corresponding R8
crash dump (``experiments/_wip/<step>-<ts>/``) is referenced by id.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION: str = "0.2"
STEP_IDS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "5.5", "6")
GATE_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4")

StatusLiteral = Literal["pending", "running", "done", "failed"]


class StateError(RuntimeError):
    """Anything wrong with state.json shape/version/path."""


@dataclass
class StepRecord:
    status: StatusLiteral = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureRecord:
    step: str
    timestamp: float
    error: str
    wip_dir: str


@dataclass
class StateDoc:
    version: str = SCHEMA_VERSION
    current_step: float = 0.0
    steps: dict[str, StepRecord] = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)
    gates_passed: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "current_step": self.current_step,
            "steps": {sid: asdict(rec) for sid, rec in self.steps.items()},
            "failures": [asdict(f) for f in self.failures],
            "gates_passed": list(self.gates_passed),
        }
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> StateDoc:
        if raw.get("version") != SCHEMA_VERSION:
            raise StateError(
                f"state.json version mismatch — got {raw.get('version')!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )
        steps = {
            sid: StepRecord(
                status=rec.get("status", "pending"),
                started_at=rec.get("started_at"),
                finished_at=rec.get("finished_at"),
                extra=rec.get("extra", {}),
            )
            for sid, rec in raw.get("steps", {}).items()
        }
        failures = [
            FailureRecord(
                step=f["step"],
                timestamp=f["timestamp"],
                error=f["error"],
                wip_dir=f["wip_dir"],
            )
            for f in raw.get("failures", [])
        ]
        return cls(
            version=raw["version"],
            current_step=float(raw.get("current_step", 0)),
            steps=steps,
            failures=failures,
            gates_passed=list(raw.get("gates_passed", [])),
        )


# ----- atomic IO -------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile w/ delete=False, write, fsync, rename. Stays on the
    # same filesystem so the rename is atomic on POSIX.
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ----- manager ---------------------------------------------------------------


class StateManager:
    """Owns the lifetime of ``state.json``. Cheap to construct."""

    def __init__(self, path: str | Path = "state.json") -> None:
        self.path = Path(path)
        self._doc: StateDoc | None = None

    # -------- IO -----------------------------------------------------------

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> StateDoc:
        if not self.path.is_file():
            self._doc = StateDoc()
            return self._doc
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            raise StateError(f"failed to read state.json at {self.path}: {e}") from e
        doc = StateDoc.from_json(raw)
        self._doc = doc
        return doc

    def save(self) -> None:
        if self._doc is None:
            raise StateError("no in-memory state to save; call load() first")
        _atomic_write_json(self.path, self._doc.as_json())

    @property
    def doc(self) -> StateDoc:
        if self._doc is None:
            return self.load()
        return self._doc

    # -------- 3-branch resume protocol -------------------------------------

    def branch(self) -> Literal["fresh", "partial", "done"]:
        if not self.exists():
            return "fresh"
        doc = self.load()
        # "done" is decided by terminal step status, not current_step.
        # mark_done("5.5") leaves current_step == 6.0 which would otherwise
        # short-circuit step 6 (arxiv); we require step "6" status="done".
        last_step = STEP_IDS[-1]
        rec = doc.steps.get(last_step)
        if rec is not None and rec.status == "done":
            return "done"
        return "partial"

    # -------- step lifecycle ----------------------------------------------

    def mark_running(self, step_id: str, **extra: Any) -> None:
        if step_id not in STEP_IDS:
            raise StateError(f"unknown step id {step_id!r}; known: {STEP_IDS}")
        doc = self.doc
        rec = doc.steps.setdefault(step_id, StepRecord())
        rec.status = "running"
        rec.started_at = time.time()
        rec.extra.update(extra)
        self.save()

    def mark_done(self, step_id: str, **extra: Any) -> None:
        doc = self.doc
        rec = doc.steps.setdefault(step_id, StepRecord())
        rec.status = "done"
        rec.finished_at = time.time()
        rec.extra.update(extra)
        # current_step advances to the float of the *next* step.
        idx = STEP_IDS.index(step_id)
        if idx + 1 < len(STEP_IDS):
            try:
                doc.current_step = float(STEP_IDS[idx + 1])
            except ValueError:
                # "5.5" parses fine but if for some reason it doesn't,
                # bump by 1 as a safe fallback.
                doc.current_step = float(int(doc.current_step) + 1)
        else:
            doc.current_step = 6.0
        self.save()

    def mark_failed(self, step_id: str, error: str, wip_dir: str) -> None:
        doc = self.doc
        rec = doc.steps.setdefault(step_id, StepRecord())
        rec.status = "failed"
        rec.finished_at = time.time()
        doc.failures.append(
            FailureRecord(
                step=step_id,
                timestamp=time.time(),
                error=str(error),
                wip_dir=str(wip_dir),
            )
        )
        self.save()

    def pass_gate(self, gate_id: str) -> None:
        if gate_id not in GATE_IDS:
            raise StateError(f"unknown gate id {gate_id!r}; known: {GATE_IDS}")
        doc = self.doc
        if gate_id not in doc.gates_passed:
            doc.gates_passed.append(gate_id)
        self.save()

    # -------- R8 退避 ------------------------------------------------------

    def quarantine(self, step_id: str, source_paths: Iterable[str | Path]) -> str:
        """Copy crash artefacts into ``experiments/_wip/<step>-<ts>/`` and
        record the location in the failure history.
        """
        ts = int(time.time())
        dest = Path("experiments/_wip") / f"{step_id}-{ts}"
        dest.mkdir(parents=True, exist_ok=True)
        for sp in source_paths:
            sp = Path(sp)
            if sp.is_dir():
                shutil.copytree(sp, dest / sp.name, dirs_exist_ok=True)
            elif sp.is_file():
                shutil.copy2(sp, dest / sp.name)
        return str(dest)


__all__ = [
    "SCHEMA_VERSION",
    "STEP_IDS",
    "GATE_IDS",
    "StateError",
    "StepRecord",
    "FailureRecord",
    "StateDoc",
    "StateManager",
]

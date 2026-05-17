"""``afm oss`` startup invariants.

PIPELINE in ``afm/cli/oss.py`` and ``configs/mvp.yaml`` MUST agree on the
8 step IDs, names, and gate attachments. ``state.json --resume`` decides
which step to run next based on the IDs; drift = silent miss.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from afm.cli.oss import PIPELINE, _assert_pipeline_matches_config


def _load_cfg():
    p = Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_pipeline_matches_mvp_yaml() -> None:
    cfg = _load_cfg()
    _assert_pipeline_matches_config(cfg)


def test_pipeline_has_eight_steps() -> None:
    assert len(PIPELINE) == 8
    assert [sid for sid, *_ in PIPELINE] == ["0", "1", "2", "3", "4", "5", "5.5", "6"]


def test_assertion_raises_on_drift() -> None:
    bad_cfg = {"pipeline": {"steps": [{"id": "0", "name": "wrong", "gate": None}]}}
    with pytest.raises(RuntimeError, match="drift detected"):
        _assert_pipeline_matches_config(bad_cfg)

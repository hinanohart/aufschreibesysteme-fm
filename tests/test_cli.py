"""CLI smoke — typer parses our subcommands and gate tokens."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from afm.cli.oss import GATE_TOKEN_TO_ID, _parse_pause_on, app


def test_pause_on_parses_all_four_tokens() -> None:
    out = _parse_pause_on("licence-fail,collapse,space-deploy,preprint")
    assert out == {"G1", "G2", "G3", "G4"}


def test_pause_on_rejects_unknown_token() -> None:
    with pytest.raises(Exception):
        _parse_pause_on("not-a-gate")


def test_gate_tokens_match_doc() -> None:
    assert set(GATE_TOKEN_TO_ID) == {
        "licence-fail",
        "collapse",
        "space-deploy",
        "preprint",
    }


def test_help_runs() -> None:
    import re

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Phrasing contract — never *claim* fully automatic. Negation is fine.
    for m in re.finditer(r"fully[\s-]+automatic", result.output, re.IGNORECASE):
        prefix = result.output[max(0, m.start() - 60) : m.start()].lower()
        assert any(tok in prefix for tok in ("not", "no", "never")), (
            f"un-negated 'fully automatic' in --help output near: "
            f"{result.output[max(0, m.start() - 30) : m.end() + 30]!r}"
        )
    assert "semi-auto" in result.output.lower() or "resumable" in result.output.lower()


def test_info_command_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "0.2" in result.output


def test_oss_dry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    cfg_path = tmp_path / "configs/mvp.yaml"
    # Minimal valid config for dry-run only. Pipeline section MUST match the
    # hardcoded PIPELINE in afm.cli.oss — drift would otherwise short-circuit
    # a `--resume` run (see _assert_pipeline_matches_config).
    cfg_path.write_text(
        """\
state:
  path: state.json
  version: '0.2'
pipeline:
  steps:
    - {id: '0', name: env, gate: null}
    - {id: '1', name: data, gate: G1}
    - {id: '2', name: lora, gate: null}
    - {id: '3', name: gate, gate: G2}
    - {id: '4', name: eval, gate: null}
    - {id: '5', name: release, gate: null}
    - {id: '5.5', name: space, gate: G3}
    - {id: '6', name: arxiv, gate: G4}
  gates:
    G1: licence-fail
    G2: collapse
    G3: space-deploy
    G4: preprint
regimes:
  enabled: [parchment, typewriter, gramophone, photograph, film, jpeg, crt]
"""
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["oss", "--config", str(cfg_path), "--dry-run", "--pause-on", "licence-fail"],
    )
    assert result.exit_code == 0, result.output

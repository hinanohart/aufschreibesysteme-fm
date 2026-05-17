"""``afm oss`` — the resumable semi-auto orchestrator.

Single entry point::

    afm oss --resume \\
            --pause-on=licence-fail,collapse,space-deploy,preprint \\
            --config configs/mvp.yaml

Behaviour:
    - reads ``state.json`` to determine fresh / partial / done branch
    - executes steps 0 → 6 (with 5.5 in between)
    - honours the four pause gates declared via ``--pause-on``
    - on success: ``state.mark_done(step)`` and advance ``current_step``
    - on failure: ``state.mark_failed(step, …)`` + copy crash artefacts to
      ``experiments/_wip/<step>-<ts>/`` (R8) + exit with non-zero

The orchestrator deliberately stays small — each step is one function. Heavy
work is delegated to ``afm/core`` / ``afm/data`` / ``afm/eval``.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from afm.regimes import assert_registry_complete, regime_names
from afm.state import SCHEMA_VERSION, StateManager

# CLI surface --------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help=(
        "Aufschreibesysteme Flow Matching — lossy codecs as first-class "
        "generative regimes. Resumable semi-auto pipeline with 4 explicit "
        "human-review gates (NOT fully automatic)."
    ),
)
console = Console()


GATE_TOKEN_TO_ID = {
    "licence-fail": "G1",
    "collapse": "G2",
    "space-deploy": "G3",
    "preprint": "G4",
}
ID_TO_GATE_TOKEN = {v: k for k, v in GATE_TOKEN_TO_ID.items()}


# --- shared state -----------------------------------------------------------


@dataclass
class RunContext:
    config: dict[str, Any]
    state: StateManager
    pause_on: set[str]  # gate IDs that pause


def _parse_pause_on(raw: str | None) -> set[str]:
    if not raw:
        return set()
    out: set[str] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in GATE_TOKEN_TO_ID:
            raise typer.BadParameter(
                f"unknown gate token {tok!r}; valid: {sorted(GATE_TOKEN_TO_ID)}"
            )
        out.add(GATE_TOKEN_TO_ID[tok])
    return out


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise typer.BadParameter(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- step body imports (lazy) ----------------------------------------------


def _step_env(ctx: RunContext) -> dict[str, Any]:
    """Step 0 — bootstrap environment.

    Checks the repo is in a clean enough shape to run, asserts the regime
    registry is complete, and warns if any physical-noise prior file is a
    synthetic placeholder.
    """
    assert_registry_complete()
    out: dict[str, Any] = {"registry_complete": True}
    from afm.core.physical_noise_prior import PhysicalNoisePrior

    prior = PhysicalNoisePrior()
    synthetic = prior.warn_if_synthetic()
    out["synthetic_priors"] = synthetic
    fail_on_synth = bool(ctx.config.get("physical_prior", {}).get("fail_on_synthetic", True))
    if synthetic and fail_on_synth:
        raise RuntimeError(
            "synthetic placeholder PSDs in use for: "
            f"{synthetic}. The measurement files under data/measurement/ are "
            "missing. Either run `python scripts/fetch_measurements.py` first "
            "or set `physical_prior.fail_on_synthetic: false` in the config "
            "to proceed (discouraged — yields wrong physics)."
        )
    if synthetic:
        rprint(
            f"[yellow]warning[/yellow]: synthetic placeholder PSDs in use for: "
            f"{synthetic}. `physical_prior.fail_on_synthetic` is set to false; "
            f"the prior physics is NOT measurement-derived."
        )
    return out


def _step_data(ctx: RunContext) -> dict[str, Any]:
    """Step 1 — fetch + audit data manifests. Pauses on G1."""
    from afm.data.loaders import audit_manifests

    report = audit_manifests(ctx.config.get("data", {}))
    return report


def _step_lora(ctx: RunContext) -> dict[str, Any]:
    """Step 2 — train one LoRA per regime, sequential (24 GB budget)."""
    from afm.cli.train import train_all_regimes

    return train_all_regimes(ctx.config, state=ctx.state)


def _step_gate(ctx: RunContext) -> dict[str, Any]:
    """Step 3 — train the regime router on top of frozen LoRAs. Pauses on G2."""
    from afm.cli.train import train_router

    return train_router(ctx.config, state=ctx.state)


def _step_eval(ctx: RunContext) -> dict[str, Any]:
    """Step 4 — M1-M4 on held-out splits."""
    from afm.eval.runner import run_eval

    return run_eval(ctx.config)


def _step_release(ctx: RunContext) -> dict[str, Any]:
    """Step 5 — HF Hub model + GH tag. Requires `huggingface-cli login`."""
    from afm.cli.space_deploy import release_to_hub

    return release_to_hub(ctx.config)


def _step_space(ctx: RunContext) -> dict[str, Any]:
    """Step 5.5 — HF Space + teaser generation. Pauses on G3."""
    from afm.cli.space_deploy import build_space, build_teaser

    teaser_path = build_teaser(ctx.config)
    space_info = build_space(ctx.config)
    return {"teaser": teaser_path, "space": space_info}


def _step_arxiv(ctx: RunContext) -> dict[str, Any]:
    """Step 6 — preprint draft. Pauses on G4 (optional gate).

    MVP scope: only the LaTeX scaffold existence is verified. The actual
    preprint draft is written by hand, so this step is flagged
    ``placeholder=True`` (same pattern as ``train_router``) so downstream G4
    reviewers know the gate is informational.
    """
    paper = Path(ctx.config.get("paper", {}).get("tex_root", "paper/main.tex"))
    exists = paper.is_file()
    print(
        "WARNING: step 6 (arxiv) is a placeholder — it verifies the .tex "
        "scaffold exists but does not draft the preprint. G4 review is on the "
        "human author.",
        file=sys.stderr,
    )
    return {"tex_root": str(paper), "exists": exists, "placeholder": True}


# --- pipeline order ---------------------------------------------------------


PIPELINE = [
    ("0", "env", _step_env, None),
    ("1", "data", _step_data, "G1"),
    ("2", "lora", _step_lora, None),
    ("3", "gate", _step_gate, "G2"),
    ("4", "eval", _step_eval, None),
    ("5", "release", _step_release, None),
    ("5.5", "space", _step_space, "G3"),
    ("6", "arxiv", _step_arxiv, "G4"),
]


def _assert_pipeline_matches_config(cfg: dict[str, Any]) -> None:
    """Fail-fast if hardcoded PIPELINE drifts from configs/mvp.yaml.

    state.json resume depends on the two staying in lockstep — if anyone
    edits one without the other, the user can silently end up running the
    wrong step on resume. Cheap to check at startup.
    """
    cfg_steps = cfg.get("pipeline", {}).get("steps", [])
    cfg_pairs = [(s["id"], s["name"], s.get("gate")) for s in cfg_steps]
    code_pairs = [(sid, name, gate) for sid, name, _body, gate in PIPELINE]
    if cfg_pairs != code_pairs:
        raise RuntimeError(
            "PIPELINE / configs/mvp.yaml drift detected:\n"
            f"  code: {code_pairs}\n  cfg : {cfg_pairs}"
        )


# --- main orchestrator ------------------------------------------------------


@app.command("oss")
def oss(
    config: Path = typer.Option(
        Path("configs/mvp.yaml"),
        "--config",
        "-c",
        help="MVP YAML config",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from state.json if present; if absent, fresh-start.",
    ),
    pause_on: str = typer.Option(
        "licence-fail,collapse,space-deploy,preprint",
        "--pause-on",
        help="Comma-separated gates to pause at. Default: all four.",
    ),
    eval_only: bool = typer.Option(
        False,
        "--eval-only",
        help="Run eval (step 4) only — useful when current_step == 6.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan, don't execute."),
) -> None:
    """Run the resumable semi-auto pipeline.

    The full pipeline has 8 steps with 4 explicit human-review gates
    (G1 licence-fail / G2 collapse / G3 space-deploy / G4 preprint).
    `afm oss --resume` reads `state.json` to decide which step to start
    from. Hitting a pause gate writes a clean state and exits non-zero —
    re-run with `--resume` after handling the gate.
    """
    cfg = _load_config(config)
    _assert_pipeline_matches_config(cfg)
    pause_set = _parse_pause_on(pause_on)
    state = StateManager(cfg.get("state", {}).get("path", "state.json"))

    branch = state.branch()
    rprint(
        Panel.fit(
            f"[bold]afm oss[/bold] — schema {SCHEMA_VERSION} — branch: [cyan]{branch}[/cyan] — "
            f"pause-on: {sorted(pause_set) or '∅'}",
            title="Aufschreibesysteme Flow Matching",
        )
    )

    if eval_only:
        ctx = RunContext(config=cfg, state=state, pause_on=pause_set)
        rprint("[bold]eval-only mode[/bold] — running step 4 only")
        if not dry_run:
            _run_step(ctx, *next(s for s in PIPELINE if s[0] == "4"))
        return

    # Build run plan.
    start_idx = 0
    if branch == "partial":
        # First not-done step
        for i, (sid, _name, _body, _gate) in enumerate(PIPELINE):
            rec = state.doc.steps.get(sid)
            if rec is None or rec.status != "done":
                start_idx = i
                break

    plan = PIPELINE[start_idx:]
    table = Table(title="Plan")
    table.add_column("step")
    table.add_column("name")
    table.add_column("gate", style="magenta")
    table.add_column("status")
    for sid, name, _body, gate in plan:
        rec = state.doc.steps.get(sid)
        table.add_row(sid, name, gate or "-", rec.status if rec else "pending")
    console.print(table)

    if dry_run:
        return

    ctx = RunContext(config=cfg, state=state, pause_on=pause_set)
    for sid, name, body, gate in plan:
        keep_going = _run_step(ctx, sid, name, body, gate)
        if not keep_going:
            sys.exit(2)  # paused at a gate — re-run with --resume


def _run_step(
    ctx: RunContext,
    sid: str,
    name: str,
    body: Callable[[RunContext], dict[str, Any]],
    gate: str | None,
) -> bool:
    """Run one step. Returns False if the run should stop (pause/fail)."""
    ctx.state.mark_running(sid, name=name)
    rprint(f"[bold]→ step {sid}[/bold] [cyan]{name}[/cyan]" + (f" (gate {gate})" if gate else ""))
    try:
        extra = body(ctx)
    except Exception as e:
        wip = ctx.state.quarantine(sid, source_paths=[])
        rprint(f"[red]✗ step {sid} {name} failed:[/red] {e}")
        rprint(f"[red]  see {wip}[/red]")
        traceback.print_exc()
        ctx.state.mark_failed(sid, error=str(e), wip_dir=wip)
        return False
    ctx.state.mark_done(sid, **extra)
    rprint(f"[green]✓ step {sid} {name} done[/green]")

    if gate is not None and gate in ctx.pause_on:
        token = ID_TO_GATE_TOKEN[gate]
        rprint(
            Panel.fit(
                f"[bold yellow]paused at gate {gate} ({token})[/bold yellow]\n"
                f"Review the step output, then re-run with `--resume` to continue.\n"
                f"To bypass, drop {token!r} from --pause-on.",
                title=f"gate {gate}",
            )
        )
        return False

    if gate is not None:
        # Gate exists but not in pause-on → mark as auto-passed.
        ctx.state.pass_gate(gate)

    return True


@app.command("info")
def info() -> None:
    """Print regime registry and state.json branch."""
    rprint(f"schema version: [cyan]{SCHEMA_VERSION}[/cyan]")
    rprint(f"regimes: {regime_names()}")
    state = StateManager()
    rprint(f"branch: [cyan]{state.branch()}[/cyan]")


# Subcommands are attached lazily so `afm --help` stays fast.


def _attach_subcommands() -> None:
    from afm.cli import demo as _d
    from afm.cli import infer as _i
    from afm.cli import morph as _m
    from afm.cli import space_deploy as _s
    from afm.cli import train as _t

    app.command("train")(_t.train_command)
    app.command("infer")(_i.infer_command)
    app.command("morph")(_m.morph_command)
    app.command("space-deploy")(_s.space_deploy_command)
    app.command("demo")(_d.demo_command)


_attach_subcommands()


if __name__ == "__main__":
    app()

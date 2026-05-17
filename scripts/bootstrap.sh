#!/usr/bin/env bash
# scripts/bootstrap.sh — one-shot user-side bootstrap for `afm oss`.
#
# All steps that REQUIRE a human (secrets, interactive auth, long GPU
# training) live here. Re-running is safe: every step skips if its
# precondition is already satisfied.
#
# Usage:
#   bash scripts/bootstrap.sh                       # full path
#   bash scripts/bootstrap.sh --no-gpu              # stop before training
#   bash scripts/bootstrap.sh --resume              # skip install/login, jump to oss
#
# Environment overrides (optional):
#   AFM_CONFIG=configs/mvp.yaml                     # alternate config
#   AFM_PAUSE_ON=licence-fail,collapse,space-deploy,preprint
#   AFM_SKIP_FETCH=1                                # measurement fetch already done
#
# DELIBERATELY DOES NOT:
#   - accept tokens via stdin/env in clear (we delegate to `huggingface-cli login`)
#   - run `rm -rf` anywhere
#   - bypass the four pause gates
#   - claim "fully automatic" — this script orchestrates a SEMI-auto pipeline

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${AFM_CONFIG:-configs/mvp.yaml}"
PAUSE_ON="${AFM_PAUSE_ON:-licence-fail,collapse,space-deploy,preprint}"
NO_GPU=0
RESUME_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --no-gpu)     NO_GPU=1 ;;
    --resume)     RESUME_ONLY=1 ;;
    --help|-h)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# step 0: Python + pip preflight
# ---------------------------------------------------------------------------
say "0/6  python preflight"
command -v python3 >/dev/null || die "python3 not found in PATH"
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
  die "python >= 3.10 required (got $PY_MAJOR.$PY_MINOR)"
fi
ok "python $PY_MAJOR.$PY_MINOR"

if [[ "$RESUME_ONLY" -eq 0 ]]; then
  # -------------------------------------------------------------------------
  # step 1: install (editable + dev extras)
  # -------------------------------------------------------------------------
  say "1/6  install afm (editable, with dev extras)"
  if python3 -c "import afm" 2>/dev/null && \
     python3 -c "import peft, diffusers, transformers" 2>/dev/null; then
    ok "already installed"
  else
    python3 -m pip install --upgrade pip wheel >/dev/null
    python3 -m pip install -e ".[dev]"
    ok "installed"
  fi

  # -------------------------------------------------------------------------
  # step 2: HF auth (delegated to huggingface_hub — script never sees token)
  # -------------------------------------------------------------------------
  say "2/6  huggingface auth"
  if python3 - <<'PY'
import sys
try:
    from huggingface_hub import HfApi
    HfApi().whoami()
except Exception:
    sys.exit(1)
PY
  then
    ok "huggingface-cli already authenticated"
  else
    warn "not authenticated — running interactive login (token will NOT be logged)"
    huggingface-cli login
  fi
fi

# ---------------------------------------------------------------------------
# step 3: fetch measurements (idempotent — SHA256 verifies)
# ---------------------------------------------------------------------------
say "3/6  fetch physical-noise measurements"
if [[ "${AFM_SKIP_FETCH:-0}" == "1" ]]; then
  warn "AFM_SKIP_FETCH=1 — skipping"
elif [[ -f data/measurement/SHA256SUMS ]] && \
     python3 -c "import pathlib, sys; sys.exit(0 if len(list(pathlib.Path('data/measurement').glob('*.npz'))) >= 7 else 1)"; then
  ok "measurements already present (>=7 .npz files)"
else
  # Fetcher exits 0 even when no source manifests are present (it just
  # logs and returns skipped=True). Re-check on the filesystem so we
  # don't display a false "fetched" success.
  python3 scripts/fetch_measurements.py --regime all || true
  npz_count=$(find data/measurement -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "$npz_count" -ge 7 ]]; then
    ok "fetched ($npz_count .npz files)"
  else
    warn "fetcher ran but only $npz_count/7 .npz file(s) present — pipeline step 0 will fail with synthetic-prior error unless data/manifests/<regime>_sources.csv is provided. Continuing so the smoke tests still run."
  fi
fi

# ---------------------------------------------------------------------------
# step 4: audit audio licence whitelist (HARD-FAIL if manifest present)
# ---------------------------------------------------------------------------
# The audit is gate G1 inside `afm oss`, so a missing manifest here is not
# fatal — pipeline step 1 will re-run it once data is fetched. But if the
# manifest IS present, any neighbouring-rights violation must hard-fail now.
say "4/6  audit audio whitelist (EU neighbouring rights 70-year exclusion)"
if [[ -f data/manifests/gramophone.csv ]]; then
  python3 scripts/audit_audio.py
  ok "audio whitelist clean"
else
  warn "data/manifests/gramophone.csv not yet present — audit deferred to pipeline step 1 (G1)"
fi

# ---------------------------------------------------------------------------
# step 5: smoke pytest (lightweight, no GPU)
# ---------------------------------------------------------------------------
say "5/6  pytest smoke (no GPU)"
python3 -m pytest -q tests/ -x
ok "tests green"

if [[ "$NO_GPU" -eq 1 ]]; then
  say "stopping before GPU training (--no-gpu); next step would be:"
  echo "    afm oss --resume --config $CONFIG --pause-on=$PAUSE_ON"
  exit 0
fi

# ---------------------------------------------------------------------------
# step 6: run the resumable semi-auto pipeline
#   - hits 4 explicit human-review gates
#   - re-running with --resume after a gate is the intended workflow
# ---------------------------------------------------------------------------
say "6/6  afm oss --resume (4 gates: G1 licence-fail / G2 collapse / G3 space-deploy / G4 preprint)"
exec afm oss --resume --config "$CONFIG" --pause-on="$PAUSE_ON"

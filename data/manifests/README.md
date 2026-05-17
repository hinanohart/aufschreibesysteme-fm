# Data manifests

Per-regime CSV manifests. Each row carries (`path`, `license`, `sha256`,
`source`, `whitelist_approved`, …). The audit step (G1) walks these and
hard-fails on anything not whitelisted.

- `audio_whitelist.csv` — per-item allow list for the gramophone regime.
  See `docs/architecture.md` C1 for why this exists. Internet Archive
  78rpm bulk import is **disallowed** — see EU neighbouring-rights
  (Leistungsschutzrechte, 70 years past performer's death).
- `captions_eval.txt` — shared eval caption set (M1-M4 all use the same
  prompts). Adding a caption here forces a re-run of every metric.
- `<regime>_sources.csv` — input to `scripts/fetch_measurements.py`.
- `<regime>.csv` — per-regime training/eval manifest, populated by the
  fetch script.

Run `python scripts/audit_audio.py` locally before pushing — CI runs the
same check at G1.

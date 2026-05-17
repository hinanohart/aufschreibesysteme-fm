# Contributing

A few non-obvious rules. Everything else is "do what makes sense."

## Commit messages

- Use plain conventional prefixes: `feat:`, `fix:`, `docs:`, `test:`,
  `chore:`, `refactor:`. Body should explain the *why*.
- **Do not** put R-numbers in commit messages (no `R14`, `R17`, etc.).
  Those reference an internal rule set, not user-facing context.

## Phrasing

- The pipeline is **resumable semi-auto** with 4 explicit human-review
  gates. Do not describe it as "fully automatic" or "permanent" in
  user-facing docs, READMEs, CLI help, or commits. There is a phrasing
  test in `tests/test_phrasing.py` that fails the build on `fully
  automatic`.

## License

- Anything you add must be Apache-2.0 compatible. GPL/AGPL deps fail CI
  via `pip-licenses --fail-on=GPL,AGPL`.
- Audio data must be either Wikimedia PD, LoC PD, or per-item
  whitelisted in `data/manifests/audio_whitelist.csv`. **Do not** add
  Internet Archive 78rpm bulk items — see `docs/architecture.md` C1.

## Tests

- `pytest tests/` runs the full no-GPU suite in under a minute.
- GPU-required tests are marked `@pytest.mark.gpu` — skipped by default.
- New regimes break the 7-regime invariant in `afm/regimes/base.py`
  *intentionally*. If you really need an eighth, that is a scope
  conversation, not a PR.

## Architecture

- The physical noise prior must stay frozen. Promoting it to
  `nn.Parameter` will trip `_assert_frozen()` and the test in
  `tests/test_prior.py`.
- Top-2 *soft* routing, not hard. The router emits full-softmax weights
  with the top-k mass renormalised.
- `save_steps=500` is the crash-bound contract. Lower it locally if you
  want; do not raise it in committed configs.

## Releasing

- Tag via `git tag v0.1.0 && git push --tags`.
- HF Hub release is automated through `afm oss step release` once
  `huggingface-cli login` has been done in your shell. Claude / agents
  must not read tokens directly — see R11 in our internal rule set.

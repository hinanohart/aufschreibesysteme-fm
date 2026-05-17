# Architecture

This is the spec a contributor would need to extend `afm` without breaking
the invariants. It is *not* the paper — the technical pitch lives in
`docs/signal_materiality.md`, and the Kittler motivation in
`docs/appendix_kittler.md`.

---

## Components

```
┌────────────────────────────┐
│ AufschreibePipeline        │     subclass of diffusers.DiffusionPipeline
│ ├─ base DiT (frozen)       │     SANA-1.6B default, Flux-schnell showcase
│ ├─ ExpertLoRAManager       │     7 PEFT adapters, rank=32
│ ├─ RegimeRouter            │     2-layer MLP, top-2 soft, L_lb=λ N Σ f_r P_r
│ ├─ MuLANSigmaHead          │     per-pixel σ, clipped from below
│ ├─ PhysicalNoisePrior      │     frozen register_buffer PSDs
│ └─ FlowMatchingScheduler   │     rectified FM, +Diffusion Forcing for film
└────────────────────────────┘
```

| Module                              | Lines (approx) | Owns                                                   |
| ----------------------------------- | -------------- | ------------------------------------------------------ |
| `afm/core/physical_noise_prior.py`  | 170            | the *Real* channel (Lacan caveat – see appendix)       |
| `afm/core/expert_lora.py`           | 130            | the *Symbolic* — PEFT adapter management               |
| `afm/core/regime_router.py`         | 160            | gating + load-balance + collapse monitor               |
| `afm/core/mulan_sigma.py`           | 80             | per-pixel σ head with the prior floor                  |
| `afm/core/schedulers.py`            | 90             | rectified FM + Diffusion Forcing                       |
| `afm/core/trainer.py`               | 210            | single-regime loop, AdamW8bit, bf16, save_steps=500    |
| `afm/core/pipeline.py`              | 180            | composed Diffusers pipeline                            |
| `afm/regimes/<7 files>`             | 50 each        | declarative codec + prior triple                       |
| `afm/state/manager.py`              | 230            | v0.2 schema, atomic IO, 3-branch resume, R8 quarantine |
| `afm/cli/oss.py`                    | 300            | step loop, gates, --pause-on, --resume                 |

---

## Invariants (CI-enforced via `tests/`)

1. **`N_REGIMES == 7`.** Adding regimes silently breaks the classifier head
   and the load-balance normalisation; the registry asserts at boot.
2. **Physical prior is frozen.** No `nn.Parameter` over PSD buffers, no
   `requires_grad=True` toggled on. The guard in `_assert_frozen` runs at
   training-loop start.
3. **Top-2 soft, never hard.** `RegimeRouter.forward` returns full-softmax
   weights with the top-k mass renormalised. There is no `one_hot` on the
   path.
4. **`save_steps == 500`** by default — mid-step crash bound ≤ 30 minutes
   at ~1 s/step.
5. **state.json schema is "0.2".** Changing the schema bumps the constant
   in `afm/state/manager.py` and gates a migration.
6. **License audit hard-fails at G1.** `afm.data.loaders.audit_manifests`
   rejects anything sourced from `internet_archive_78rpm_bulk` and any
   audio item not on the whitelist.

---

## The 4 gates

| Gate | Step | Owner          | Pause token (`--pause-on`) |
| ---- | ---- | -------------- | -------------------------- |
| G1   | 1    | license audit  | `licence-fail`             |
| G2   | 3    | collapse mon.  | `collapse`                 |
| G3   | 5.5  | HF Space push  | `space-deploy`             |
| G4   | 6    | preprint draft | `preprint`                 |

A gate is "auto-passed" only when the user removes its token from
`--pause-on`. Otherwise the orchestrator writes a clean state and exits
with code 2. Re-run `afm oss --resume` after handling the gate.

---

## Extensions in scope (post-MVP)

- **MoE expert collapse rescue beyond λ-doubling.** Currently if collapse
  retriggers after `max_doublings=6` we just stop doubling. A targeted
  warm-restart of the offending adapter is on the roadmap.
- **8-bit teaser-time inference** via `bitsandbytes` 4-bit for the Space.
- **Full DSBM IPF** in `afm/morph/dsbm_full.py` (currently `ipf_iters=1`).

---

## Extensions out of scope

- **Conference deadline-driven scoping.** Paper is a rolling preprint;
  changes that exist only to chase a Jan/May/Sep window are out.
- **A new regime "to round out a figure".** Adding regimes affects every
  metric. PR an ablation, not a figure.
- **Replacing SANA with Flux-dev.** Flux-dev is non-commercial — we keep
  the default Apache-2.0.

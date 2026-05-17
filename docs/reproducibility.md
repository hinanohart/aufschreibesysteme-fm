# Reproducibility

Single-GPU (24 GB) end-to-end MVP takes ~66 hours wall clock. Two
people, two GPUs, two seeds (1729 and 17) is the recommended
minimal-evidence configuration.

---

## Reproducing the MVP run

```bash
git clone https://github.com/hinanohart/aufschreibesysteme-fm
cd aufschreibesysteme-fm
uv venv && source .venv/bin/activate
uv pip install -e .[dev]                                # ~3 min
python scripts/fetch_measurements.py --regime all       # ~30 min, rate-limited
python scripts/audit_audio.py                           # exits 0 on clean run
afm oss --resume --config configs/mvp.yaml \
        --pause-on=licence-fail,collapse,space-deploy,preprint
```

Wall clock breakdown (RTX 4090 reference):

| Step            | Time   | Notes                                              |
| --------------- | ------ | -------------------------------------------------- |
| 0 env           | 1 min  | regime registry + prior sanity                     |
| 1 data          | 30 min | mostly the fetch; **pauses at G1**                 |
| 2 lora (×7)     | ~60 h  | 30 k step / regime, sequential                     |
| 3 gate          | 3 h    | router fit on frozen LoRAs; **may pause at G2**    |
| 4 eval          | 2 h    | M1–M4 against B1–B4                                |
| 5 release       | 5 min  | HF Hub upload                                      |
| 5.5 space       | 1 h    | teaser render + Space push; **pauses at G3**       |
| 6 arxiv         | 1 h    | tex compile; **optional pause at G4**              |

Save points at every 500 train steps (~30 min) means a mid-step crash
loses at most half an hour per regime.

---

## Hardware

- 24 GB VRAM is the floor (SANA-1.6B frozen + r=32 LoRA + bf16 + grad-ckpt
  fits in 18–20 GB).
- `batch=2 grad_accum=8` is the training contract.
- `batch=8` is permitted in **inference/eval** only (not training).

---

## Determinism

- `train.seed = 1729` and `eval.eval_seed = 17` are the canonical seeds.
- The text encoder is frozen across baselines.
- Caption set is `data/manifests/captions_eval.txt`.
- Adding a caption invalidates every per-regime FID / classifier number.

---

## Known sources of non-determinism

- `bitsandbytes` AdamW8bit is not bit-deterministic across CUDA versions.
- `torch.use_deterministic_algorithms(True)` is not currently enabled at
  training time — flipping it cuts step rate by ~25 % and is reserved for
  the deterministic-comparison appendix.
- HF datasets download URLs can drift; the SHA256SUMS files in
  `data/datasets/<regime>/` are the ground truth.

---

## A regime fails — what now?

If the per-regime training loop terminates with a non-zero exit:

1. `afm oss` writes `state.failures[]` with the wip dir path.
2. The wip dir under `experiments/_wip/<step>-<ts>/` contains the most
   recent checkpoint and a copy of the running config.
3. Re-run `afm oss --resume` — the orchestrator picks up at the failed
   step and skips already-done regimes.

We deliberately do not auto-retry. Silent retries hide real bugs.

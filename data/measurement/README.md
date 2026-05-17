# data/measurement/

Place measurement-derived PSD/kernel .npz files here, one per regime:
- parchment.npz
- typewriter.npz
- gramophone.npz
- photograph.npz
- film.npz
- jpeg.npz
- crt.npz

If a file is absent, `afm/core/physical_noise_prior.py` falls back to a
synthetic 1/f placeholder so the pipeline boots. The CLI prints a yellow
warning when synthetic priors are in use — you should never publish a
paper or HF model trained against the synthetic fallback.

Acquire real measurements via `scripts/fetch_measurements.py` (which
populates `data/datasets/<regime>/`) then fit each regime's PSD to a
held-out subset and save the resulting tensor as `<regime>.npz` with the
key `psd`.

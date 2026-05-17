# `examples/` quickstart

Five examples, in the order you should read them.

1. **`infer_cli.py`** — load pre-trained weights, sample one image, save
   it to disk. The 30-second "did it work" test.
2. **`train_sana.ipynb`** — fine-tune one regime LoRA from scratch on a
   24 GB GPU. Skeleton notebook, requires `data/datasets/<regime>/` to
   be populated by `scripts/fetch_measurements.py`.
3. **`morph_demo.ipynb`** — DSBM-driven cross-regime morph. Inference-time
   only, runs on the same 24 GB.
4. **`run_eval.ipynb`** *(post-MVP)* — M1–M4 against the four baselines.
5. **`space_local.ipynb`** *(post-MVP)* — run the HF Space gradio app
   locally, useful for iterating on the teaser image.

If you only have CPU, only `infer_cli.py` will run end-to-end with the
synthetic priors (slow). Everything else needs the GPU.

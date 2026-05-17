"""HF Space app — Gradio interface for ``afm`` demos.

Provides:
    1. single-regime inference
    2. cross-regime morph (DSBM)
    3. the 7-regime teaser gallery for a single prompt

This file is uploaded as-is by ``afm space-deploy``. Keep it self-contained
so HF Space autoresolves dependencies via ``space/requirements.txt``.
"""

from __future__ import annotations

import os

import gradio as gr

REGIMES = ["parchment", "typewriter", "gramophone", "photograph", "film", "jpeg", "crt"]
PRETRAINED = os.environ.get("AFM_HUB_REPO", "hinanohart/aufschreibesysteme-fm")
EXAMPLE_PROMPTS = [
    "a lighthouse at the edge of a stormy sea",
    "a deserted train station at dusk",
    "a wheat field at noon",
]


def _load_pipeline():
    from afm.core.pipeline import AufschreibePipeline

    return AufschreibePipeline.from_pretrained(PRETRAINED)


def infer_one(prompt: str, regime: str, steps: int, seed: int):
    from afm.core.pipeline import InferenceConfig

    pipe = _load_pipeline()
    x = pipe(
        prompt=prompt, regime=regime, config=InferenceConfig(num_inference_steps=steps, seed=seed)
    )
    return _to_pil(x)


def morph_two(prompt: str, a: str, b: str, steps: int):
    pipe = _load_pipeline()
    frames = pipe.morph(prompt=prompt, regime_a=a, regime_b=b, steps=steps)
    return [_to_pil(f) for f in frames]


def gallery_seven(prompt: str, seed: int):
    from afm.core.pipeline import InferenceConfig

    pipe = _load_pipeline()
    return [
        _to_pil(pipe(prompt=prompt, regime=r, config=InferenceConfig(seed=seed))) for r in REGIMES
    ]


def _to_pil(x):
    from PIL import Image

    x = x.detach().cpu().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


with gr.Blocks(title="Aufschreibesysteme Flow Matching") as demo:
    gr.Markdown(
        "# Aufschreibesysteme Flow Matching\n\n"
        "Lossy codecs as first-class generative regimes. "
        "This is a research preview — see "
        "[the repo](https://github.com/hinanohart/aufschreibesysteme-fm) for details."
    )

    with gr.Tab("Single regime"):
        prompt = gr.Textbox(label="prompt", value=EXAMPLE_PROMPTS[0])
        regime = gr.Dropdown(REGIMES, value="jpeg", label="regime")
        steps = gr.Slider(8, 100, value=50, step=1, label="inference steps")
        seed = gr.Number(value=17, label="seed", precision=0)
        run_btn = gr.Button("generate")
        out = gr.Image(label="output", height=512)
        run_btn.click(infer_one, [prompt, regime, steps, seed], out)

    with gr.Tab("Morph between regimes"):
        prompt_m = gr.Textbox(label="prompt", value=EXAMPLE_PROMPTS[1])
        a = gr.Dropdown(REGIMES, value="photograph", label="from")
        b = gr.Dropdown(REGIMES, value="gramophone", label="to")
        steps_m = gr.Slider(8, 100, value=50, step=1, label="steps")
        morph_btn = gr.Button("morph")
        gallery_m = gr.Gallery(label="morph frames", columns=8, height=300)
        morph_btn.click(morph_two, [prompt_m, a, b, steps_m], gallery_m)

    with gr.Tab("Teaser: 7-regime gallery"):
        prompt_g = gr.Textbox(label="prompt", value=EXAMPLE_PROMPTS[2])
        seed_g = gr.Number(value=17, label="seed", precision=0)
        gal_btn = gr.Button("render gallery")
        gallery_g = gr.Gallery(label="7 regimes", columns=7, height=240)
        gal_btn.click(gallery_seven, [prompt_g, seed_g], gallery_g)


if __name__ == "__main__":
    demo.launch()

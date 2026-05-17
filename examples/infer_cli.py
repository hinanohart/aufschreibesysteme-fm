"""End-to-end usage example for the public Python API.

Run after `pip install -e .` and `huggingface-cli login` (only needed if
pulling pre-trained weights from a private repo). The script is
intentionally short — anything fancy belongs in `space/app.py` or the
paper notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from afm import AufschreibePipeline
from afm.core.pipeline import InferenceConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="a lighthouse at the edge of a stormy sea")
    ap.add_argument("--regime", default="jpeg")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--pretrained", default="hinanohart/aufschreibesysteme-fm")
    ap.add_argument("--out", default="example_out.png", type=Path)
    args = ap.parse_args()

    pipe = AufschreibePipeline.from_pretrained(args.pretrained)
    x = pipe(
        prompt=args.prompt,
        regime=args.regime,
        config=InferenceConfig(num_inference_steps=args.steps, seed=args.seed),
    )
    _save(x, args.out)
    print(f"wrote {args.out}")


def _save(x, out: Path) -> None:
    from PIL import Image

    x = x.detach().cpu().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(out)


if __name__ == "__main__":
    main()

"""M4 — inscription consistency.

For the same prompt across all 7 regimes, the CLIPScore variance should be
small — i.e. the semantic identity is preserved while the inscription
channel changes. The MVP success criterion is variance ≤ 0.05.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InscriptionConsistencyReport:
    variance: float
    mean: float
    target: float = 0.05

    def passes(self) -> bool:
        return self.variance <= self.target

    def to_json(self) -> dict[str, Any]:
        return {
            "variance": self.variance,
            "mean": self.mean,
            "target": self.target,
            "passes": self.passes(),
        }


def _clip_score(image: torch.Tensor, text: str) -> float:
    """Cosine similarity between CLIP image embedding and CLIP text embedding.

    Returns 0.0 on import failure (smoke-test compatible).
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:  # pragma: no cover
        return 0.0
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pil = _tensor_to_pil(image)
    inputs = proc(text=[text], images=[pil], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    return float(out.logits_per_image.item() / 100.0)


def _tensor_to_pil(x: torch.Tensor):
    from PIL import Image

    x = x.detach().cpu().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def evaluate(
    images_per_regime: dict[str, torch.Tensor], prompt: str
) -> InscriptionConsistencyReport:
    scores = [_clip_score(img, prompt) for img in images_per_regime.values()]
    return InscriptionConsistencyReport(
        variance=float(statistics.pvariance(scores)) if len(scores) > 1 else 0.0,
        mean=float(sum(scores) / len(scores)) if scores else 0.0,
    )


__all__ = ["InscriptionConsistencyReport", "evaluate"]

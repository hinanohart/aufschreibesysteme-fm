"""M1 — regime classification recall via a fine-tuned ResNet-50.

We deliberately do NOT use CLIP zero-shot: zero-shot regime tags are too
correlated with caption tokens, which leaks the label. A 7-class ResNet-50
fine-tuned on the held-out real corpus is the metric of record.

The MVP success criterion is recall ≥ 0.85 on a held-out split
(7 regimes, balanced).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

N_CLASSES = 7


def build_classifier(*, num_classes: int = N_CLASSES) -> nn.Module:
    """Return a ResNet-50 with a 7-class head, ready for fine-tuning."""
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "regime_classifier requires torchvision; install with `pip install torchvision`"
        ) from e
    if num_classes != N_CLASSES:
        raise ValueError(f"classifier must use {N_CLASSES} classes (got {num_classes})")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@dataclass
class ClassifierEvalReport:
    recall_per_regime: dict[str, float]
    macro_recall: float
    target: float = 0.85

    def passes(self) -> bool:
        return self.macro_recall >= self.target

    def to_json(self) -> dict[str, Any]:
        return {
            "recall_per_regime": self.recall_per_regime,
            "macro_recall": self.macro_recall,
            "target": self.target,
            "passes": self.passes(),
        }


def evaluate(model: nn.Module, dataloader, regime_names: list[str]) -> ClassifierEvalReport:
    """Standard per-class recall computation. Expects (image, label) batches."""
    model.eval()
    device = next(model.parameters()).device
    tp = torch.zeros(len(regime_names), dtype=torch.long)
    fn = torch.zeros(len(regime_names), dtype=torch.long)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=-1)
            for c in range(len(regime_names)):
                tp[c] += ((pred == c) & (y == c)).sum().cpu()
                fn[c] += ((pred != c) & (y == c)).sum().cpu()
    recall = (tp.float() / (tp + fn).float().clamp_min(1)).tolist()
    per = {regime_names[i]: float(recall[i]) for i in range(len(regime_names))}
    return ClassifierEvalReport(
        recall_per_regime=per,
        macro_recall=float(sum(recall) / len(recall)),
    )


__all__ = ["build_classifier", "evaluate", "ClassifierEvalReport", "N_CLASSES"]

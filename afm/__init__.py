"""Aufschreibesysteme Flow Matching — public API.

Importing ``afm`` does NOT pull torch/diffusers eagerly so ``afm --help``
stays fast. Heavyweight symbols are exposed lazily via ``__getattr__``.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):  # noqa: N807
    # Static dispatch — no dynamic importlib.import_module path. Keeps
    # ``afm --help`` snappy while not handing user input to ``import_module``.
    if name == "AufschreibePipeline":
        from afm.core.pipeline import AufschreibePipeline

        return AufschreibePipeline
    if name == "RegimeSpec":
        from afm.regimes.base import RegimeSpec

        return RegimeSpec
    if name == "PhysicalNoisePrior":
        from afm.core.physical_noise_prior import PhysicalNoisePrior

        return PhysicalNoisePrior
    if name == "RegimeRouter":
        from afm.core.regime_router import RegimeRouter

        return RegimeRouter
    if name == "ExpertLoRAManager":
        from afm.core.expert_lora import ExpertLoRAManager

        return ExpertLoRAManager
    raise AttributeError(f"module 'afm' has no attribute {name!r}")


__all__ = [
    "AufschreibePipeline",
    "RegimeSpec",
    "PhysicalNoisePrior",
    "RegimeRouter",
    "ExpertLoRAManager",
    "__version__",
]

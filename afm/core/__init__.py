"""Core modules: trainer, physical prior, router, LoRA, MuLAN σ, schedulers.

Imports are deferred via ``__getattr__`` so that pulling in a peft-free
symbol (e.g. ``FlowMatchingScheduler``) does not force the user to install
peft. The eager-import variant was bundling expert_lora — and therefore peft
— into every ``import afm.core.*`` path.
"""

from __future__ import annotations

__all__ = [
    "ExpertLoRAManager",
    "MuLANSigmaHead",
    "PhysicalNoisePrior",
    "RegimeRouter",
    "RouterConfig",
    "FlowMatchingScheduler",
    "DiffusionForcingScheduler",
    "fm_velocity_loss",
    "cfm_loss",
]


def __getattr__(name: str):  # noqa: N807
    if name == "ExpertLoRAManager":
        from afm.core.expert_lora import ExpertLoRAManager

        return ExpertLoRAManager
    if name == "MuLANSigmaHead":
        from afm.core.mulan_sigma import MuLANSigmaHead

        return MuLANSigmaHead
    if name == "PhysicalNoisePrior":
        from afm.core.physical_noise_prior import PhysicalNoisePrior

        return PhysicalNoisePrior
    if name in {"RegimeRouter", "RouterConfig"}:
        from afm.core.regime_router import RegimeRouter, RouterConfig

        return {"RegimeRouter": RegimeRouter, "RouterConfig": RouterConfig}[name]
    if name in {"FlowMatchingScheduler", "DiffusionForcingScheduler"}:
        from afm.core.schedulers import DiffusionForcingScheduler, FlowMatchingScheduler

        return {
            "FlowMatchingScheduler": FlowMatchingScheduler,
            "DiffusionForcingScheduler": DiffusionForcingScheduler,
        }[name]
    if name in {"fm_velocity_loss", "cfm_loss"}:
        from afm.core.losses import cfm_loss, fm_velocity_loss

        return {"fm_velocity_loss": fm_velocity_loss, "cfm_loss": cfm_loss}[name]
    raise AttributeError(f"module 'afm.core' has no attribute {name!r}")

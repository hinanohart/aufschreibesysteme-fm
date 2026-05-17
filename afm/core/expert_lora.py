"""LoRA hook manager: per-regime rank-32 adapters on a frozen DiT base.

Architecture spec:
    - rank = 32 (alpha defaults to 32; ``alpha/rank == 1``)
    - target modules: cross-attn ``to_q``, ``to_k``, ``to_v``, ``to_out.0``
      and feed-forward ``ff.net.0.proj`` / ``ff.net.2`` of SANA linear-attn
      DiT blocks
    - one PEFT ``LoraConfig`` per regime; routing is composed at forward time
      by ``RegimeRouter`` which mixes top-2 soft expert outputs

We deliberately keep the hook layer thin and depend on PEFT so we get gradient
checkpointing / safetensors / fp8 export for free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as e:  # pragma: no cover -- documented dep
    raise ImportError("afm requires `peft>=0.11`; install with `pip install peft`") from e


LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


@dataclass
class ExpertSpec:
    regime: str
    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    def to_peft(self) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias="none",
            task_type=None,
            init_lora_weights="gaussian",
        )


class ExpertLoRAManager:
    """Owns one PEFT adapter per regime, attached to a single frozen base.

    Usage::

        mgr = ExpertLoRAManager(base_dit, regimes=["jpeg", "film", ...])
        mgr.set_active(["jpeg", "film"])          # top-2 routing per step
        out = base_dit(x, ...)                    # adapters apply additively

    The manager does NOT mix expert outputs — that is the job of
    ``RegimeRouter``. It only owns adapter parameters and toggles which are
    active.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        regimes: Iterable[str],
        rank: int = LORA_RANK,
        alpha: int = LORA_ALPHA,
        dropout: float = LORA_DROPOUT,
        target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    ) -> None:
        self.regimes: list[str] = list(regimes)
        if not self.regimes:
            raise ValueError("at least one regime is required")
        # Freeze base.
        for p in base.parameters():
            p.requires_grad_(False)
        peft_model: PeftModel | None = None
        for regime in self.regimes:
            cfg = ExpertSpec(
                regime=regime,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            ).to_peft()
            if peft_model is None:
                peft_model = get_peft_model(base, cfg, adapter_name=regime)
            else:
                peft_model.add_adapter(regime, cfg)
        assert peft_model is not None
        self.peft_model = peft_model
        # The PEFT wrapper exposes the base on .base_model.
        self._active: list[str] = []

    @property
    def base(self) -> nn.Module:
        return self.peft_model

    # -------- expert toggling ----------------------------------------------

    def set_active(self, regimes: list[str]) -> None:
        for r in regimes:
            if r not in self.regimes:
                raise KeyError(r)
        # PEFT's set_adapter accepts a list — adapter outputs are summed.
        self.peft_model.set_adapter(regimes)
        self._active = list(regimes)

    @property
    def active(self) -> list[str]:
        return list(self._active)

    # -------- parameter accounting -----------------------------------------

    def trainable_parameter_groups(self) -> list[dict]:
        """Optimizer parameter groups grouped by regime adapter.

        Useful when you want per-regime LR schedules — not used in MVP but
        kept for ablation. By default, all adapters share the same LR.
        """
        groups: dict[str, list[nn.Parameter]] = {r: [] for r in self.regimes}
        for name, p in self.peft_model.named_parameters():
            if not p.requires_grad:
                continue
            for r in self.regimes:
                if f"lora_A.{r}" in name or f"lora_B.{r}" in name:
                    groups[r].append(p)
                    break
        return [{"params": params, "name": f"lora_{r}"} for r, params in groups.items() if params]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)

    # -------- save / load --------------------------------------------------

    def save(self, path: str) -> None:
        self.peft_model.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls, base: nn.Module, path: str, *, regimes: list[str]
    ) -> ExpertLoRAManager:
        """Reload all per-regime adapters from a save_pretrained() dir.

        Each regime is loaded under its own adapter_name so the router can
        select between them at inference; an earlier version only loaded
        ``regimes[0]`` which left 6 of 7 experts as freshly-initialised
        (silent quality bug).
        """
        mgr = cls(base, regimes=regimes)
        from pathlib import Path as _Path

        root = _Path(path)
        for r in regimes:
            sub = root / r
            adapter_path = sub if sub.is_dir() else root
            mgr.peft_model.load_adapter(str(adapter_path), adapter_name=r)
        return mgr


__all__ = ["ExpertLoRAManager", "ExpertSpec", "LORA_RANK", "DEFAULT_TARGET_MODULES"]

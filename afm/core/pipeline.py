"""``AufschreibePipeline`` — Diffusers ``DiffusionPipeline`` subclass.

The pipeline composes:
    - frozen base DiT (SANA-1.6B by default)
    - ``ExpertLoRAManager`` (7 LoRA adapters)
    - ``RegimeRouter`` (top-2 soft gate)
    - ``MuLANSigmaHead`` (per-pixel σ, clipped)
    - ``PhysicalNoisePrior`` (frozen, register_buffer)
    - ``FlowMatchingScheduler`` (rectified FM)

``infer(prompt, regime)`` and ``morph(...)`` are exposed both as methods on
the pipeline and as top-level functions in ``afm`` for backwards-compatible
imports.

Heavy diffusers imports are deferred to instantiation time so ``afm --help``
stays snappy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from afm.core.expert_lora import ExpertLoRAManager
from afm.core.mulan_sigma import MuLANSigmaHead
from afm.core.physical_noise_prior import PhysicalNoisePrior
from afm.core.regime_router import RegimeRouter
from afm.core.schedulers import FlowMatchingScheduler
from afm.regimes import RegimeSpec, get


def _import_diffusers():
    try:
        from diffusers import DiffusionPipeline

        return DiffusionPipeline
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "afm requires diffusers>=0.30; install with `pip install diffusers`"
        ) from e


_DiffusionPipeline = _import_diffusers()


class WeightsNotAvailableError(RuntimeError):
    """Raised when ``from_pretrained`` is called before the first MVP run
    has produced weights on the Hub. ``afm oss`` step 5 (release) is what
    populates the target repo; before then there's nothing to load.
    """


@dataclass
class InferenceConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    seed: int | None = None
    height: int = 1024
    width: int = 1024


class AufschreibePipeline(_DiffusionPipeline):
    """Composed pipeline. Subclasses ``DiffusionPipeline`` so HF Hub
    upload / download / safetensors plumbing works out of the box.
    """

    def __init__(
        self,
        base: Any,
        lora_manager: ExpertLoRAManager,
        router: RegimeRouter,
        mulan: MuLANSigmaHead,
        prior: PhysicalNoisePrior,
        scheduler: FlowMatchingScheduler,
        *,
        text_encoder: Any = None,
        tokenizer: Any = None,
        vae: Any = None,
    ) -> None:
        super().__init__()
        # Register modules with diffusers' bookkeeping so save/load works.
        self.register_modules(
            base=base,
            lora_manager=lora_manager.peft_model,
            router=router,
            mulan=mulan,
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        # Diffusers expects scheduler on .scheduler — but our FM scheduler
        # is a small dataclass; keep it as a plain attribute.
        self._fm_scheduler = scheduler
        self._lora_manager = lora_manager
        # Map regime name → router output index
        self._regime_index = {r: i for i, r in enumerate(lora_manager.regimes)}

    # -------- construction from HF Hub --------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):  # type: ignore[override]
        """Load a previously released AufschreibePipeline from the Hub.

        This requires step 5 (release) of ``afm oss`` to have run on the
        target repo. Before then the target repo is empty and we raise a
        clear ``WeightsNotAvailableError`` rather than letting diffusers'
        default error path bubble up.
        """
        try:
            from huggingface_hub import HfApi

            HfApi().repo_info(pretrained_model_name_or_path)
        except Exception as e:
            raise WeightsNotAvailableError(
                f"could not load AufschreibePipeline from {pretrained_model_name_or_path!r}. "
                f"This usually means step 5 (release) of `afm oss` has not run yet. "
                f"Run `afm oss --resume --config configs/mvp.yaml` first."
            ) from e
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    # -------- inference: single regime -------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        regime: str | RegimeSpec | None = None,
        config: InferenceConfig | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        config = config or InferenceConfig()
        # Encode prompt → text features
        text_emb = self._encode_prompt(prompt)
        # Build initial noise sample drawn from the requested regime's prior
        spec = self._resolve_regime(regime, text_emb)
        gen = torch.Generator(device=self._device()).manual_seed(
            config.seed if config.seed is not None else torch.seed()
        )
        x = self.prior.sample(
            spec.name,
            (1, 3, config.height, config.width),
            device=self._device(),
            generator=gen,
        )
        # Activate the regime adapter and step.
        self._lora_manager.set_active([spec.name])
        for i in range(config.num_inference_steps):
            t = torch.tensor(
                1.0 - (i + 1) / config.num_inference_steps,
                device=self._device(),
            )
            v = self._predict_velocity(x, t, text_emb, regime=spec.name)
            # Simple Euler step of rectified FM ODE: x ← x - dt * v
            dt = 1.0 / config.num_inference_steps
            x = x - dt * v
        return x

    # -------- morph (DSBM, inference-time) ---------------------------------

    @torch.no_grad()
    def morph(
        self,
        prompt: str,
        regime_a: str,
        regime_b: str,
        steps: int = 50,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        from afm.morph import DSBMConfig, morph_between_regimes

        text_emb = self._encode_prompt(prompt)
        return morph_between_regimes(
            pipeline=self,
            text_emb=text_emb,
            regime_a=regime_a,
            regime_b=regime_b,
            cfg=DSBMConfig(steps=steps),
        )

    # -------- helpers ------------------------------------------------------

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _resolve_regime(self, regime, text_emb) -> RegimeSpec:
        if isinstance(regime, RegimeSpec):
            return regime
        if isinstance(regime, str):
            return get(regime)
        # Auto-route via the router on the prompt feature mean.
        feat = text_emb.mean(dim=1)  # (B, D)
        weights, idx = self.router(feat)
        chosen = idx[0, 0].item()
        return get(list(self._regime_index.keys())[chosen])

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError(
                "AufschreibePipeline was instantiated without text_encoder/tokenizer; "
                "use AufschreibePipeline.from_pretrained(...) to wire them up."
            )
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        ).to(self._device())
        out = self.text_encoder(**tokens)
        return out.last_hidden_state

    def _predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        regime: str,
    ) -> torch.Tensor:
        # Concrete velocity prediction is delegated to the underlying base
        # model's call signature. SANA / Flux differ — kept thin here.
        return self.base(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=text_emb,
        ).sample


__all__ = ["AufschreibePipeline", "InferenceConfig"]

"""Regime specification for Aufschreibesysteme Flow Matching.

Each regime is a (codec, frozen-physical-prior, Aufschreibesystem-epoch) triple.
The registry below MUST contain exactly 7 entries — adding regimes silently
breaks the 7-class ResNet-50 classifier, the load-balance loss normalisation,
the teaser gallery layout and the paper figures. Use ``register`` to add or
override (override is for tests only).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Epoch = Literal["1800", "1900", "2000"]
CodecKind = Literal["pixel", "dct", "vector", "mel"]


class RegimeSpec(BaseModel):
    """Declarative spec for a single inscription regime.

    Fields are deliberately serialisable — RegimeSpec instances are written
    into ``state.json`` so a partial run can be inspected without loading
    torch.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(..., min_length=1)
    epoch: Epoch
    codec: CodecKind
    # Filesystem path to the measurement-derived prior (``register_buffer``
    # source). Looked up under ``data/measurement/{name}.npz`` by default.
    prior_path: str
    # Per-regime σ_min for MuLAN clipping. Set per the measurement noise floor
    # at the channel; learning is clipped against this from below.
    sigma_min: float = Field(..., gt=0.0)
    strength: float = Field(1.0, ge=0.0, le=10.0)
    # Codec-specific knobs (JPEG quality table, EnCodec bandwidth, …).
    codec_params: dict[str, Any] = Field(default_factory=dict)
    # If True the regime supports temporal modality (per-frame σ + Diffusion
    # Forcing). Only film does this in MVP.
    temporal: bool = False

    @field_validator("name")
    @classmethod
    def _name_lower(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError(f"regime name must be lowercase, got {v!r}")
        return v


# ----- registry --------------------------------------------------------------


_REGISTRY: dict[str, RegimeSpec] = {}


def register(spec: RegimeSpec, *, override: bool = False) -> RegimeSpec:
    if not override and spec.name in _REGISTRY:
        raise ValueError(f"regime {spec.name!r} already registered; pass override=True for tests")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> RegimeSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown regime {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_regimes() -> list[RegimeSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def regime_names() -> list[str]:
    return sorted(_REGISTRY)


def epoch_of(name: str) -> Epoch:
    return get(name).epoch


# Aufschreibesystem groupings, used only for documentation/figure ordering.
EPOCH_GROUPS: dict[Epoch, tuple[str, ...]] = {
    "1800": ("parchment", "typewriter"),
    "1900": ("gramophone", "photograph", "film"),
    "2000": ("jpeg", "crt"),
}


N_REGIMES = 7  # invariant — wired into classifier head, gate normalisation, …


def assert_registry_complete() -> None:
    """Raise if the registry does not match the architecture contract.

    Called from ``afm.cli.oss`` at startup, so a misconfigured environment
    fails at step 0 rather than silently 6-way training.
    """
    expected = {n for group in EPOCH_GROUPS.values() for n in group}
    actual = set(_REGISTRY)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise RuntimeError(
            f"regime registry mismatch — missing={sorted(missing)} extra={sorted(extra)}"
        )
    if len(_REGISTRY) != N_REGIMES:
        raise RuntimeError(
            f"regime registry must contain exactly {N_REGIMES} entries, got {len(_REGISTRY)}"
        )


def _load_all() -> None:
    """Import every regime module so its ``register(...)`` call fires.

    Called from afm.regimes.__init__ — import order matters only for the
    failure message of ``assert_registry_complete``.
    """
    from afm.regimes import (  # noqa: F401
        crt,
        film,
        gramophone,
        jpeg,
        parchment,
        photograph,
        typewriter,
    )

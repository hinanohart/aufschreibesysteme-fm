"""Measurement-derived, **frozen** physical noise prior.

The core invariant: a regime's noise floor is read once from a measurement
file and never updated by gradient descent. This is the architectural choice
that keeps the "Real" channel (in Kittler's Lacanian sense) outside the
symbolic — making it learnable would collapse the whole regime back into a
generic stylisation prior.

Concretely:
    - PSDs are loaded from ``data/measurement/<regime>.npz``
    - Stored via ``register_buffer`` (not ``nn.Parameter``)
    - Sampling: ε_r = F⁻¹(√PSD_r · F(z)), z ~ N(0, I)
    - No code path in this module accepts ``requires_grad=True`` for the prior
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn

from afm.regimes import RegimeSpec, all_regimes, get


def _buffer_name(regime: str) -> str:
    return f"psd_{regime}"


def _load_psd(path: str | Path, *, fallback_shape: tuple[int, ...]) -> torch.Tensor:
    """Load a measurement-derived PSD tensor from .npz.

    If the file is missing we synthesise a deterministic placeholder so the
    rest of the pipeline can boot. ``afm oss`` step 0 (env) fails loudly if
    placeholders are in use at training time — see ``warn_if_synthetic``.
    """
    p = Path(path)
    if p.is_file():
        try:
            arr = np.load(p)
            # convention: key "psd" inside the npz; squeeze trailing 1s
            psd = arr["psd"] if "psd" in arr.files else arr[arr.files[0]]
            t = torch.from_numpy(np.ascontiguousarray(psd)).float()
            return t
        except (OSError, KeyError, ValueError) as e:
            raise RuntimeError(f"failed to read measurement file {p}: {e}") from e
    # Synthetic placeholder — pink-ish 1/f for image regimes; harmless until
    # actual measurement is available. We deliberately tag the buffer.
    h, w = fallback_shape[-2:]
    fy = torch.fft.fftfreq(h).abs()
    fx = torch.fft.fftfreq(w).abs()
    grid = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) + 1e-6
    psd = (1.0 / grid).clamp(max=1e6)
    psd = psd / psd.mean()
    return psd


class PhysicalNoisePrior(nn.Module):
    """Frozen physical-noise prior across all registered regimes.

    Buffers are registered as ``psd_{regime}``. Nothing in this module
    creates ``nn.Parameter`` and nothing toggles ``requires_grad`` on the
    buffers. The buffers move with ``.to(device)`` like normal.
    """

    SYNTHETIC_TAG_BUFFER = "_synthetic_mask"

    def __init__(
        self,
        regimes: list[RegimeSpec] | None = None,
        *,
        spatial_shape: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.spatial_shape = spatial_shape
        self._regime_names: list[str] = []
        self._synthetic: set[str] = set()
        regimes = regimes if regimes is not None else all_regimes()
        for spec in regimes:
            self._regime_names.append(spec.name)
            path = Path(spec.prior_path)
            tensor = _load_psd(path, fallback_shape=spatial_shape)
            # Pad/crop to a known shape so the inverse FFT path is regular.
            tensor = self._fit_to_shape(tensor, spatial_shape)
            # Use register_buffer so .to() works but no grad is attached.
            # persistent=True so the measurement-derived PSD travels with the
            # state_dict / safetensors file. Without this an HF Hub clone of
            # the released checkpoint would silently fall back to synthetic
            # PSDs (no error path, just wrong physics).
            self.register_buffer(_buffer_name(spec.name), tensor, persistent=True)
            if not path.is_file():
                self._synthetic.add(spec.name)

    @staticmethod
    def _fit_to_shape(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if t.ndim == 1:
            # 1-D PSDs (audio) — replicate along time as 2-D for unified path
            t = t[None, :].expand(shape[0], -1)
        h, w = shape
        th, tw = t.shape[-2], t.shape[-1]
        # center-crop or zero-pad to target shape
        if th >= h:
            t = t[..., (th - h) // 2 : (th - h) // 2 + h, :]
        else:
            pad_h = (h - th) // 2
            t = torch.nn.functional.pad(t, (0, 0, pad_h, h - th - pad_h))
        if tw >= w:
            t = t[..., :, (tw - w) // 2 : (tw - w) // 2 + w]
        else:
            pad_w = (w - tw) // 2
            t = torch.nn.functional.pad(t, (pad_w, w - tw - pad_w, 0, 0))
        # Normalise so sampling has unit variance baseline.
        t = t / (t.mean() + 1e-12)
        return t

    # -------- public API ----------------------------------------------------

    @property
    def regime_names(self) -> list[str]:
        return list(self._regime_names)

    def psd(self, regime: str) -> torch.Tensor:
        if regime not in self._regime_names:
            raise KeyError(regime)
        return getattr(self, _buffer_name(regime))

    @torch.no_grad()
    def sample(
        self,
        regime: str,
        shape: tuple[int, ...],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample coloured noise ε_r from the regime PSD via inverse FFT.

        Output is real-valued, has the requested shape, and has unit-scale
        variance (the prior is normalised in ``__init__``). The function is
        marked ``no_grad`` because the prior must not appear on any autograd
        graph — see Kittler/Lacan caveat in ``docs/appendix_kittler.md``.
        """
        psd = self.psd(regime).to(
            device=device if device is not None else self.psd(regime).device,
            dtype=dtype or self.psd(regime).dtype,
        )
        # Broadcast PSD to spatial dims of `shape`; sample white in `shape`.
        h, w = psd.shape[-2], psd.shape[-1]
        if shape[-2:] != (h, w):
            # Resize via bilinear in log-domain for stability — fine for MVP.
            psd = (
                torch.nn.functional.interpolate(
                    psd.log().unsqueeze(0).unsqueeze(0),
                    size=shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .exp()
            )
        z = torch.randn(shape, generator=generator, device=psd.device, dtype=psd.dtype)
        Z = fft.fft2(z, dim=(-2, -1))
        eps = fft.ifft2(torch.sqrt(psd) * Z, dim=(-2, -1)).real
        # Strip the DC component per sample/channel — 1/f-shaped PSDs leave a
        # ±15–50 mean on each sample, which would silently inflate any
        # velocity-loss target that consumed `eps` directly. Centre first,
        # then unit-variance normalise. (Regression: test_sample_runs_under_no_grad.)
        mean = eps.flatten(start_dim=-2).mean(dim=-1, keepdim=True).unsqueeze(-1)
        eps = eps - mean
        std = eps.flatten(start_dim=-2).std(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
        return eps / std

    @torch.no_grad()
    def sigma_min(self, regime: str) -> float:
        """Return the regime's MuLAN clip-floor (read from RegimeSpec)."""
        return get(regime).sigma_min

    # -------- diagnostics ---------------------------------------------------

    def warn_if_synthetic(self) -> list[str]:
        """Return regimes whose PSD is the synthetic placeholder.

        ``afm.cli.oss`` calls this at step 0; non-empty list aborts training
        with a clear "go run ``scripts/fetch_measurements.py`` first" error.
        """
        return sorted(self._synthetic)

    # -------- safety check (no learnable prior) -----------------------------

    def _assert_frozen(self) -> None:
        """Internal: confirm no buffer was accidentally promoted to Parameter."""
        for name in self._regime_names:
            buf = getattr(self, _buffer_name(name))
            if isinstance(buf, nn.Parameter):
                raise RuntimeError(
                    f"physical prior {name} became nn.Parameter — Lacan misreading guard tripped"
                )
            if buf.requires_grad:
                raise RuntimeError(f"physical prior {name} has requires_grad=True — must be frozen")


__all__ = ["PhysicalNoisePrior"]

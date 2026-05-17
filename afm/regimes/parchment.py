"""Parchment / handwriting regime (Aufschreibesystem 1800).

Codec: DiffVG vector — strokes are first-class, raster only at the very end.
Prior: gaussian ink-bleed kernel, σ measured from a held-out scan of Codex
Sinaiticus folios (CC0, see ``data/manifests/parchment.csv``).
"""

from afm.regimes.base import RegimeSpec, register

PARCHMENT = register(
    RegimeSpec(
        name="parchment",
        epoch="1800",
        codec="vector",
        prior_path="data/measurement/parchment.npz",
        sigma_min=1.0e-3,
        strength=1.0,
        codec_params={
            "stroke_max_points": 32,
            "stroke_max_width": 4.0,
            "raster_size": 512,
            "anti_alias": True,
            # Ink-bleed kernel sigma in raster pixels; fits the measured
            # ink-on-vellum gaussian PSF.
            "bleed_sigma_px": 0.85,
        },
        temporal=False,
    )
)

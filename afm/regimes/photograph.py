"""Photograph regime — still-optical (Aufschreibesystem 1900).

Distinct from ``film`` (which is moving-optical + temporal). Codec: pixel.
Prior: ISO12233 grain profile measured from a Kodak Tri-X archival scan
(PD via Kodak archive). Mixed Poisson + gaussian model — Poisson on photon
counts, gaussian on read-out.

Critic C4 fix: photograph is its own regime so the 1900 triad
(gramophone + photograph + film) is complete and the 7-class classifier sees
optical-still and optical-moving as distinct, as Kittler does.
"""

from afm.regimes.base import RegimeSpec, register

PHOTOGRAPH = register(
    RegimeSpec(
        name="photograph",
        epoch="1900",
        codec="pixel",
        prior_path="data/measurement/photograph.npz",
        sigma_min=3.0e-3,
        strength=1.0,
        codec_params={
            # Tri-X 400 → measured at ISO12233 chart.
            "grain_iso": 400,
            "poisson_gain": 1.0,
            "gaussian_readout_sigma": 0.01,
            # Grain power spectrum 1/f-ish, fit to measured PSD.
            "psd_kind": "pink_clipped",
        },
        temporal=False,
    )
)

"""Film regime — moving optical with temporal grain (Aufschreibesystem 1900).

Distinct from ``photograph`` (still). Codec: pixel + temporal extension.
Prior: ISO12233 grain plus *per-frame* Diffusion Forcing σ to handle the
fact that grain decorrelates frame-to-frame.

This is the only regime that touches the temporal scheduler path
(``afm/core/schedulers.py``); ``temporal=True`` is what gates that.
"""

from afm.regimes.base import RegimeSpec, register

FILM = register(
    RegimeSpec(
        name="film",
        epoch="1900",
        codec="pixel",
        prior_path="data/measurement/film.npz",
        sigma_min=4.0e-3,
        strength=1.0,
        codec_params={
            "grain_iso": 200,
            "fps": 24,
            # Inter-frame grain correlation — measured from a 35 mm
            # archival scan. ~0.15 means almost-but-not-fully decorrelated.
            "interframe_corr": 0.15,
            # Telecine cadence (3:2 pulldown) is optional — disabled in MVP
            # because the temporal eval set is short clips.
            "telecine_pulldown": False,
        },
        temporal=True,
    )
)

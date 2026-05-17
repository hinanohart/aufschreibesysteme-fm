"""Typewriter regime (Aufschreibesystem 1800).

Codec: pixel. Prior: impact-PDF — the per-glyph displacement+pressure noise
sampled from a stack of Olivetti Lettera-22 typewritten pages (LoC PD).
The PDF is stored as a binned 2-D histogram and sampled at training time.
"""

from afm.regimes.base import RegimeSpec, register

TYPEWRITER = register(
    RegimeSpec(
        name="typewriter",
        epoch="1800",
        codec="pixel",
        prior_path="data/measurement/typewriter.npz",
        sigma_min=2.0e-3,
        strength=1.0,
        codec_params={
            # Discrete impact glyph grid (monospace).
            "pitch_x_px": 16,
            "pitch_y_px": 22,
            # Ribbon-ink bleed measured at 600 dpi scan.
            "bleed_sigma_px": 0.5,
            # Mis-strike probability — small but non-zero, matches the corpus.
            "misstrike_p": 0.012,
        },
        temporal=False,
    )
)

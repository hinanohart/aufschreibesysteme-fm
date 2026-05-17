"""JPEG regime (Aufschreibesystem 2000).

Codec: DCT 8×8 with the standard luminance + chrominance quantisation tables
at three quality levels (10 / 50 / 90). The model learns to recover the
posterior over the quantisation cell, conditioned on Q.

The prior is the DCT-domain Laplacian fitted to a held-out CLIC2020 split
(CC-BY). The Q level is sampled uniformly during training so the model
sees all three regimes.
"""

from afm.regimes.base import RegimeSpec, register

JPEG = register(
    RegimeSpec(
        name="jpeg",
        epoch="2000",
        codec="dct",
        prior_path="data/measurement/jpeg.npz",
        sigma_min=2.0e-3,
        strength=1.0,
        codec_params={
            "block": 8,
            "qualities": [10, 50, 90],
            # Sampling distribution at training time.
            "quality_sampling": "uniform",
            "subsample": "4:2:0",
            # DCT-domain Laplacian scale (b in Lap(0, b)) fitted on the
            # CLIC2020 split.
            "laplacian_b": 0.18,
        },
        temporal=False,
    )
)

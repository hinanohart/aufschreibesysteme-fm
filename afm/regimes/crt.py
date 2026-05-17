"""CRT regime (Aufschreibesystem 2000).

Codec: pixel with horizontal-scanline + phosphor-decay kernel.
Prior: a published phosphor-decay double-exponential plus a measured
scanline gap. We use the well-known ``P22`` phosphor RGB triplet kinetics
as a baseline; the exact kernel is computed lazily at training time so we
can swap in measured kernels later.
"""

from afm.regimes.base import RegimeSpec, register

CRT = register(
    RegimeSpec(
        name="crt",
        epoch="2000",
        codec="pixel",
        prior_path="data/measurement/crt.npz",
        sigma_min=3.0e-3,
        strength=1.0,
        codec_params={
            # P22 phosphor decay time constants (ms). Two-exponential mix.
            "phosphor_tau1_ms": 1.5,
            "phosphor_tau2_ms": 12.0,
            "phosphor_mix": 0.7,
            # Scanline gap measured as fraction of pixel pitch (~0.3 means
            # 30 % of each pixel is dark).
            "scanline_gap": 0.3,
            # Triadal RGB sub-pixel layout — fixed for MVP.
            "triad_layout": "rgb",
            "fps": 60,
        },
        temporal=False,
    )
)

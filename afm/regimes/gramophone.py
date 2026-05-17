"""Gramophone regime (Aufschreibesystem 1900).

Codec: EnCodec mel. Prior: 78 rpm shellac PSD measured from Wikimedia Commons
PD wax-cylinder transfers AND items from a hand-curated whitelist (see
``data/manifests/audio_whitelist.csv``). **Internet Archive 78rpm bulk import
is deliberately not used** — EU neighbouring rights run 70 years past the
performer's death and many items in the IA bulk are still in copyright in EU
jurisdictions. ``scripts/audit_audio.py`` hard-fails on items not in the
whitelist.
"""

from afm.regimes.base import RegimeSpec, register

GRAMOPHONE = register(
    RegimeSpec(
        name="gramophone",
        epoch="1900",
        codec="mel",
        prior_path="data/measurement/gramophone.npz",
        sigma_min=5.0e-3,
        strength=1.0,
        codec_params={
            "sample_rate": 24000,
            "encodec_bandwidth_kbps": 6.0,
            "encodec_n_quantizers": 8,
            # PSD shaped to 78 rpm shellac surface noise — frequency response
            # peaks ~2.5 kHz, sharp roll-off above 5 kHz, audible noise floor
            # below 200 Hz.
            "psd_peak_hz": 2500.0,
            "psd_rolloff_hz": 5000.0,
            "lowend_noise_floor_hz": 200.0,
        },
        temporal=False,
    )
)

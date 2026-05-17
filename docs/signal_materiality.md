# Signal Materiality — the main pitch

We claim one thing:

> Lossy codecs are not styles. Each is its own first-class generative
> regime, with its own measured noise floor, its own inscription geometry,
> and its own posterior. Treating them as post-hoc filters is the reason
> generic style-transfer pipelines fail to keep semantic identity stable
> across "regimes."

This document is signal-materiality only — no media-theoretical framing.
For the Kittler / Aufschreibesysteme motivation see
`docs/appendix_kittler.md` (it is appendix-only by design).

---

## Why codecs are not styles

A JPEG at Q=10 is not "a noisy version of a clean image." It is the
*posterior* over images consistent with a particular DCT-domain
quantisation. Sampling from that posterior requires (a) the DCT lattice
geometry, (b) the quantisation table, and (c) a prior over what the clean
signal looks like *given* the quantisation lattice. That prior is not the
same as the prior over clean natural images — it's offset by the lattice.

Style-LoRA stacks miss (a) and (b) entirely. They learn a marginal
appearance and then apply it everywhere, which is why JPEG-style LoRA
output never looks JPEG-correct on small text or hard edges.

The same argument applies to:

| Regime     | The geometry that styles ignore                                  |
| ---------- | ---------------------------------------------------------------- |
| gramophone | 78 rpm shellac PSD peaks at ~2.5 kHz; rolls off sharply > 5 kHz  |
| photograph | ISO12233 grain → mixed Poisson + gaussian read-out               |
| film       | grain decorrelates frame-to-frame (interframe corr ≈ 0.15)       |
| crt        | scanline gap ≈ 30 % pixel pitch; P22 phosphor 1.5/12 ms decay    |
| typewriter | discrete glyph pitch (16 × 22 px); ribbon-ink ~0.5 px bleed      |
| parchment  | DiffVG vector strokes + ink-bleed gaussian PSF                   |
| jpeg       | DCT 8 × 8 quantisation, Q ∈ {10, 50, 90}                         |

In every case, the *channel geometry* is non-negligible information that a
post-hoc style filter throws away.

---

## What we do instead

For each regime we attach a **frozen physical noise prior** derived from
measurement and a **rank-32 LoRA expert**. A top-2 soft mixture of experts
routes per-token between regimes. The trainer never updates the prior.

The MuLAN per-pixel σ head is clipped from below by the regime's physical
noise floor, so the model cannot wish away the channel's noise.

Concretely:

```
ε_r = F⁻¹(√PSD_r · F(z)),  z ~ N(0, I)
σ_pred(x, r) = max(σ_min^(r), MuLAN_θ(x))
weights = softmax(MLP(layer4_patch_mean ⊕ text_cls))
L_total = E_t [‖v_θ(x_t, r) - (noise - x_0)‖² / σ_pred²] + λ N Σ_r f_r P_r
```

That's the entire technical content. The Kittler framing is what made us
write the regime list the way we did; it is not part of any loss term.

---

## How we evaluate

| Metric | What it measures                                       | Target            |
| ------ | ------------------------------------------------------ | ----------------- |
| M1     | regime classifier recall (ResNet-50, 7-class)          | ≥ 0.85            |
| M2     | per-regime FID against 5 k real refs                   | ≤ baseline        |
| M3     | morph smoothness (LPIPS trajectory monotonicity, PPL)  | monotonicity > 0.90 |
| M4     | inscription consistency (CLIPScore variance across r)  | variance ≤ 0.05   |

Baselines: B1 vanilla SANA + style-LoRA stack, B2 SANA + IP-Adapter,
B3 DDCM (single-regime baseline), B4 FGA-NN (grain sanity). All baselines
share captions, step budget, eval seed and the frozen text encoder.

---

## What the metrics will catch

- **M1 ≥ 0.85 but FID worse than baseline:** the LoRAs overfit the *channel
  identity* at the cost of image quality. Mitigation in `docs/architecture.md`
  under "Extensions in scope."
- **M3 monotonicity < 0.90:** the morph trajectory loops back. The DSBM
  bridge in `afm/morph/dsbm.py` is single-iteration IPF in MVP; the full
  IPF lives in the post-MVP extension list.
- **M4 variance > 0.05:** semantic identity is shifting across regimes —
  the experts have been allowed to leak content. Tighten the load-balance
  λ floor in `configs/mvp.yaml`.

---

## What this is not

- **A style transfer system.** See "Why codecs are not styles" above.
- **A claim about Kittler.** The Kittler framing is the motivation that
  made us pick *these* seven regimes rather than the obvious five. The
  metrics here are all signal-materiality.
- **Not a fully automatic pipeline.** Four explicit human-review gates by
  design. See `docs/architecture.md` and the README.

# Appendix — Kittler motivation (and Lacan caveat)

This appendix exists so people can audit our use of Kittler. The technical
content sits in `docs/signal_materiality.md` and `docs/architecture.md`;
nothing in the codebase depends on the Lacanian R/S/I triad being correct.

---

## Why the seven regimes are the seven

Friedrich Kittler distinguished three historical *Aufschreibesysteme*
("inscription networks"):

| Epoch | Inscription                                       | Our regimes                            |
| ----- | ------------------------------------------------- | -------------------------------------- |
| 1800  | handwriting; the rise of bourgeois reading        | `parchment`, `typewriter`              |
| 1900  | gramophone + film + photograph (the 1900 triad)   | `gramophone`, `photograph`, `film`     |
| 2000  | software-mediated digital signal                  | `jpeg`, `crt`                          |

The 1900 triad is non-negotiable in Kittler's reading — typewriter,
gramophone, and film/photograph are the three channels through which the
modern subject is constituted (the famous "Gramophone, Film, Typewriter"
argument). Photograph and film are functionally separated in our codebase
because still-optical and moving-optical have different temporal
behaviour: film grain decorrelates frame-to-frame and demands a
per-frame σ schedule (Diffusion Forcing), photograph does not.

---

## Caveat: the Lacan dependency

Kittler reads the gramophone / film / typewriter triad through Lacan's
Real / Symbolic / Imaginary. "The Real" is what cannot be inscribed —
the noise floor of the channel. "The Symbolic" is what discretises —
the typewriter as the model. "The Imaginary" is what registers
analogue continuity — film.

The temptation is to map our components 1-to-1:

| Lacan      | Component                  | Status          |
| ---------- | -------------------------- | --------------- |
| Real       | physical noise prior       | frozen          |
| Symbolic   | LoRA adapters (codecs)     | learnable       |
| Imaginary  | MuLAN per-pixel σ          | learnable, clipped |

That mapping is suggestive but not load-bearing in the code. We
explicitly keep the physical prior frozen — see `_assert_frozen()` in
`afm/core/physical_noise_prior.py`. The reason is that a *learnable*
Real layer is just Symbolic with extra steps: the Lacanian object disappears
the moment gradient descent is allowed to update it. The architectural
choice to freeze it is the only place where the Lacan reading constrains
the implementation.

Geoffrey Winthrop-Young's discussion of "productively unfaithful"
inheritance is the most relevant secondary reading here: we are
deliberately *productively unfaithful* to the strict Lacanian frame, and
we say so loudly to avoid the standard Lacan-misreading critique.

---

## Secondary literature we consulted

- Geoffrey Winthrop-Young, *Kittler and the Media* (Polity, 2011).
- John Durham Peters, *The Marvelous Clouds: Toward a Philosophy of
  Elemental Media* (Chicago, 2015).
- Sybille Krämer, *Medium, Messenger, Transmission: An Approach to Media
  Philosophy* (Amsterdam University Press, 2015).

Three claims of Kittler's that we treat as load-bearing for motivation:

1. *Discourse network = data network.* The artefacts a culture can
   inscribe shape the discourse that culture can have. We translate this
   into "the channel's physical prior must show up at training time."
2. *Hardware determines subjectivity.* We translate this weakly: the
   model has to *see* the codec as a structural feature, not a label.
   Hence first-class regimes, not style adapters.
3. *R/S/I = channels.* We acknowledge but do not depend on this — see
   "Caveat: the Lacan dependency" above.

We cross-checked these against Winthrop-Young's reading first, then
Peters and Krämer. Divergences (e.g. Krämer's broader media-philosophy
frame vs Kittler's narrower discourse-network claim) are noted in the
paper appendix.

---

## What the paper will and will not claim

The paper will claim:

- These seven regimes are an empirically motivated and theoretically
  defensible regime list for a study of inscription-channel-conditional
  generation.
- Frozen, measurement-derived priors are *materially* different from
  learnable noise floors, and the difference shows up in M4 (inscription
  consistency).
- The architectural choice traces back to a media-theoretical reading,
  acknowledged with citations.

The paper will not claim:

- That this is an implementation of Kittler.
- That M1 / M2 / M3 / M4 directly measure anything Lacanian.
- That "the Real" has been captured in the buffer tensor. (It is a
  frozen tensor. That is exactly the point. It cannot speak.)

---

## Why the Kittler framing is in an appendix, not the abstract

To reach an audience beyond media studies, the signal-materiality pitch
has to lead. The Kittler framing is what made us pick *these* seven
regimes and what justifies the frozen prior, but it cannot carry the
evaluation. We are not the first paper to make this kind of choice —
see the way *Generative Camera Noise* (2023) cites sensor physics in
the body and historical context in the appendix.

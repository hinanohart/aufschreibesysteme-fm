---
title: Aufschreibesysteme Flow Matching
emoji: 🖋️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Aufschreibesysteme Flow Matching — demo Space

This Space serves three things:

1. **Single-regime inference** — pick a regime, type a prompt, get a sample
2. **Cross-regime morph** — DSBM-based interpolation between any two regimes
3. **7-regime gallery** — the teaser image, regenerated live for your prompt

Trained weights are loaded from
[`hinanohart/aufschreibesysteme-fm`](https://huggingface.co/hinanohart/aufschreibesysteme-fm).
Source: [github.com/hinanohart/aufschreibesysteme-fm](https://github.com/hinanohart/aufschreibesysteme-fm).

This is a **research preview**, not a production system. The underlying
pipeline is a resumable semi-auto OSS workflow with 4 explicit
human-review gates — see the README in the repo for details.

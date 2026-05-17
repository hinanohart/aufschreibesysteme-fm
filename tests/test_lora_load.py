"""``ExpertLoRAManager.from_pretrained`` loads all 7 adapters.

Earlier version only loaded ``regimes[0]`` which left 6 of 7 experts as
freshly-initialised at inference time (silent quality bug).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")


def test_from_pretrained_calls_load_adapter_for_every_regime(tmp_path, monkeypatch) -> None:
    import torch.nn as nn

    from afm.core import expert_lora as el

    calls: list[tuple[str, str]] = []

    class _FakePeftModel:
        def __init__(self):
            self.adapters = []

        def add_adapter(self, name, cfg):
            self.adapters.append(name)

        def set_adapter(self, names):
            self.active = names

        def load_adapter(self, path, adapter_name):
            calls.append((str(path), adapter_name))
            self.adapters.append(adapter_name)

        def parameters(self):
            return iter(())

        def named_parameters(self):
            return iter(())

    def _fake_get_peft_model(base, cfg, adapter_name):
        m = _FakePeftModel()
        m.add_adapter(adapter_name, cfg)
        return m

    monkeypatch.setattr(el, "get_peft_model", _fake_get_peft_model)

    base = nn.Linear(4, 4)
    regimes = ["parchment", "typewriter", "gramophone", "photograph", "film", "jpeg", "crt"]
    # Provide a directory with one subdir per regime so the path resolution
    # path is exercised.
    root = tmp_path / "ckpt"
    for r in regimes:
        (root / r).mkdir(parents=True, exist_ok=True)

    mgr = el.ExpertLoRAManager.from_pretrained(base, str(root), regimes=regimes)
    loaded_regimes = sorted({r for _, r in calls})
    assert loaded_regimes == sorted(regimes), (
        f"from_pretrained must load all 7 adapters; got {loaded_regimes}"
    )
    # smoke: manager still has the regimes list
    assert mgr.regimes == regimes

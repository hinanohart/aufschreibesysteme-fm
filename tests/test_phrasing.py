"""Phrasing audit — keep us honest about the "fully automatic" rule.

The rule is: do not *claim* the pipeline is fully automatic. The phrase is
allowed in negation contexts (e.g. "this is NOT a fully automatic pipeline")
because that is the exact disclaimer we want. The regex below requires a
negation token within ~60 chars preceding any "fully automatic" occurrence.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

USER_FACING = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/architecture.md",
    REPO_ROOT / "docs/signal_materiality.md",
    REPO_ROOT / "docs/reproducibility.md",
    REPO_ROOT / "docs/appendix_kittler.md",
    REPO_ROOT / "space/README.md",
    REPO_ROOT / "afm/cli/oss.py",
]

NEGATION_TOKENS = (
    "not",
    "never",
    "without",
    "non-",
    "nor",
    " no ",
    "rather than",
    "instead of",
)

PHRASE_RE = re.compile(r"fully[\s-]+automatic", re.IGNORECASE)


def _all_claims_are_negated(text: str) -> tuple[bool, list[str]]:
    """Return (ok, bad_snippets). A claim is OK iff a NEGATION_TOKEN appears
    in the ~60 chars preceding the phrase.
    """
    bad: list[str] = []
    for m in PHRASE_RE.finditer(text):
        start = max(0, m.start() - 60)
        prefix = text[start : m.start()].lower()
        if not any(tok in prefix for tok in NEGATION_TOKENS):
            bad.append(text[max(0, m.start() - 30) : m.end() + 30])
    return (not bad, bad)


def test_no_unnegated_fully_automatic_claim_in_user_facing_docs() -> None:
    for f in USER_FACING:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        ok, bad = _all_claims_are_negated(text)
        assert ok, f"{f} contains an UN-negated 'fully automatic' phrase:\n" + "\n".join(
            f"  ... {b} ..." for b in bad
        )


def test_resumable_semi_auto_phrase_appears_at_least_once() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "resumable semi-auto" in readme or "resumable semi auto" in readme


def test_negation_detector_works() -> None:
    ok, _ = _all_claims_are_negated("It is NOT a fully automatic pipeline.")
    assert ok
    ok, bad = _all_claims_are_negated("This is a fully automatic pipeline.")
    assert not ok
    assert bad

"""Docstring regression tests for the memorization_risk field.

Background (F1, 2026-05-31): the SDK 1.0.12 ValidationReport and
GenerationResult both shipped a docstring on
``memorization_risk`` that described the verdict as coming from
"NN-distance + MIA accuracy". MIA (Membership Inference Attack) was
dropped server-side; the surviving signal is the Carlini-style
NN-distance ratio only.

These tests pin the docstring so it can't drift back to mentioning
MIA / membership inference and mislead customers about what the
verdict actually measures.
"""

from __future__ import annotations

from sablier_flow.types import GenerationResult, ValidationReport


def _memorization_risk_doc(cls: type) -> str:
    """Pull the inline docstring for the ``memorization_risk`` attribute
    off the class's ``__doc__`` block.

    The dataclass docstrings live in the source file under each field
    via the ``attr: T = None\\n\\"\\"\\"...\\"\\"\\"`` pattern, so we read
    the source rather than ``__doc__`` (which only carries the class
    summary)."""

    import inspect

    src = inspect.getsource(cls)
    marker = "memorization_risk:"
    idx = src.index(marker)
    # Grab everything from the field declaration to the next blank line
    # — that captures the trailing triple-quoted docstring.
    tail = src[idx:]
    # Stop at the next field declaration (line starting with a name and
    # ``:`` after a blank line) — for our purposes splitting on the
    # closing ``"""`` is sufficient.
    parts = tail.split('"""')
    assert len(parts) >= 3, f"no docstring found after {marker} in {cls.__name__}"
    return parts[1]


def test_ValidationReport_memorization_risk_doc_drops_MIA() -> None:
    doc = _memorization_risk_doc(ValidationReport)
    assert "MIA" not in doc, (
        "ValidationReport.memorization_risk docstring must not mention "
        "MIA — that signal was dropped server-side."
    )
    assert "membership inference" not in doc.lower(), (
        "ValidationReport.memorization_risk docstring must not mention "
        "membership inference — that signal was dropped server-side."
    )


def test_GenerationResult_memorization_risk_doc_drops_MIA() -> None:
    doc = _memorization_risk_doc(GenerationResult)
    assert "MIA" not in doc, (
        "GenerationResult.memorization_risk docstring must not mention "
        "MIA — that signal was dropped server-side."
    )
    assert "membership inference" not in doc.lower(), (
        "GenerationResult.memorization_risk docstring must not mention "
        "membership inference — that signal was dropped server-side."
    )

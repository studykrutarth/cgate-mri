"""Grounded report generation (Project-II).

BraTS ships no free-text reports, so a report generator cannot be trained or
evaluated against reference text. What it *can* do -- and what this package
does -- is generate a radiology-style report from the system's own structured
findings, with one non-negotiable property:

    **every sentence in a released report is traceable to evidence** -- a
    calibrated finding probability, an image-quality metric, or the gate
    decision. A VLM may contribute qualitative appearance phrases, but only
    after a veto filter that removes any phrase contradicting the findings.

That property is what makes generated text auditable rather than fluent
hallucination, and it is the same discipline the rest of the project applies
to numbers: the gate decides what is released, and the ledger says why.

Layout:
    grounding.py   the contract: sentences, evidence, vetoes
    template.py    deterministic findings -> prose
    vlm.py         optional VLM phrases over a 4-sequence mosaic
    render.py      assembles the report + ledger
    cli.py         `cgate-report --preds predictions.json ...`
"""

from .grounding import (  # noqa: F401
    Evidence,
    ReportCase,
    Sentence,
    VetoRule,
    VETO_RULES,
    assert_grounded,
    vetoes_for,
    filter_phrases,
)
from .render import Report, render_report  # noqa: F401

"""The grounding contract: what makes a generated sentence auditable.

BraTS ships no free-text reports, so there is no reference text to train a
generator on or score output against (BLEU against nothing is not a metric).
What *can* be held fixed is a structural property instead:

    every sentence in a released report carries evidence -- the calibrated
    probability of a specific finding, an image-quality metric, or the gate
    decision itself.

A sentence with no evidence is not emitted. A VLM-contributed phrase is kept
only if it survives `filter_phrases`, which vetoes any phrase contradicting
the system's own findings (a fluent "no evidence of enhancement" must not
survive next to et_present = present, p = 0.97).

The ledger this produces is the report's equivalent of the project's metric
tables: not "does it read well" but "which number says this".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..schema import (
    FINDING_SCHEMA,
    validate_probs,
)

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

# Source vocabularies. Kept as prefixes rather than an enum so a new evidence
# type cannot be invented silently -- assert_grounded checks the prefix.
SOURCE_PREFIXES = ("finding:", "iqm:", "gate", "vlm", "meta")


@dataclass(frozen=True)
class Evidence:
    """One source backing a sentence.

    `source` is `finding:<name>`, `iqm:<name>`, `gate`, `vlm`, or `meta`.
    `detail` records the number or decision the sentence actually used, so the
    ledger can be checked against the report without re-running the model.
    """

    source: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not any(self.source == p or self.source.startswith(p) for p in SOURCE_PREFIXES):
            raise ValueError(
                f"unknown evidence source {self.source!r}; expected one of {SOURCE_PREFIXES}"
            )


@dataclass
class Sentence:
    """One report sentence plus everything that justifies it."""

    text: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("empty sentence")
        if not self.evidence:
            raise ValueError(f"ungrounded sentence: {self.text!r}")


def assert_grounded(sentences: list[Sentence]) -> None:
    """Fail loudly if any sentence slipped out without evidence."""
    for s in sentences:
        if not s.evidence:
            raise AssertionError(f"ungrounded sentence: {s.text!r}")
        for e in s.evidence:
            # Re-runs the prefix check deliberately: Evidence constructed
            # through dataclasses.replace must not bypass the vocabulary.
            Evidence(e.source, e.detail)


# ---------------------------------------------------------------------------
# The case a report is generated from
# ---------------------------------------------------------------------------


@dataclass
class ReportCase:
    """Everything report generation is allowed to know about one case.

    Deliberately a flat snapshot rather than a live TrainedSystem: reports are
    rendered from what was *exported*, so a report can be regenerated and
    audited long after the fitted models are gone.
    """

    patient_id: str
    cohort: str
    probs: list[np.ndarray]  # K arrays of shape (n_classes,)
    confidence: float
    gate_threshold: float
    accepted: bool  # gate decision: released vs deferred
    iqms: dict[str, float] = field(default_factory=dict)  # raw, per metric name
    q_std: tuple[float, ...] = ()  # standardised router features (z-scores)
    q_names: tuple[str, ...] = ()
    labels: np.ndarray | None = None  # ground truth if available (for notes only)
    correct: bool | None = None

    def __post_init__(self) -> None:
        rows = [np.asarray(p, dtype=float)[None, :] for p in self.probs]
        validate_probs(rows, context=f"report case {self.patient_id!r}")
        self.probs = [r[0] for r in rows]
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must lie in [0, 1], got {self.confidence!r}"
            )

    # -- predictions --------------------------------------------------------

    @property
    def predicted(self) -> dict[str, tuple[str, float]]:
        """finding name -> (predicted class, its probability)."""
        out: dict[str, tuple[str, float]] = {}
        for spec, p in zip(FINDING_SCHEMA, self.probs):
            k = int(np.argmax(p))
            out[spec.name] = (spec.classes[k], float(p[k]))
        return out

    @property
    def predicted_names(self) -> dict[str, str]:
        return {k: v[0] for k, v in self.predicted.items()}

    @property
    def weakest_critical(self) -> tuple[str, str, float]:
        """The critical finding the system is least sure about.

        A deferred report must say *why* it was deferred, and "the weakest
        critical finding" is the only reason the gate itself encodes: case
        confidence is the minimum over critical findings, so whichever critical
        finding is least confident is what held the case below threshold.
        """
        best: tuple[str, str, float] | None = None
        for spec, p in zip(FINDING_SCHEMA, self.probs):
            if not spec.critical:
                continue
            k = int(np.argmax(p))
            cand = (spec.name, spec.classes[k], float(p[k]))
            if best is None or cand[2] < best[2]:
                best = cand
        if best is None:  # schema has critical findings; defensive only
            raise RuntimeError("no critical findings in schema")
        return best


# ---------------------------------------------------------------------------
# VLM veto rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VetoRule:
    """A phrase pattern that contradicts a finding when the finding is firm.

    The rule fires only above `min_p`, because a hedge against a genuinely
    uncertain finding is not a contradiction -- it is appropriate hedging, and
    vetoing it would silence the VLM exactly where it adds most.
    """

    name: str
    pattern: str
    finding: str  # which structured finding the phrase must agree with
    contradicts: tuple[str, ...]  # predicted classes that make it a contradiction
    min_p: float = 0.70
    why: str = ""


# Written against the *predicted* class of the named finding. `et_present` is
# non-critical but still constrains VLM text: a released report that hedges
# "no clear enhancement" beside et_present = present (p = 0.97) is internally
# inconsistent regardless of which finding gates.
VETO_RULES: tuple[VetoRule, ...] = (
    VetoRule(
        name="negated-enhancement",
        pattern=r"\b(no|absent|without|lacking|lack of|not seen|no clear)\b[^.]{0,40}\benhanc",
        finding="et_present",
        contradicts=("present",),
        why="phrase denies enhancement while et_present is predicted present",
    ),
    VetoRule(
        name="asserted-enhancement",
        pattern=r"\b(avid|strong|intense|prominent|marked|significant)\b[^.]{0,30}\benhanc|\benhanc\w*\s+(vigorously|strongly|brightly)",
        finding="et_present",
        contradicts=("absent",),
        why="phrase asserts enhancement while et_present is predicted absent",
    ),
    VetoRule(
        name="negated-necrosis",
        pattern=r"\b(no|absent|without|lacking|lack of|no clear)\b[^.]{0,40}\bnecros",
        finding="ncr_present",
        contradicts=("present",),
        why="phrase denies necrosis while ncr_present is predicted present",
    ),
    VetoRule(
        name="asserted-necrosis",
        pattern=r"\b(necrotic|necrosis)\b",
        finding="ncr_present",
        contradicts=("absent",),
        why="phrase asserts necrosis while ncr_present is predicted absent",
    ),
    VetoRule(
        name="left-only",
        pattern=r"\b(left|left-sided|left hemispher\w*)\b",
        finding="laterality",
        contradicts=("right",),
        why="phrase assigns the mass to the left while laterality is right",
    ),
    VetoRule(
        name="right-only",
        pattern=r"\b(right|right-sided|right hemispher\w*)\b",
        finding="laterality",
        contradicts=("left",),
        why="phrase assigns the mass to the right while laterality is left",
    ),
    VetoRule(
        name="unilateral",
        pattern=r"\bunilateral\b|\bconfined to (one|a single) hemisphere\b",
        finding="laterality",
        contradicts=("bilateral",),
        why="phrase calls the mass unilateral while laterality is bilateral",
    ),
    VetoRule(
        name="contralateral",
        pattern=r"\bbilateral\b|\bcross(es|ing)? the midline\b|\bboth hemispheres\b",
        finding="laterality",
        contradicts=("left", "right"),
        why="phrase spreads the mass bilaterally while laterality is unilateral",
    ),
)


def vetoes_for(phrase: str, case: ReportCase) -> list[VetoRule]:
    """Rules this phrase triggers for this case (empty = phrase is safe)."""
    text = phrase.lower()
    preds = case.predicted
    hits: list[VetoRule] = []
    for rule in VETO_RULES:
        cls, p = preds[rule.finding]
        if cls in rule.contradicts and p >= rule.min_p and re.search(rule.pattern, text):
            hits.append(rule)
    return hits


def filter_phrases(
    phrases: list[str], case: ReportCase
) -> tuple[list[str], list[tuple[str, VetoRule]]]:
    """Split raw VLM phrases into (kept, rejected-with-reason).

    Rejection is recorded, never silent: a dropped phrase is exactly the kind
    of event an auditor wants to see, and it is the only evidence that the
    veto did anything.
    """
    kept: list[str] = []
    rejected: list[tuple[str, VetoRule]] = []
    seen: set[str] = set()
    for raw in phrases:
        if not isinstance(raw, str):
            continue
        phrase = " ".join(raw.split()).strip().strip('"').strip()
        if len(phrase) < 8 or len(phrase) > 300:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        hits = vetoes_for(phrase, case)
        if hits:
            rejected.append((phrase, hits[0]))
        else:
            kept.append(phrase)
    return kept, rejected


# ---------------------------------------------------------------------------
# Small shared helpers used by template.py and render.py
# ---------------------------------------------------------------------------


def band(p: float) -> str:
    """Coarse confidence band for prose hedging (not a calibrated interval)."""
    if p >= 0.9:
        return "high"
    if p >= 0.7:
        return "moderate"
    return "low"


def finding_evidence(name: str, detail: str) -> tuple[Evidence, ...]:
    return (Evidence(f"finding:{name}", detail),)


__all__ = [
    "Evidence",
    "ReportCase",
    "Sentence",
    "VetoRule",
    "VETO_RULES",
    "assert_grounded",
    "band",
    "filter_phrases",
    "finding_evidence",
    "vetoes_for",
]

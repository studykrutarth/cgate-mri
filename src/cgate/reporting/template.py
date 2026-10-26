"""Deterministic findings -> prose. The part of the report that never guesses.

Each finding gets a fixed phrasing keyed on (predicted class, confidence band),
so two cases with the same calibrated probabilities produce the same words.
That is the point: prose that varies run-to-run for identical input cannot be
reviewed against a ledger, because the ledger describes probabilities while
the prose was free to paraphrase.

Hedging follows the confidence band, not the class: "no necrotic core
(confidence 0.71)" and "no necrotic core (confidence 0.97)" are different
claims and must not read the same.
"""

from __future__ import annotations

from ..quality import ROUTER_FEATURES
from ..schema import CRITICAL_MASK, FINDING_SCHEMA
from .grounding import Evidence, ReportCase, Sentence, band, finding_evidence

# Sequence protocol sentence; BraTS pre-operative glioma protocol.
TECHNIQUE = (
    "Pre-operative multi-sequence MRI: axial T1, contrast-enhanced T1 (T1ce), "
    "T2 and FLAIR, co-registered, skull-stripped and resampled to 1 mm isotropic "
    "(BraTS preparation)."
)


def _finding_sentence(spec, cls: str, p: float) -> Sentence:
    """One finding, phrased from its predicted class and probability band."""
    name, b = spec.name, band(p)
    cited = f"{p:.2f}"

    if name == "et_present":
        text = (
            f"An enhancing tumour component is present (confidence {cited})."
            if cls == "present"
            else f"No enhancing tumour component exceeds the volume threshold (confidence {cited})."
        )
    elif name == "ncr_present":
        text = (
            f"The tumour contains a necrotic / non-enhancing core (confidence {cited})."
            if cls == "present"
            else f"No necrotic / non-enhancing core exceeds the volume threshold (confidence {cited})."
        )
    elif name == "laterality":
        text = {
            "left": f"The whole-tumour mass is predominantly left-sided (confidence {cited}).",
            "right": f"The whole-tumour mass is predominantly right-sided (confidence {cited}).",
            "bilateral": f"The whole-tumour mass is bilateral (confidence {cited}).",
        }[cls]
    elif name == "burden":
        text = (
            f"Whole-tumour burden is assessed as {cls} "
            f"(volume tercile fitted on the training split; confidence {cited})."
        )
    elif name == "et_fraction":
        text = (
            f"The enhancing fraction of the whole tumour is {cls} "
            f"(ET/WT tercile fitted on the training split; confidence {cited})."
        )
    else:  # schema gained a finding; phrase it generically rather than drop it
        text = f"{name} predicted as {cls} (confidence {cited})."

    # Low-band claims are visibly hedged in prose, not just in the number:
    # "confidence 0.55" is easy to skim past, "suggestive" is not.
    if b == "low":
        text += " This finding is low-confidence and is hedged accordingly."
    return Sentence(text, finding_evidence(name, f"{cls} p={cited}"))


def _quality_sentence(case: ReportCase) -> Sentence | None:
    """One sentence on how far acquisition quality sits from the cohort."""
    if not case.q_std:
        return None
    import math

    zs = [abs(float(z)) for z in case.q_std if math.isfinite(float(z))]
    if not zs:
        return None
    mean_abs = sum(zs) / len(zs)
    named = ", ".join(case.q_names) if case.q_names else "quality metrics"
    detail = f"mean|z|={mean_abs:.2f}"
    if mean_abs < 0.5:
        text = (
            f"Acquisition quality ({named}) is within the range of the "
            "training cohort."
        )
    elif mean_abs < 1.0:
        text = (
            f"Acquisition quality ({named}) is mildly outside the training "
            f"cohort range (mean |z| = {mean_abs:.2f}); interpret borderline "
            "findings with corresponding caution."
        )
    else:
        text = (
            f"Acquisition quality ({named}) is substantially outside the "
            f"training cohort range (mean |z| = {mean_abs:.2f}); this case is "
            "a known weak point of the system."
        )
    return Sentence(
        text, tuple(Evidence(f"iqm:{n}", detail) for n in (case.q_names or ROUTER_FEATURES))
    )


def _gate_sentences(case: ReportCase) -> list[Sentence]:
    out: list[Sentence] = []
    if case.accepted:
        out.append(
            Sentence(
                f"Automated release: case confidence {case.confidence:.2f} meets "
                f"the calibrated gate threshold {case.gate_threshold:.2f}.",
                (Evidence("gate", f"accepted c={case.confidence:.2f} t={case.gate_threshold:.2f}"),),
            )
        )
        return out

    name, cls, p = case.weakest_critical
    out.append(
        Sentence(
            f"DEFERRED TO RADIOLOGIST: case confidence {case.confidence:.2f} is "
            f"below the calibrated gate threshold {case.gate_threshold:.2f}.",
            (Evidence("gate", f"deferred c={case.confidence:.2f} t={case.gate_threshold:.2f}"),),
        )
    )
    out.append(
        Sentence(
            f"Lowest-confidence critical finding: {name} -> {cls} "
            f"(confidence {p:.2f}).",
            finding_evidence(name, f"weakest-critical {cls} p={p:.2f}"),
        )
    )
    return out


def findings_section(case: ReportCase) -> list[Sentence]:
    """FINDINGS: one sentence per structured finding, plus acquisition quality."""
    out: list[Sentence] = []
    # Critical findings first: they are what the gate triages on.
    ordered = [s for s, c in zip(FINDING_SCHEMA, CRITICAL_MASK) if c] + [
        s for s, c in zip(FINDING_SCHEMA, CRITICAL_MASK) if not c
    ]
    preds = case.predicted
    for spec in ordered:
        cls, p = preds[spec.name]
        out.append(_finding_sentence(spec, cls, p))
    q = _quality_sentence(case)
    if q is not None:
        out.append(q)
    return out


def impression_section(case: ReportCase) -> list[Sentence]:
    """IMPRESSION: gate outcome, then the findings that carry the diagnosis."""
    out = _gate_sentences(case)
    preds = case.predicted
    # Impression re-states only the critical (gating) findings; supporting
    # findings were already stated once in FINDINGS and are not repeated --
    # a report that repeats itself trains readers to skim.
    parts = [
        f"{name} {preds[name][0]} (p={preds[name][1]:.2f})"
        for name, crit in zip((s.name for s in FINDING_SCHEMA), CRITICAL_MASK)
        if crit
    ]
    detail = "; ".join(parts)
    ev = tuple(
        finding_evidence(s.name, f"{preds[s.name][0]} p={preds[s.name][1]:.2f}")[0]
        for s, c in zip(FINDING_SCHEMA, CRITICAL_MASK)
        if c
    )
    lead = (
        "Key findings: "
        if case.accepted
        else "For radiologist confirmation: "
    )
    out.append(Sentence(lead + detail + ".", ev))
    return out


def technique_sentence() -> Sentence:
    return Sentence(TECHNIQUE, (Evidence("meta", "brats-protocol"),))

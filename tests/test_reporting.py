"""Tests for the grounded report generator.

The properties tested here are the ones the module's own docstring claims:
every sentence is grounded, deferral says why, and the veto filter removes
contradicting VLM phrases. These are the contract -- prose quality is not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cgate.reporting import (
    Evidence,
    ReportCase,
    Sentence,
    assert_grounded,
    filter_phrases,
    render_report,
    vetoes_for,
)
from cgate.reporting.grounding import VETO_RULES
from cgate.reporting.template import findings_section, impression_section
from cgate.reporting.vlm import _parse_phrases, build_mosaic
from cgate.schema import CLASS_COUNTS, N_FINDINGS


def _probs(overrides: dict[str, str] | None = None, p_high: float = 0.9) -> list[np.ndarray]:
    """Build a valid fused probability list; default = everything high-confidence."""
    classes = {
        "et_present": ("absent", "present"),
        "ncr_present": ("absent", "present"),
        "laterality": ("left", "right", "bilateral"),
        "burden": ("small", "medium", "large"),
        "et_fraction": ("low", "mid", "high"),
    }
    predicted = {
        "et_present": "present",
        "ncr_present": "present",
        "laterality": "right",
        "burden": "medium",
        "et_fraction": "mid",
    }
    predicted.update(overrides or {})
    out = []
    for name, cls_tuple in classes.items():
        vec = np.full(len(cls_tuple), (1 - p_high) / (len(cls_tuple) - 1), dtype=float)
        vec[cls_tuple.index(predicted[name])] = p_high
        out.append(vec)
    return out


def _case(accepted: bool = True, p_high: float = 0.9, **overrides) -> ReportCase:
    probs = _probs(overrides, p_high=p_high)
    return ReportCase(
        patient_id="BraTS2021_00000",
        cohort="high",
        probs=probs,
        confidence=0.93 if accepted else 0.41,
        gate_threshold=0.6,
        accepted=accepted,
        iqms={"cjv": 0.4, "efc": 0.6},
        q_std=(-0.2, 0.3, -0.1, 0.4, 0.0, -0.3, 0.2),
        q_names=("cjv", "efc", "snr", "sharpness", "inu", "contrast", "entropy"),
    )


# ---------------------------------------------------------------------------
# Grounding contract
# ---------------------------------------------------------------------------


def test_every_sentence_is_grounded():
    for accepted in (True, False):
        sents = findings_section(_case(accepted=accepted)) + impression_section(
            _case(accepted=accepted)
        )
        assert_grounded(sents)
        assert all(s.evidence for s in sents)


def test_ungrounded_sentence_fails():
    with pytest.raises(ValueError):
        Sentence("floating claim with no source", ())


def test_unknown_evidence_source_fails():
    with pytest.raises(ValueError):
        Evidence("vibes", "felt right")


def test_deferred_report_names_weakest_critical_finding():
    # Build a case where every critical finding is firm except laterality,
    # which is the uniquely weakest -- that is the one a deferral must name.
    c = _case(accepted=False, p_high=0.9)
    lat = c.probs[2]  # laterality: left/right/bilateral
    lat[:] = np.array([0.55, 0.30, 0.15])  # argmax left, p=0.55 is weakest critical
    name, cls, p = c.weakest_critical
    assert name == "laterality"
    assert cls == "left"
    assert p == pytest.approx(0.55, abs=0.05)
    imp = impression_section(c)
    joined = " ".join(s.text for s in imp)
    assert "DEFERRED" in joined
    assert "laterality" in joined


def test_released_report_has_no_deferral_language():
    joined = " ".join(s.text for s in impression_section(_case(accepted=True)))
    assert "DEFERRED" not in joined
    assert "release" in joined.lower()


# ---------------------------------------------------------------------------
# VLM veto filter
# ---------------------------------------------------------------------------


def test_veto_kills_contradicting_phrase():
    c = _case()  # et_present=present (0.9), laterality=right (0.9)
    kept, rejected = filter_phrases(
        ["No evidence of enhancement in this mass.", "Heterogeneous signal with irregular margins."],
        c,
    )
    assert kept == ["Heterogeneous signal with irregular margins."]
    assert rejected and rejected[0][1].name == "negated-enhancement"


def test_veto_blocks_wrong_laterality():
    c = _case()
    hits = vetoes_for("A left frontal mass with surrounding edema.", c)
    assert any(h.name == "left-only" for h in hits)


def test_veto_spares_hedged_phrase_on_low_confidence_case():
    # et_present at 0.55 -> the rule's min_p is not met; hedging is allowed.
    c = _case(et_present="absent", p_high=0.55)
    kept, rejected = filter_phrases(["No definite enhancement seen."], c)
    assert kept and not rejected


def test_veto_rules_reference_real_findings():
    from cgate.schema import FINDING_NAMES

    for rule in VETO_RULES:
        assert rule.finding in FINDING_NAMES


def test_duplicate_and_malformed_phrases_are_dropped():
    c = _case()
    kept, _ = filter_phrases(
        ["Marginal heterogeneity with irregular infiltrative margins.",
         "Marginal heterogeneity with irregular infiltrative margins.",
         "", "short"],
        c,
    )
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_contains_status_banner_and_ledger():
    rep = render_report(_case(accepted=True))
    md = rep.markdown()
    assert "AUTOMATED RELEASE" in md
    for section in ("## Technique", "## Findings", "## Impression", "## Grounding ledger"):
        assert section in md
    # The ledger must have one table row per sentence -- the whole point is that
    # no sentence exists in prose without an evidence entry beside it.
    ledger_rows = [ln for ln in md.splitlines() if ln.startswith("| ") and ln[2].isdigit()]
    assert len(ledger_rows) == len(rep.sentences)


def test_deferred_markdown_carries_deferral_banner():
    md = render_report(_case(accepted=False)).markdown()
    assert "DEFERRED TO RADIOLOGIST" in md


def test_json_sidecar_round_trips():
    rep = render_report(_case(), vlm_phrases=["Mild surrounding edema."], vlm_model="test-model")
    payload = json.loads(rep.to_json())
    assert payload["released"] is True
    assert payload["vlm_phrases"] == ["Mild surrounding edema."]
    assert payload["vlm_model"] == "test-model"
    assert all(s["evidence"] for s in payload["sentences"])


def test_vlm_phrases_appear_with_vlm_evidence():
    rep = render_report(_case(), vlm_phrases=["Heterogeneous signal with irregular margins."])
    vlm_sents = [s for s in rep.sentences if any(e.source == "vlm" for e in s.evidence)]
    assert len(vlm_sents) == 1


def test_vetted_phrases_rendered_with_reason():
    rep = render_report(
        _case(), vlm_rejected=[("No enhancement present.", "phrase denies enhancement")]
    )
    md = rep.markdown()
    assert "~~No enhancement present.~~" in md
    assert "vetoed" in md


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_report_case_rejects_malformed_probs():
    bad = [np.full(n, 1.0 / n) for n in CLASS_COUNTS]
    bad[0] = np.array([0.3, 0.3])  # does not sum to 1
    with pytest.raises(ValueError):
        ReportCase("p", "high", bad, 0.9, 0.5, True)


def test_report_case_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        ReportCase("p", "high", _probs(), 1.7, 0.5, True)


def test_wrong_class_count_fails():
    with pytest.raises(ValueError):
        ReportCase("p", "high", _probs()[:-1], 0.9, 0.5, True)


# ---------------------------------------------------------------------------
# VLM helpers (no model download)
# ---------------------------------------------------------------------------


def test_parse_phrases_from_json_array():
    assert _parse_phrases('["First observation.", "Second one."]') == [
        "First observation.",
        "Second one.",
    ]


def test_parse_phrases_from_bullet_fallback():
    text = "Sure! Here you go:\n- A first observation here.\n- A second observation."
    assert _parse_phrases(text) == ["A first observation here.", "A second observation."]


def test_build_mosaic_shape_and_labels():
    from PIL import Image

    slices = np.random.rand(4, 4, 32, 32).astype(np.float32)
    img = build_mosaic(slices, slice_index=1)
    assert isinstance(img, Image.Image)
    # 2x2 panels (32px each) with a 4px gap, plus 22px label rows.
    assert img.size == (2 * 32 + 4, 2 * (32 + 22) + 4)


def test_template_phrases_cite_probabilities():
    c = _case(p_high=0.9)
    joined = " ".join(s.text for s in findings_section(c))
    assert "0.90" in joined


def test_n_findings_is_five():
    # The report template enumerates findings explicitly; if the schema grows,
    # template.py needs a phrase for the new one and this test says so.
    assert N_FINDINGS == 5

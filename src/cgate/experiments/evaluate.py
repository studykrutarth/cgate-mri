"""Metrics for one fitted system, overall and stratified by scan quality.

Reported together, because they answer different questions:

  accuracy        can it do the task at all
  ECE / Brier     does stated confidence match observed accuracy
  AURC / E-AURC   does confidence *rank* correct above incorrect (gate quality)
  risk @ coverage the operational number: "at 80% coverage, error is X"

The quality stratification is the point of the project, so every metric is also
reported separately on the high- and low-quality cohorts. An improvement that
shows up only in the pooled number, and vanishes within strata, is Simpson's
paradox rather than a result.
"""

from __future__ import annotations

import numpy as np

from ..calibration import (
    adaptive_ece,
    accuracy,
    brier,
    confidence_ece,
    ece,
    nll,
)
from ..pipeline import CaseTable, TrainedSystem, case_confidence, case_correct
from ..schema import FINDING_NAMES, N_FINDINGS
from ..selective import (
    ConfidenceGate,
    aurc,
    coverage_at_risk,
    excess_aurc,
    optimal_aurc,
    risk_at_coverage,
)

# Minimum cases before a stratum's metrics are reported. Below this an AURC is
# dominated by which single case happens to be least confident.
MIN_STRATUM_SIZE = 15


def _core_metrics(
    conf: np.ndarray, correct: np.ndarray, target_coverage: float
) -> dict:
    n = int(conf.size)
    if n == 0:
        return {"n_cases": 0}
    out = {
        "n_cases": n,
        "case_accuracy": float(correct.mean()),
        "full_risk": float((~correct).mean()),
        "aurc": aurc(conf, correct),
        "optimal_aurc": optimal_aurc(correct),
        "excess_aurc": excess_aurc(conf, correct),
        "confidence_ece": confidence_ece(conf, correct),
        f"risk_at_{int(target_coverage * 100)}pct_coverage": risk_at_coverage(
            conf, correct, target_coverage
        ),
        "coverage_at_5pct_risk": coverage_at_risk(conf, correct, 0.05),
        "coverage_at_10pct_risk": coverage_at_risk(conf, correct, 0.10),
    }
    return out


def evaluate_system(
    system: TrainedSystem,
    table: CaseTable,
    test_idx: np.ndarray,
    target_coverage: float = 0.8,
) -> dict:
    if table.labels is None:
        raise RuntimeError("table.assign_labels() must be called before evaluating")

    fused = system.predict(table, test_idx)
    labels = table.labels[test_idx]
    conf = case_confidence(fused)
    correct = case_correct(fused, labels)

    result: dict = {
        "overall": _core_metrics(conf, correct, target_coverage),
        "per_finding": {},
        "strata": {},
        # Kept for paired bootstrap comparisons between systems.
        "_confidence": conf,
        "_correct": correct,
        "_test_idx": np.asarray(test_idx),
    }

    # Gate at the threshold chosen on calibration data -- the honest operating
    # point, as opposed to re-optimising the threshold on test.
    op = system.gate.evaluate(conf, correct)
    result["gate"] = {
        "threshold": op.threshold,
        "coverage": op.coverage,
        "selective_risk": op.selective_risk,
        "full_risk": op.full_risk,
        "error_reduction": op.error_reduction,
        "n_deferred": op.n_deferred,
        "n_total": op.n_total,
        "summary": op.summary(),
    }

    for j in range(N_FINDINGS):
        p, y = fused[j], labels[:, j]
        result["per_finding"][FINDING_NAMES[j]] = {
            "accuracy": accuracy(p, y),
            "ece": ece(p, y),
            "adaptive_ece": adaptive_ece(p, y),
            "brier": brier(p, y),
            "nll": nll(p, y),
        }

    cohorts = table.cohorts[test_idx]
    for cohort in np.unique(cohorts):
        m = cohorts == cohort
        if int(m.sum()) < MIN_STRATUM_SIZE:
            result["strata"][str(cohort)] = {
                "n_cases": int(m.sum()),
                "skipped": f"fewer than {MIN_STRATUM_SIZE} cases",
            }
            continue
        result["strata"][str(cohort)] = _core_metrics(
            conf[m], correct[m], target_coverage
        )
    return result


def format_table(results: dict[str, dict], target_coverage: float = 0.8) -> str:
    """Render the ablation grid as a fixed-width table."""
    risk_key = f"risk_at_{int(target_coverage * 100)}pct_coverage"
    header = (
        f"{'key':<4} {'system':<46} {'acc':>6} {'ECE':>7} {'AURC':>8} "
        f"{'E-AURC':>8} {'risk@' + str(int(target_coverage * 100)) + '%':>9}"
    )
    lines = [header, "-" * len(header)]
    for key, res in results.items():
        o = res["overall"]
        label = res.get("spec", {}).get("label", key)
        lines.append(
            f"{key:<4} {label[:46]:<46} "
            f"{o['case_accuracy']:>6.3f} {o['confidence_ece']:>7.4f} "
            f"{o['aurc']:>8.4f} {o['excess_aurc']:>8.4f} {o[risk_key]:>9.3f}"
        )
    return "\n".join(lines)


def format_strata(results: dict[str, dict], target_coverage: float = 0.8) -> str:
    """Per-cohort view -- where the routing claim actually lives."""
    risk_key = f"risk_at_{int(target_coverage * 100)}pct_coverage"
    cohorts: list[str] = []
    for res in results.values():
        for c in res.get("strata", {}):
            if c not in cohorts:
                cohorts.append(c)

    out: list[str] = []
    for cohort in sorted(cohorts):
        header = (
            f"{'key':<4} {'system':<40} {'n':>4} {'acc':>6} {'AURC':>8} "
            f"{'E-AURC':>8} {'risk@' + str(int(target_coverage * 100)) + '%':>9}"
        )
        out.append(f"\ncohort = {cohort}")
        out.append(header)
        out.append("-" * len(header))
        for key, res in results.items():
            s = res.get("strata", {}).get(cohort)
            label = res.get("spec", {}).get("label", key)
            if not s or s.get("skipped") or s.get("n_cases", 0) == 0:
                reason = s.get("skipped", "no cases") if s else "no cases"
                out.append(f"{key:<4} {label[:40]:<40} {'-':>4}  ({reason})")
                continue
            out.append(
                f"{key:<4} {label[:40]:<40} {s['n_cases']:>4} "
                f"{s['case_accuracy']:>6.3f} {s['aurc']:>8.4f} "
                f"{s['excess_aurc']:>8.4f} {s[risk_key]:>9.3f}"
            )
    return "\n".join(out)

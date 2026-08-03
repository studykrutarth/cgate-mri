"""Selective prediction: does thresholding on confidence actually remove errors?

This is the evaluation that decides whether the gate works. A model can be
perfectly calibrated in aggregate (low ECE) and still be useless to gate on, if
its confidence does not *rank* correct predictions above incorrect ones. The
risk-coverage curve measures exactly that ranking quality, and AURC summarises
it in one number.

Definitions follow Geifman & El-Yaniv (NeurIPS 2017) and the E-AURC formulation
of Geifman et al. (ICLR 2019):

  coverage(k) = k / n                      after keeping the k most-confident cases
  risk(k)     = (# errors in top k) / k    the error rate among what we kept
  AURC        = mean over k of risk(k)
  E-AURC      = AURC - AURC of a perfect confidence ranking

E-AURC matters because raw AURC is bounded below by the model's own error rate:
an accurate model looks good on AURC even with worthless confidence. Subtracting
the optimal curve isolates the quality of the confidence ordering itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _prepare(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct).ravel().astype(bool)
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence {confidence.shape} and correct {correct.shape} must match"
        )
    if confidence.size and not np.all(np.isfinite(confidence)):
        raise ValueError("confidence contains non-finite values")
    return confidence, correct


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> dict[str, np.ndarray]:
    """Risk and coverage at every possible operating point.

    Cases are ranked by confidence, most confident first. Ties are broken by
    original order (stable sort); with continuous confidences ties are rare, but
    on a heavily-tied score the curve should be read as one of several valid
    orderings.
    """
    confidence, correct = _prepare(confidence, correct)
    n = confidence.size
    if n == 0:
        return {
            "coverage": np.zeros(0),
            "risk": np.zeros(0),
            "threshold": np.zeros(0),
        }

    order = np.argsort(-confidence, kind="stable")
    errors_sorted = (~correct[order]).astype(np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    return {
        "coverage": k / n,
        "risk": np.cumsum(errors_sorted) / k,
        "threshold": confidence[order],
    }


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    curve = risk_coverage_curve(confidence, correct)
    if curve["risk"].size == 0:
        return float("nan")
    return float(np.mean(curve["risk"]))


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of a confidence ranking that puts every correct case first."""
    correct = np.asarray(correct).ravel().astype(bool)
    n = correct.size
    if n == 0:
        return float("nan")
    n_correct = int(correct.sum())
    k = np.arange(1, n + 1, dtype=np.float64)
    risk = np.maximum(k - n_correct, 0.0) / k
    return float(np.mean(risk))


def excess_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """E-AURC = AURC - optimal AURC. Zero means perfect confidence ranking."""
    a = aurc(confidence, correct)
    o = optimal_aurc(correct)
    if not np.isfinite(a) or not np.isfinite(o):
        return float("nan")
    return float(a - o)


def risk_at_coverage(
    confidence: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    """Error rate among the most-confident `coverage` fraction of cases.

    This is the headline number for the gate: "at 80% coverage, error drops
    from X to Y".
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    curve = risk_coverage_curve(confidence, correct)
    n = curve["risk"].size
    if n == 0:
        return float("nan")
    k = int(np.ceil(coverage * n))
    k = max(1, min(k, n))
    return float(curve["risk"][k - 1])


def coverage_at_risk(
    confidence: np.ndarray, correct: np.ndarray, max_risk: float
) -> float:
    """Largest coverage whose selective risk stays at or below `max_risk`.

    Returns 0.0 if even the single most-confident case is wrong. The curve is
    not monotonic, so this deliberately takes the largest satisfying k rather
    than the first crossing.
    """
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError(f"max_risk must be in [0, 1], got {max_risk}")
    curve = risk_coverage_curve(confidence, correct)
    risk = curve["risk"]
    if risk.size == 0:
        return float("nan")
    ok = np.nonzero(risk <= max_risk + 1e-12)[0]
    if ok.size == 0:
        return 0.0
    return float(curve["coverage"][ok[-1]])


@dataclass
class GateOperatingPoint:
    """A concrete gate setting and what it buys."""

    threshold: float
    coverage: float
    selective_risk: float
    full_risk: float
    n_deferred: int
    n_total: int

    @property
    def error_reduction(self) -> float:
        """Relative drop in error rate among retained cases."""
        if self.full_risk <= 0:
            return 0.0
        return float((self.full_risk - self.selective_risk) / self.full_risk)

    def summary(self) -> str:
        return (
            f"threshold={self.threshold:.4f}  coverage={self.coverage:.1%}  "
            f"error {self.full_risk:.1%} -> {self.selective_risk:.1%} "
            f"({self.error_reduction:+.1%} relative)  "
            f"deferred {self.n_deferred}/{self.n_total}"
        )


class ConfidenceGate:
    """Threshold on calibrated confidence, chosen on a held-out split.

    Two ways to choose the operating point:
      * `fit_for_coverage` -- "we can afford to send 20% of cases to a radiologist"
      * `fit_for_risk`     -- "we need error among retained cases below 5%"

    The threshold is always chosen on validation data and then applied unchanged
    to test. Choosing it on test would make any reported gain circular.
    """

    def __init__(self) -> None:
        self.threshold_: float = 0.0
        self.criterion_: str = "none"

    def fit_for_coverage(
        self, confidence: np.ndarray, target_coverage: float
    ) -> ConfidenceGate:
        confidence = np.asarray(confidence, dtype=np.float64).ravel()
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
        if confidence.size == 0:
            raise ValueError("cannot fit a gate on zero cases")
        # Keep the top `target_coverage` fraction: the threshold is the
        # (1 - coverage) quantile of confidence.
        self.threshold_ = float(np.quantile(confidence, 1.0 - target_coverage))
        self.criterion_ = f"coverage>={target_coverage:.3f}"
        return self

    def fit_for_risk(
        self, confidence: np.ndarray, correct: np.ndarray, max_risk: float
    ) -> ConfidenceGate:
        confidence, correct = _prepare(confidence, correct)
        if confidence.size == 0:
            raise ValueError("cannot fit a gate on zero cases")
        curve = risk_coverage_curve(confidence, correct)
        ok = np.nonzero(curve["risk"] <= max_risk + 1e-12)[0]
        if ok.size == 0:
            # Cannot meet the target at any coverage: defer everything rather
            # than silently accepting a risk we promised not to.
            self.threshold_ = float(np.max(confidence)) + 1.0
        else:
            self.threshold_ = float(curve["threshold"][ok[-1]])
        self.criterion_ = f"risk<={max_risk:.3f}"
        return self

    def accept(self, confidence: np.ndarray) -> np.ndarray:
        """Boolean mask of cases the gate lets through."""
        return np.asarray(confidence, dtype=np.float64).ravel() >= self.threshold_

    def evaluate(
        self, confidence: np.ndarray, correct: np.ndarray
    ) -> GateOperatingPoint:
        confidence, correct = _prepare(confidence, correct)
        n = confidence.size
        keep = self.accept(confidence)
        n_keep = int(keep.sum())
        full_risk = float((~correct).mean()) if n else float("nan")
        sel_risk = float((~correct[keep]).mean()) if n_keep else 0.0
        return GateOperatingPoint(
            threshold=self.threshold_,
            coverage=(n_keep / n) if n else float("nan"),
            selective_risk=sel_risk,
            full_risk=full_risk,
            n_deferred=n - n_keep,
            n_total=n,
        )


def paired_bootstrap_aurc_delta(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap CI for AURC(a) - AURC(b) over the same cases.

    Resampling cases (not predictions) and keeping both systems' outputs paired
    is what makes the interval a statement about the *difference*, which is the
    claim being made in the ablation table. Without this, a small AURC gap on a
    few hundred cases is not distinguishable from noise.
    """
    conf_a, correct_a = _prepare(conf_a, correct_a)
    conf_b, correct_b = _prepare(conf_b, correct_b)
    n = conf_a.size
    if n == 0 or conf_b.size != n:
        raise ValueError("both systems must have predictions for the same cases")

    rng = np.random.default_rng(seed)
    observed = aurc(conf_a, correct_a) - aurc(conf_b, correct_b)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = aurc(conf_a[idx], correct_a[idx]) - aurc(conf_b[idx], correct_b[idx])

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        # Two-sided bootstrap p-value for H0: delta = 0.
        "p_two_sided": float(2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
    }

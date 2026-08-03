import numpy as np
import pytest

from cgate.calibration import (
    MultiFindingTemperatureScaler,
    TemperatureScaler,
    adaptive_ece,
    brier,
    confidence_ece,
    ece,
    nll,
)
from cgate.selective import (
    ConfidenceGate,
    aurc,
    coverage_at_risk,
    excess_aurc,
    optimal_aurc,
    paired_bootstrap_aurc_delta,
    risk_at_coverage,
    risk_coverage_curve,
)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def test_ece_is_zero_for_perfectly_calibrated_model():
    # 100 cases at stated confidence 0.7, of which exactly 70% are correct.
    n = 100
    probs = np.tile([0.3, 0.7], (n, 1))
    labels = np.array([1] * 70 + [0] * 30)
    assert ece(probs, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    n = 100
    probs = np.tile([0.01, 0.99], (n, 1))
    labels = np.array([1] * 50 + [0] * 50)  # only 50% correct at 99% confidence
    assert ece(probs, labels, n_bins=10) == pytest.approx(0.49, abs=1e-6)


def test_brier_and_nll_reward_the_truth():
    labels = np.array([0, 1])
    good = np.array([[0.9, 0.1], [0.1, 0.9]])
    bad = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert brier(good, labels) < brier(bad, labels)
    assert nll(good, labels) < nll(bad, labels)


def test_nll_is_finite_for_zero_probability():
    # A hard zero on the true class must not produce inf and poison training.
    assert np.isfinite(nll(np.array([[1.0, 0.0]]), np.array([1])))


def test_temperature_scaling_reduces_nll_of_overconfident_model():
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, size=n)
    # Confident but only ~75% accurate -> should be softened (T > 1).
    p = np.where(rng.random(n) < 0.75, labels, 1 - labels)
    probs = np.zeros((n, 2))
    probs[np.arange(n), p] = 0.97
    probs[np.arange(n), 1 - p] = 0.03

    scaler = TemperatureScaler().fit(probs, labels)
    assert scaler.temperature_ > 1.0
    assert nll(scaler.transform(probs), labels) < nll(probs, labels)
    assert ece(scaler.transform(probs), labels) < ece(probs, labels)


def test_temperature_scaling_preserves_argmax():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([1, 1, 1], size=50)
    labels = rng.integers(0, 3, size=50)
    out = TemperatureScaler().fit_transform(probs, labels)
    assert np.array_equal(np.argmax(probs, axis=1), np.argmax(out, axis=1))


def test_multifinding_scaler_fits_one_temperature_per_finding():
    from cgate.schema import FINDING_SCHEMA, N_FINDINGS

    rng = np.random.default_rng(2)
    n = 120
    probs = [rng.dirichlet(np.ones(s.n_classes), size=n) for s in FINDING_SCHEMA]
    labels = np.stack(
        [rng.integers(0, s.n_classes, size=n) for s in FINDING_SCHEMA], axis=1
    )
    scaler = MultiFindingTemperatureScaler().fit(probs, labels)
    assert len(scaler.temperatures) == N_FINDINGS
    out = scaler.transform(probs)
    for p in out:
        assert np.allclose(p.sum(axis=1), 1.0)


def test_confidence_ece_is_zero_for_calibrated_input():
    n = 200
    probs = np.tile([0.4, 0.6], (n, 1))
    labels = np.array([1] * 120 + [0] * 80)  # 60% correct at 60% confidence
    conf = probs.max(axis=1)
    correct = np.argmax(probs, axis=1) == labels
    assert confidence_ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_adaptive_ece_is_small_for_a_calibrated_spread_of_confidences():
    rng = np.random.default_rng(11)
    n = 4000
    conf = rng.uniform(0.5, 1.0, size=n)
    # Correct with probability exactly equal to the stated confidence.
    correct = rng.random(n) < conf
    probs = np.stack([1 - conf, conf], axis=1)
    labels = np.where(correct, 1, 0)
    assert adaptive_ece(probs, labels) < 0.03


def test_adaptive_ece_is_inflated_by_tied_confidences():
    """Documents a real limitation rather than papering over it.

    Every case has the same confidence, and the model is perfectly calibrated,
    but equal-mass binning splits the tie by original order -- which here is
    sorted by outcome -- so the metric reports large miscalibration.
    """
    n = 200
    probs = np.tile([0.4, 0.6], (n, 1))
    labels = np.array([1] * 120 + [0] * 80)
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-9)
    assert adaptive_ece(probs, labels) > 0.4


# --------------------------------------------------------------------------
# selective prediction
# --------------------------------------------------------------------------
def test_perfect_confidence_ranking_matches_optimal_aurc():
    correct = np.array([True] * 7 + [False] * 3)
    conf = np.where(correct, 0.9, 0.1)  # confidence perfectly separates
    assert aurc(conf, correct) == pytest.approx(optimal_aurc(correct))
    assert excess_aurc(conf, correct) == pytest.approx(0.0, abs=1e-12)


def test_worst_confidence_ranking_is_worse_than_optimal():
    correct = np.array([True] * 7 + [False] * 3)
    conf = np.where(correct, 0.1, 0.9)  # exactly inverted
    assert aurc(conf, correct) > optimal_aurc(correct)
    assert excess_aurc(conf, correct) > 0


def test_aurc_is_zero_when_everything_is_correct():
    correct = np.ones(20, dtype=bool)
    assert aurc(np.linspace(0, 1, 20), correct) == pytest.approx(0.0)


def test_risk_coverage_curve_shape_and_endpoint():
    rng = np.random.default_rng(3)
    n = 50
    correct = rng.random(n) < 0.7
    conf = rng.random(n)
    curve = risk_coverage_curve(conf, correct)
    assert curve["coverage"].shape == (n,)
    assert curve["coverage"][-1] == pytest.approx(1.0)
    # At full coverage the selective risk equals the overall error rate.
    assert curve["risk"][-1] == pytest.approx(float((~correct).mean()))


def test_risk_at_coverage_matches_manual_count():
    correct = np.array([True, True, False, True, False])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # Top 3 by confidence: True, True, False -> risk 1/3
    assert risk_at_coverage(conf, correct, 0.6) == pytest.approx(1 / 3)


def test_coverage_at_risk_returns_zero_when_unreachable():
    correct = np.zeros(10, dtype=bool)  # everything wrong
    conf = np.linspace(0, 1, 10)
    assert coverage_at_risk(conf, correct, 0.05) == 0.0


def test_gate_for_coverage_hits_requested_coverage():
    rng = np.random.default_rng(4)
    conf = rng.random(200)
    gate = ConfidenceGate().fit_for_coverage(conf, 0.8)
    assert gate.accept(conf).mean() == pytest.approx(0.8, abs=0.02)


def test_gate_reduces_error_when_confidence_is_informative():
    rng = np.random.default_rng(5)
    n = 400
    correct = rng.random(n) < 0.6
    # Confidence correlated with correctness.
    conf = np.clip(np.where(correct, 0.8, 0.4) + rng.normal(0, 0.1, n), 0, 1)
    gate = ConfidenceGate().fit_for_coverage(conf, 0.7)
    op = gate.evaluate(conf, correct)
    assert op.selective_risk < op.full_risk
    assert op.error_reduction > 0


def test_gate_for_risk_defers_everything_when_target_impossible():
    correct = np.zeros(20, dtype=bool)
    conf = np.linspace(0.1, 0.9, 20)
    gate = ConfidenceGate().fit_for_risk(conf, correct, 0.01)
    assert gate.accept(conf).sum() == 0


def test_paired_bootstrap_finds_no_difference_between_identical_systems():
    rng = np.random.default_rng(6)
    n = 150
    correct = rng.random(n) < 0.7
    conf = rng.random(n)
    stats = paired_bootstrap_aurc_delta(conf, conf, correct, correct, n_boot=200, seed=0)
    assert stats["delta"] == pytest.approx(0.0)
    assert stats["ci_low"] <= 0 <= stats["ci_high"]


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(7)
    n = 400
    correct = rng.random(n) < 0.6
    good = np.where(correct, 0.9, 0.2) + rng.normal(0, 0.02, n)
    useless = rng.random(n)
    stats = paired_bootstrap_aurc_delta(good, useless, correct, correct, n_boot=300, seed=0)
    assert stats["delta"] < 0  # negative favours the first system
    assert stats["ci_high"] < 0

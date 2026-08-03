"""Quality metrics, fusion, routing, splits, and one end-to-end run."""

import numpy as np
import pytest

from cgate.data.features import extract_views, view_dimensions
from cgate.data.splits import (
    assert_no_leakage,
    patient_level_split,
    stratified_patient_split,
)
from cgate.data.synthetic import SyntheticConfig, build_synthetic_dataset, synth_case
from cgate.fusion import (
    combine,
    confidence_weights,
    fuse_uniform,
    normalised_entropy,
    uniform_weights,
)
from cgate.quality import ROUTER_FEATURES, QualityStandardizer, compute_iqms, otsu_threshold
from cgate.routing import QualityRouter, RouterConfig
from cgate.schema import FINDING_SCHEMA, N_FINDINGS

SMALL = (24, 24, 16)


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------
def test_otsu_separates_two_modes():
    x = np.concatenate([np.full(500, 1.0), np.full(500, 9.0)])
    assert 1.0 < otsu_threshold(x) < 9.0


def test_iqms_flag_degenerate_background_on_skull_stripped_data():
    rng = np.random.default_rng(0)
    case = synth_case(rng, shape=SMALL, quality_level=0.9, skull_strip=True)
    iqms = compute_iqms(case.volumes[0].astype(np.float64), case.brain_mask)
    assert iqms["background_degenerate"] == 1.0
    # Air-based metrics must be NaN, never a fake number.
    assert np.isnan(iqms["snr_air"])
    assert np.isnan(iqms["fber"])
    # Skull-strip-safe metrics must still be finite.
    for name in ROUTER_FEATURES:
        assert np.isfinite(iqms[name]), name


def test_noise_lowers_measured_snr():
    base = np.zeros((20, 20, 12))
    base[4:16, 4:16, 2:10] = 1.0
    mask = base > 0
    rng = np.random.default_rng(1)
    clean = base + rng.normal(0, 0.005, base.shape) * mask
    noisy = base + rng.normal(0, 0.15, base.shape) * mask
    assert compute_iqms(noisy, mask)["snr"] < compute_iqms(clean, mask)["snr"]


def test_quality_level_tracks_measured_iqms():
    """Lower synthetic quality must move the IQMs, or routing has no signal."""
    rng = np.random.default_rng(2)
    good, bad = [], []
    for i in range(6):
        g = synth_case(rng, shape=SMALL, quality_level=0.95)
        b = synth_case(rng, shape=SMALL, quality_level=0.10)
        good.append(compute_iqms(g.volumes[3].astype(np.float64), g.brain_mask))
        bad.append(compute_iqms(b.volumes[3].astype(np.float64), b.brain_mask))
    # CJV rises with nonuniformity and motion; it is the headline degradation
    # metric in the BraTS-Africa comparison.
    assert np.mean([x["cjv"] for x in bad]) > np.mean([x["cjv"] for x in good])


def test_standardizer_uses_training_statistics_only():
    train = np.array([[0.0, 10.0], [2.0, 20.0]])
    std = QualityStandardizer().fit(train)
    out = std.transform(train)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
    # A shifted test set must NOT be re-centred to zero.
    shifted = std.transform(train + 100.0)
    assert np.all(shifted.mean(axis=0) > 1.0)


def test_standardizer_replaces_nan_with_training_mean():
    std = QualityStandardizer().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = std.transform(np.array([[np.nan, 3.0]]))
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# feature views
# --------------------------------------------------------------------------
def test_views_have_documented_dimensions():
    rng = np.random.default_rng(3)
    case = synth_case(rng, shape=SMALL)
    views = extract_views(case.volumes, case.brain_mask)
    expected = view_dimensions(case.volumes.shape[0])
    for name, dim in expected.items():
        assert views[name].shape == (dim,), name
        assert np.isfinite(views[name]).all(), name


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def _dummy_probs(n, rng):
    return [rng.dirichlet(np.ones(s.n_classes), size=n) for s in FINDING_SCHEMA]


def test_uniform_fusion_is_the_mean_of_experts():
    rng = np.random.default_rng(4)
    experts = [_dummy_probs(10, rng) for _ in range(3)]
    fused = fuse_uniform(experts)
    for j in range(N_FINDINGS):
        expected = np.mean([e[j] for e in experts], axis=0)
        assert np.allclose(fused[j], expected, atol=1e-9)


def test_normalised_entropy_bounds():
    certain = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert normalised_entropy(certain)[0] == pytest.approx(0.0, abs=1e-6)
    assert normalised_entropy(uniform)[0] == pytest.approx(1.0, abs=1e-6)


def test_confidence_weighting_favours_the_certain_expert():
    n = 5
    certain = [np.tile(np.eye(s.n_classes)[0], (n, 1)) * 0.98 + 0.02 / s.n_classes
               for s in FINDING_SCHEMA]
    unsure = [np.full((n, s.n_classes), 1.0 / s.n_classes) for s in FINDING_SCHEMA]
    w = confidence_weights([certain, unsure])
    assert np.all(w[:, :, 0] > w[:, :, 1])


def test_combine_rejects_badly_shaped_weights():
    rng = np.random.default_rng(5)
    experts = [_dummy_probs(4, rng) for _ in range(2)]
    with pytest.raises(ValueError, match="shape"):
        combine(experts, np.ones((4, N_FINDINGS, 3)) / 3)


def test_combine_rejects_unnormalised_weights():
    rng = np.random.default_rng(6)
    experts = [_dummy_probs(4, rng) for _ in range(2)]
    with pytest.raises(ValueError, match="sum to 1"):
        combine(experts, np.ones((4, N_FINDINGS, 2)))


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def test_untrained_router_is_exactly_uniform_fusion():
    """The B6-vs-B4 comparison depends on this."""
    rng = np.random.default_rng(7)
    n, n_experts = 12, 3
    experts = [_dummy_probs(n, rng) for _ in range(n_experts)]
    q = rng.normal(size=(n, len(ROUTER_FEATURES)))
    router = QualityRouter(n_experts, q.shape[1])
    assert np.allclose(router.weights(q), uniform_weights(n, n_experts))
    for a, b in zip(router.fuse(q, experts), fuse_uniform(experts)):
        assert np.allclose(a, b, atol=1e-9)


def test_router_learns_to_prefer_the_expert_that_is_right():
    """A router that cannot beat uniform on a rigged case is broken."""
    rng = np.random.default_rng(8)
    n = 300
    q = rng.normal(size=(n, 2))
    labels = np.zeros((n, N_FINDINGS), dtype=np.int64)

    # Expert 0 is correct when q[:,0] > 0, expert 1 otherwise. The signal is in
    # the quality feature, which is exactly what the router is meant to read.
    good = q[:, 0] > 0
    e0, e1 = [], []
    for spec in FINDING_SCHEMA:
        c = spec.n_classes
        p0 = np.full((n, c), 0.1 / (c - 1))
        p1 = np.full((n, c), 0.1 / (c - 1))
        p0[good, 0] = 0.9
        p0[~good, 0] = 0.1 / (c - 1)
        p0[~good, 1] = 0.9
        p1[~good, 0] = 0.9
        p1[good, 0] = 0.1 / (c - 1)
        p1[good, 1] = 0.9
        e0.append(p0 / p0.sum(axis=1, keepdims=True))
        e1.append(p1 / p1.sum(axis=1, keepdims=True))

    router = QualityRouter(2, 2, RouterConfig(epochs=300, lr=0.1, l2=1e-4))
    router.fit(q, [e0, e1], labels)

    w = router.weights(q)
    # On cases where expert 0 is right it should get the larger weight.
    assert w[good, 0, 0].mean() > w[good, 0, 1].mean()
    assert w[~good, 0, 1].mean() > w[~good, 0, 0].mean()


def test_router_reports_its_own_capacity():
    """A negative routing result must be distinguishable from overfitting."""
    rng = np.random.default_rng(20)
    n = 40  # a realistic calibration-split size
    q = rng.normal(size=(n, 7))
    experts = [_dummy_probs(n, rng) for _ in range(3)]
    labels = np.stack(
        [rng.integers(0, s.n_classes, size=n) for s in FINDING_SCHEMA], axis=1
    )

    router = QualityRouter(3, 7, RouterConfig(epochs=20))
    router.fit(q, experts, labels)
    # 5 findings x 3 experts x (7 features + bias)
    assert router.fit_log_["n_params"] == 5 * 3 * 8
    assert router.fit_log_["n_samples"] == n
    assert router.fit_log_["overparameterised"] is True


def test_shared_router_cuts_parameters_fivefold():
    rng = np.random.default_rng(21)
    n = 40
    q = rng.normal(size=(n, 7))
    experts = [_dummy_probs(n, rng) for _ in range(3)]
    labels = np.stack(
        [rng.integers(0, s.n_classes, size=n) for s in FINDING_SCHEMA], axis=1
    )

    shared = QualityRouter(3, 7, RouterConfig(epochs=20, per_finding=False))
    shared.fit(q, experts, labels)
    assert shared.fit_log_["n_params"] == 3 * 8
    assert shared.fit_log_["overparameterised"] is False
    # Shared weights still produce a valid, per-finding weight tensor.
    w = shared.weights(q)
    assert w.shape == (n, N_FINDINGS, 3)
    assert np.allclose(w.sum(axis=2), 1.0)


def test_shared_router_is_also_uniform_at_initialisation():
    rng = np.random.default_rng(22)
    n = 10
    q = rng.normal(size=(n, 5))
    router = QualityRouter(3, 5, RouterConfig(per_finding=False))
    assert np.allclose(router.weights(q), uniform_weights(n, 3))


def test_router_training_reduces_loss():
    rng = np.random.default_rng(9)
    n = 120
    q = rng.normal(size=(n, 3))
    experts = [_dummy_probs(n, rng) for _ in range(2)]
    labels = np.stack(
        [rng.integers(0, s.n_classes, size=n) for s in FINDING_SCHEMA], axis=1
    )
    router = QualityRouter(2, 3, RouterConfig(epochs=120))
    router.fit(q, experts, labels)
    assert router.history_[-1]["monitor"] <= router.history_[0]["monitor"]


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def test_patient_split_has_no_leakage():
    ids = np.array([f"P{i//4:03d}" for i in range(80)])  # 4 samples per patient
    splits = patient_level_split(ids, seed=0)
    assert_no_leakage(splits, ids)
    assert sum(len(v) for v in splits.values()) == len(ids)


def test_stratified_split_keeps_both_cohorts_everywhere():
    ids = np.array([f"P{i:03d}" for i in range(120)])
    cohorts = np.array(["low" if i % 3 == 0 else "high" for i in range(120)])
    splits = stratified_patient_split(ids, cohorts, seed=0)
    for name, idx in splits.items():
        assert set(np.unique(cohorts[idx])) == {"high", "low"}, name


def test_leakage_detector_actually_fires():
    ids = np.array(["A", "A", "B", "B"])
    bad = {"train": np.array([0, 2]), "test": np.array([1, 3])}
    with pytest.raises(ValueError, match="appears in both"):
        assert_no_leakage(bad, ids)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_full_pipeline_runs_and_produces_valid_output():
    from cgate.experiments.ablations import ABLATIONS, run_ablations
    from cgate.findings import FindingExtractor
    from cgate.pipeline import ingest_cases

    cfg = SyntheticConfig(shape=SMALL, n_cases=60, seed=0)
    table = ingest_cases(build_synthetic_dataset(cfg))
    assert table.n_cases == 60
    assert table.iqms.shape == (60, len(ROUTER_FEATURES))
    assert np.isfinite(table.iqms).all()

    splits = stratified_patient_split(table.patient_ids, table.cohorts, seed=0)
    extractor = FindingExtractor(min_voxels=20).fit_from_measurements(
        [table.measurements[i] for i in splits["train"]]
    )
    table.assign_labels(extractor)

    # Only the cheap rows, to keep the test suite usable.
    specs = tuple(s for s in ABLATIONS if s.key in ("B1", "B4", "B6"))
    results = run_ablations(table, splits, specs=specs, verbose=False)

    for key in ("B1", "B4", "B6"):
        o = results[key]["overall"]
        assert 0.0 <= o["case_accuracy"] <= 1.0
        assert 0.0 <= o["aurc"] <= 1.0
        assert o["excess_aurc"] >= -1e-9  # AURC can never beat the optimal curve
        assert results[key]["gate"]["n_total"] == len(splits["test"])

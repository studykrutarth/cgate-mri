import json
from pathlib import Path

import numpy as np
import pytest

from cgate.data.splits import assert_no_leakage, external_cohort_split


# --------------------------------------------------------------------------
# results serialisation
# --------------------------------------------------------------------------
def test_every_path_argument_survives_json_serialisation():
    """Regression: a Path CLI flag crashed the JSON dump after a full run.

    The dump happens last, so an unserialisable value throws away every result
    the run just computed. Any `type=Path` flag must round-trip.
    """
    from cgate.experiments.run import _to_jsonable, build_parser

    args = build_parser().parse_args(
        ["--out", "runs/x.json", "--figures", "runs/figs", "--root", "D:/data"]
    )
    payload = _to_jsonable(vars(args))
    restored = json.loads(json.dumps(payload))
    # Compare via Path so the assertion holds on both Windows and POSIX
    # separators rather than baking in one platform's convention.
    assert Path(restored["figures"]) == Path("runs/figs")
    assert Path(restored["out"]) == Path("runs/x.json")
    # No Path may survive into the payload, whatever the flag is called.
    assert not any(isinstance(v, Path) for v in payload.values())


def test_to_jsonable_handles_numpy_and_non_finite():
    from cgate.experiments.run import _to_jsonable

    out = _to_jsonable(
        {"a": np.float64(1.5), "b": np.arange(3), "c": float("nan"),
         "d": float("inf"), "_private": "dropped"}
    )
    assert out == {"a": 1.5, "b": [0, 1, 2], "c": None, "d": None}
    json.dumps(out)


# --------------------------------------------------------------------------
# external-cohort protocol
# --------------------------------------------------------------------------
def _cohorts(n_high=60, n_low=20):
    ids = np.array([f"P{i:03d}" for i in range(n_high + n_low)])
    cohorts = np.array(["high"] * n_high + ["low"] * n_low)
    return ids, cohorts


def test_external_split_puts_the_whole_cohort_in_test():
    ids, cohorts = _cohorts()
    splits = external_cohort_split(ids, cohorts, test_cohort="low", seed=0)
    assert set(cohorts[splits["test"]]) == {"low"}
    assert len(splits["test"]) == 20
    # ...and none of it leaks into development.
    for name in ("train", "calib"):
        assert set(cohorts[splits[name]]) == {"high"}


def test_external_split_uses_every_case_exactly_once():
    ids, cohorts = _cohorts()
    splits = external_cohort_split(ids, cohorts, test_cohort="low", seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == len(ids)
    seen = np.concatenate(list(splits.values()))
    assert len(np.unique(seen)) == len(ids)
    assert_no_leakage(splits, ids)


def test_external_split_always_leaves_a_calibration_set():
    """Temperature, router and gate threshold all depend on calib being non-empty."""
    ids, cohorts = _cohorts(n_high=4, n_low=10)
    splits = external_cohort_split(
        ids, cohorts, test_cohort="low", train_fraction=0.99, seed=0
    )
    assert len(splits["calib"]) >= 1
    assert len(splits["train"]) >= 1


def test_external_split_gives_far_more_test_cases_than_a_random_split():
    """The whole reason this protocol exists."""
    from cgate.data.splits import stratified_patient_split

    ids, cohorts = _cohorts(n_high=1251, n_low=146)  # BraTS 2021 vs BraTS-Africa
    ext = external_cohort_split(ids, cohorts, test_cohort="low", seed=0)
    strat = stratified_patient_split(ids, cohorts, seed=0)
    n_low_strat = int((cohorts[strat["test"]] == "low").sum())
    assert len(ext["test"]) == 146
    assert len(ext["test"]) > 4 * n_low_strat


def test_calib_target_fraction_lends_target_cases_to_calibration():
    ids, cohorts = _cohorts(n_high=60, n_low=40)
    splits = external_cohort_split(
        ids, cohorts, test_cohort="low", calib_target_fraction=0.25, seed=0
    )
    calib_cohorts = cohorts[splits["calib"]]
    assert "low" in set(calib_cohorts)
    assert "high" in set(calib_cohorts)
    assert int((calib_cohorts == "low").sum()) == 10  # 25% of 40


def test_calib_target_fraction_keeps_calib_and_test_disjoint():
    """Lent patients must leave test, or the reported numbers are circular."""
    ids, cohorts = _cohorts(n_high=60, n_low=40)
    splits = external_cohort_split(
        ids, cohorts, test_cohort="low", calib_target_fraction=0.25, seed=0
    )
    assert len(splits["test"]) == 30
    assert_no_leakage(splits, ids)
    assert not set(ids[splits["calib"]]) & set(ids[splits["test"]])
    total = sum(len(v) for v in splits.values())
    assert total == len(ids)


def test_calib_target_fraction_never_empties_the_test_set():
    ids, cohorts = _cohorts(n_high=20, n_low=5)
    splits = external_cohort_split(
        ids, cohorts, test_cohort="low", calib_target_fraction=0.99, seed=0
    )
    assert len(splits["test"]) >= 1


def test_calib_target_fraction_zero_matches_the_plain_external_split():
    ids, cohorts = _cohorts()
    a = external_cohort_split(ids, cohorts, test_cohort="low", seed=3)
    b = external_cohort_split(
        ids, cohorts, test_cohort="low", calib_target_fraction=0.0, seed=3
    )
    for name in ("train", "calib", "test"):
        assert np.array_equal(a[name], b[name])


def test_external_split_rejects_a_bad_calib_fraction():
    ids, cohorts = _cohorts()
    with pytest.raises(ValueError, match="calib_target_fraction"):
        external_cohort_split(
            ids, cohorts, test_cohort="low", calib_target_fraction=1.0
        )


def test_external_split_rejects_an_unknown_cohort():
    ids, cohorts = _cohorts()
    with pytest.raises(ValueError, match="no cases with cohort"):
        external_cohort_split(ids, cohorts, test_cohort="medium")


def test_external_split_rejects_holding_out_everything():
    ids = np.array(["A", "B", "C"])
    cohorts = np.array(["low", "low", "low"])
    with pytest.raises(ValueError, match="nothing left to train on"):
        external_cohort_split(ids, cohorts, test_cohort="low")


def test_external_split_keeps_patients_intact():
    # Four samples per patient, as with 2-D slices.
    ids = np.array([f"P{i // 4:03d}" for i in range(80)])
    cohorts = np.where(np.arange(80) < 40, "high", "low")
    splits = external_cohort_split(ids, cohorts, test_cohort="low", seed=1)
    assert_no_leakage(splits, ids)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _fake_results(rng, n=80):
    out = {}
    for key in ("B4", "B6"):
        correct = rng.random(n) < 0.6
        conf = np.clip(np.where(correct, 0.8, 0.5) + rng.normal(0, 0.08, n), 0.01, 0.99)
        out[key] = {
            "_confidence": conf,
            "_correct": correct,
            "_test_idx": np.arange(n),
            "spec": {"label": f"system {key}"},
        }
    return out


def test_risk_coverage_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from cgate.experiments.figures import risk_coverage_figure

    rng = np.random.default_rng(0)
    out = tmp_path / "rc.png"
    path = risk_coverage_figure(_fake_results(rng), out)
    assert path is not None
    assert out.exists() and out.stat().st_size > 1000


def test_reliability_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from cgate.experiments.figures import reliability_figure

    rng = np.random.default_rng(1)
    out = tmp_path / "rel.png"
    path = reliability_figure(_fake_results(rng), out)
    assert path is not None
    assert out.exists() and out.stat().st_size > 1000


def test_figures_return_none_rather_than_crashing_on_empty_input(tmp_path):
    pytest.importorskip("matplotlib")
    from cgate.experiments.figures import reliability_figure, risk_coverage_figure

    empty = {"B4": {"_confidence": np.zeros(0), "_correct": np.zeros(0, bool),
                    "_test_idx": np.zeros(0, int), "spec": {"label": "x"}}}
    assert risk_coverage_figure(empty, tmp_path / "a.png") is None
    assert reliability_figure(empty, tmp_path / "b.png") is None


def test_save_all_figures_writes_per_cohort_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from cgate.experiments.figures import save_all_figures

    rng = np.random.default_rng(2)
    results = _fake_results(rng, n=80)

    class _Table:
        cohorts = np.array(["high"] * 40 + ["low"] * 40)

    written = save_all_figures(results, tmp_path, _Table())
    names = {p.name for p in written}
    assert "risk_coverage_overall.png" in names
    assert "risk_coverage_high.png" in names
    assert "risk_coverage_low.png" in names
    assert "reliability.png" in names

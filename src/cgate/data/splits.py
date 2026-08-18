"""Patient-level splits with an explicit calibration partition.

Three splits, not two:

  train  -- fits the experts
  calib  -- fits everything post-hoc: temperature, the router, the gate threshold
  test   -- touched once, for reporting

The calibration split is not optional. Temperature scaling fitted on train
returns T ~ 1 because the model is already overfitted there; a gate threshold
chosen on test makes every reported coverage/risk number circular. Keeping a
dedicated split is what lets the headline claim ("at 80% coverage, error drops
from X to Y") mean anything.

Splitting is by patient, never by sample. With 2-D slice or 3-D patch training,
one patient contributes many samples; splitting those independently puts
neighbouring slices of the same tumour on both sides of the split, which inflates
accuracy dramatically and is the most common silent bug in this literature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.6, "calib": 0.2, "test": 0.2}


def _normalise_fractions(fractions: dict[str, float]) -> dict[str, float]:
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("split fractions must sum to something positive")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    return dict(fractions)


def patient_level_split(
    patient_ids: Sequence[str],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split sample indices so that no patient spans two splits.

    Returns a mapping split-name -> array of indices into `patient_ids`.
    """
    fractions = _normalise_fractions(fractions or DEFAULT_FRACTIONS)
    ids = np.asarray(patient_ids)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)

    names = list(fractions)
    cuts = np.cumsum([fractions[n] for n in names]) * len(perm)
    bounds = [0, *[int(round(c)) for c in cuts]]
    bounds[-1] = len(perm)

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        chosen = set(perm[bounds[i] : bounds[i + 1]].tolist())
        out[name] = np.nonzero(np.isin(ids, list(chosen)))[0] if chosen else np.zeros(0, dtype=int)
    assert_no_leakage(out, patient_ids)
    return out


def stratified_patient_split(
    patient_ids: Sequence[str],
    strata: Sequence,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Patient-level split that balances each stratum across splits.

    Used to keep the high- and low-quality cohorts proportionally represented,
    so a quality-stratified evaluation is not confounded by one split happening
    to receive most of the degraded scans.

    A patient whose samples carry conflicting strata is assigned to its first
    observed stratum, since the unit being split is the patient.
    """
    fractions = _normalise_fractions(fractions or DEFAULT_FRACTIONS)
    ids = np.asarray(patient_ids)
    strata = np.asarray(strata)
    if ids.shape[0] != strata.shape[0]:
        raise ValueError("patient_ids and strata must have the same length")

    patient_stratum: dict[str, object] = {}
    for pid, s in zip(ids, strata):
        patient_stratum.setdefault(pid, s)

    names = list(fractions)
    buckets: dict[str, list[str]] = {n: [] for n in names}
    rng = np.random.default_rng(seed)

    for stratum in np.unique(list(patient_stratum.values())):
        members = np.array(
            [p for p, s in patient_stratum.items() if s == stratum], dtype=object
        )
        perm = rng.permutation(members)
        cuts = np.cumsum([fractions[n] for n in names]) * len(perm)
        bounds = [0, *[int(round(c)) for c in cuts]]
        bounds[-1] = len(perm)
        for i, name in enumerate(names):
            buckets[name].extend(perm[bounds[i] : bounds[i + 1]].tolist())

    out = {
        name: np.nonzero(np.isin(ids, members))[0] if members else np.zeros(0, dtype=int)
        for name, members in buckets.items()
    }
    assert_no_leakage(out, patient_ids)
    return out


def external_cohort_split(
    patient_ids: Sequence[str],
    cohorts: Sequence,
    test_cohort: str,
    train_fraction: float = 0.75,
    seed: int = 0,
    calib_target_fraction: float = 0.0,
) -> dict[str, np.ndarray]:
    """Hold out an entire cohort as the test set.

    This is the protocol the project actually needs. Train and calibrate on the
    high-quality cohort (BraTS 2021, mostly 3 T), then test on *every* case of
    the degraded cohort (BraTS-Africa, 1.5 T).

    Two reasons it beats a random stratified split:

    1. Statistical power. BraTS-Africa has ~146 cases in total. A 20% test split
       leaves ~29, which cannot resolve the size of AURC difference this project
       is looking for -- the confidence interval swamps the effect. Using the
       whole cohort as test multiplies the test set fivefold at no cost.

    2. It is the real deployment question. "We trained where data is good; what
       happens when the scanner is worse?" A random split mixes degraded scans
       into training, which is precisely the confound that hides whether
       quality-aware routing transfers.

    The trade-off, which must be stated when reporting: train and test now
    differ in acquisition as well as quality, so a drop cannot be attributed to
    quality alone. Report the stratified protocol alongside it.

    `calib_target_fraction` moves that share of the target cohort's patients
    into calibration instead of test. It exists because of a measured failure:
    with the default of 0.0, everything post-hoc -- temperatures, the router,
    the gate threshold -- is fitted on high-quality cases only. The router then
    has to extrapolate its quality-to-weight mapping into a range it never
    observed, and on phantoms this made routing actively *worse* than uniform
    fusion under shift (AURC +0.06), with the gate collapsing to 56% coverage.

    Setting it to e.g. 0.15 mirrors what a deployment can realistically obtain:
    a small labelled sample from the new site. Patients moved into calibration
    are removed from test, so the two remain disjoint and the comparison stays
    honest.
    """
    ids = np.asarray(patient_ids)
    cohorts = np.asarray(cohorts)
    if ids.shape[0] != cohorts.shape[0]:
        raise ValueError("patient_ids and cohorts must have the same length")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 <= calib_target_fraction < 1.0:
        raise ValueError(
            f"calib_target_fraction must be in [0, 1), got {calib_target_fraction}"
        )

    is_test = cohorts == test_cohort
    if not np.any(is_test):
        raise ValueError(
            f"no cases with cohort {test_cohort!r}; found {sorted(set(map(str, cohorts)))}"
        )
    if np.all(is_test):
        raise ValueError(
            f"every case has cohort {test_cohort!r}; nothing left to train on"
        )

    rng = np.random.default_rng(seed)

    # Optionally lend a slice of the target cohort to calibration.
    target_ids = np.unique(ids[is_test])
    moved: set = set()
    if calib_target_fraction > 0.0:
        t_perm = rng.permutation(target_ids)
        n_move = int(round(calib_target_fraction * len(t_perm)))
        # Never empty the test set to feed calibration.
        n_move = min(n_move, max(len(t_perm) - 1, 0))
        moved = set(t_perm[:n_move].tolist())

    dev_ids = np.unique(ids[~is_test])
    perm = rng.permutation(dev_ids)
    cut = int(round(train_fraction * len(perm)))
    # Guarantee a non-empty calibration split: temperature, the router and the
    # gate threshold all depend on it.
    cut = min(max(cut, 1), len(perm) - 1)

    train_ids = set(perm[:cut].tolist())
    calib_ids = set(perm[cut:].tolist()) | moved

    in_calib = np.isin(ids, list(calib_ids)) if calib_ids else np.zeros(len(ids), bool)
    splits = {
        "train": np.nonzero(np.isin(ids, list(train_ids)))[0],
        "calib": np.nonzero(in_calib)[0],
        # Patients lent to calibration leave the test set, so the two stay
        # disjoint and the reported test numbers remain untouched by fitting.
        "test": np.nonzero(is_test & ~in_calib)[0],
    }
    assert_no_leakage(splits, patient_ids)
    return splits


def assert_no_leakage(
    splits: dict[str, np.ndarray], patient_ids: Sequence[str]
) -> None:
    """Raise if any patient appears in more than one split.

    Called automatically by both split functions. Cheap, and it turns the most
    damaging class of bug in this pipeline into an immediate failure.
    """
    ids = np.asarray(patient_ids)
    seen: dict[str, str] = {}
    for name, idx in splits.items():
        for pid in np.unique(ids[idx]):
            if pid in seen and seen[pid] != name:
                raise ValueError(
                    f"patient {pid!r} appears in both {seen[pid]!r} and {name!r} splits"
                )
            seen[pid] = name


def split_summary(
    splits: dict[str, np.ndarray], strata: Sequence | None = None
) -> dict[str, dict]:
    """Case counts (and per-stratum counts) for the run log."""
    out: dict[str, dict] = {}
    strata_arr = np.asarray(strata) if strata is not None else None
    for name, idx in splits.items():
        entry: dict = {"n_cases": int(len(idx))}
        if strata_arr is not None and len(idx):
            values, counts = np.unique(strata_arr[idx], return_counts=True)
            entry["strata"] = {str(v): int(c) for v, c in zip(values, counts)}
        out[name] = entry
    return out

import numpy as np
import pytest

from cgate.findings import (
    FindingExtractor,
    laterality_split,
    measure,
    region_volumes,
)
from cgate.schema import (
    CRITICAL_MASK,
    FINDING_SCHEMA,
    LABEL_ED,
    LABEL_ET,
    LABEL_NCR,
    N_FINDINGS,
    finding_index,
    spec_by_name,
    validate_probs,
)


def _blank(shape=(20, 20, 10)):
    return np.zeros(shape, dtype=np.int16)


def test_validate_probs_rejects_unnormalised():
    probs = [np.full((3, s.n_classes), 0.9) for s in FINDING_SCHEMA]
    with pytest.raises(ValueError, match="sums to"):
        validate_probs(probs)


def test_validate_probs_rejects_wrong_class_count():
    probs = [np.full((3, s.n_classes), 1.0 / s.n_classes) for s in FINDING_SCHEMA]
    probs[0] = np.full((3, 5), 0.2)
    with pytest.raises(ValueError, match="expects 2 classes"):
        validate_probs(probs)


def test_validate_probs_accepts_valid():
    probs = [np.full((4, s.n_classes), 1.0) / s.n_classes for s in FINDING_SCHEMA]
    validate_probs(probs)


def test_region_volumes_counts_composites():
    m = _blank()
    m[2:6, 2:6, 2:4] = LABEL_NCR  # 4*4*2 = 32
    m[8:10, 2:6, 2:4] = LABEL_ET  # 2*4*2 = 16
    m[12:16, 2:6, 2:4] = LABEL_ED  # 32
    v = region_volumes(m)
    assert v["NCR"] == 32
    assert v["ET"] == 16
    assert v["TC"] == 48
    assert v["WT"] == 80


@pytest.mark.parametrize(
    "lo_hi, left_low, expected",
    [
        ((0, 8), True, 0),  # mass at low indices, low index == left
        ((0, 8), False, 1),  # same mass, flipped orientation
        ((12, 20), True, 1),  # mass at high indices
    ],
)
def test_laterality_follows_orientation(lo_hi, left_low, expected):
    m = _blank()
    a, b = lo_hi
    m[a:b, 5:12, 3:7] = LABEL_ED
    assert laterality_split(m, left_is_low_index=left_low) == expected


def test_laterality_detects_bilateral():
    m = _blank()
    m[6:14, 5:12, 3:7] = LABEL_ED  # straddles the midplane at index 10
    assert laterality_split(m) == 2


def test_extractor_requires_fit_before_use():
    with pytest.raises(RuntimeError, match="fit"):
        FindingExtractor().extract(_blank())


def test_extractor_thresholds_small_findings_as_absent():
    ext = FindingExtractor(min_voxels=100)
    m = _blank()
    m[0:8, 0:8, 0:5] = LABEL_ED  # large whole tumour
    m[0:2, 0:2, 0:2] = LABEL_ET  # 8 voxels, below threshold
    ext.fit_from_measurements([measure(m)])
    labels = ext.labels_from_measurements([measure(m)])
    assert labels[0, finding_index("et_present")] == 0


def test_extractor_marks_large_findings_present():
    ext = FindingExtractor(min_voxels=10)
    m = _blank()
    m[0:8, 0:8, 0:5] = LABEL_ED
    m[0:4, 0:4, 0:4] = LABEL_ET  # 64 voxels, above threshold
    ext.fit_from_measurements([measure(m)])
    labels = ext.labels_from_measurements([measure(m)])
    assert labels[0, finding_index("et_present")] == 1


def test_terciles_split_burden_into_three_classes():
    masks = []
    for size in range(4, 20):
        m = _blank((40, 20, 10))
        m[0:size, 0:size, 0:4] = LABEL_ED
        masks.append(m)
    meas = [measure(m) for m in masks]
    ext = FindingExtractor(min_voxels=1).fit_from_measurements(meas)
    labels = ext.labels_from_measurements(meas)
    burden = labels[:, finding_index("burden")]
    assert set(np.unique(burden)) == {0, 1, 2}
    # Burden must be monotone in tumour size.
    assert np.all(np.diff(burden) >= 0)


def test_labels_have_expected_shape_and_range():
    m = _blank()
    m[0:8, 0:8, 0:5] = LABEL_ED
    meas = [measure(m)] * 5
    ext = FindingExtractor(min_voxels=1).fit_from_measurements(meas)
    labels = ext.labels_from_measurements(meas)
    assert labels.shape == (5, N_FINDINGS)
    for j, spec in enumerate(FINDING_SCHEMA):
        assert labels[:, j].max() < spec.n_classes


def test_empty_mask_does_not_crash():
    meas = [measure(_blank())]
    ext = FindingExtractor(min_voxels=1)
    with pytest.raises(ValueError, match="empty"):
        ext.fit_from_measurements(meas)


# --- F13: what gates a case, and where the NCR cut comes from ---------------


def test_the_critical_set_is_exactly_ncr_present_and_laterality():
    """Pin the critical set: it decides what a case is gated on.

    Nothing else in the suite constrains it, so demoting or promoting a finding
    used to be a silent one-word edit. `et_present` is deliberately absent --
    F13 measured it at 0.973 prevalence on real BraTS masks, and a near-constant
    finding contributes a near-constant term to a minimum that never selects it.
    """
    critical = {s.name for s in FINDING_SCHEMA if s.critical}
    assert critical == {"ncr_present", "laterality"}


def test_et_present_is_reported_but_does_not_gate():
    assert "et_present" in {s.name for s in FINDING_SCHEMA}
    assert not spec_by_name("et_present").critical
    assert not CRITICAL_MASK[finding_index("et_present")]


def _measurements_with_ncr(volumes):
    return [
        {
            "wt": 100_000.0,
            "tc": float(v),
            "et": 0.0,
            "ncr": float(v),
            "et_fraction": 0.0,
            "laterality": 0.0,
        }
        for v in volumes
    ]


def test_ncr_cut_is_the_training_median_so_the_finding_is_balanced():
    # A spread where a fixed 500-voxel threshold calls every case present.
    meas = _measurements_with_ncr(range(1_000, 21_000, 200))
    ext = FindingExtractor(min_voxels=500).fit_from_measurements(meas)

    assert ext.ncr_cut > 500, "an unfitted cut leaves the phantom-era constant"
    labels = ext.labels_from_measurements(meas)[:, finding_index("ncr_present")]
    assert 0.45 <= labels.mean() <= 0.55

    # The same data under the old fixed threshold is the degeneracy F13 found.
    fixed = FindingExtractor(min_voxels=500, burden_edges=(1.0, 2.0), et_fraction_edges=(0.1, 0.2))
    assert fixed.labels_from_measurements(meas)[:, finding_index("ncr_present")].mean() == 1.0


def test_ncr_cut_never_falls_below_the_annotation_jitter_floor():
    """On a cohort with genuinely tiny necrosis, balance must lose to the floor.

    Splitting at a median of 5 voxels would make `ncr_present` balanced by
    calling annotation jitter a finding -- a label that is wrong rather than
    merely uninformative.
    """
    ext = FindingExtractor(min_voxels=500).fit_from_measurements(
        _measurements_with_ncr([2, 3, 4, 5, 6, 7, 8])
    )
    assert ext.ncr_cut == 500.0


def test_ncr_cut_is_fitted_on_train_only_and_then_frozen():
    train = _measurements_with_ncr(range(1_000, 3_000, 100))
    ext = FindingExtractor(min_voxels=500).fit_from_measurements(train)
    cut = ext.ncr_cut

    # A test split an order of magnitude larger must not move the label
    # definition; labelling it goes through the frozen cut.
    held_out = _measurements_with_ncr(range(50_000, 70_000, 1_000))
    labels = ext.labels_from_measurements(held_out)[:, finding_index("ncr_present")]
    assert ext.ncr_cut == cut
    assert labels.mean() == 1.0, "held-out cases are all above the frozen cut"


def test_extract_and_labels_from_measurements_agree_on_the_fitted_cut():
    masks = []
    for size in range(6, 18):
        m = _blank((40, 20, 12))
        m[0:size, 0:size, 0:6] = LABEL_ED
        m[0:size, 0:2, 0:2] = LABEL_NCR
        masks.append(m)

    ext = FindingExtractor(min_voxels=1).fit(masks)
    from_masks = ext.extract_many(masks)[:, finding_index("ncr_present")]
    from_meas = ext.labels_from_measurements([measure(m) for m in masks])[
        :, finding_index("ncr_present")
    ]
    assert np.array_equal(from_masks, from_meas)

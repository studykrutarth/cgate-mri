import numpy as np
import pytest

from cgate.findings import (
    FindingExtractor,
    laterality_split,
    measure,
    region_volumes,
)
from cgate.schema import (
    FINDING_SCHEMA,
    LABEL_ED,
    LABEL_ET,
    LABEL_NCR,
    N_FINDINGS,
    finding_index,
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

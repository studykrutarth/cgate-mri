import numpy as np
import pytest

from cgate.data.slices import (
    SliceCache,
    SliceCacheConfig,
    select_slice_indices,
)


class FakeCase:
    def __init__(self, pid, volumes, mask=None):
        self.patient_id = pid
        self.volumes = volumes
        self.mask = mask


def _volumes(c=4, h=16, w=16, d=20, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros((c, h, w, d), dtype=np.float32)
    # Signal concentrated in the middle slices, as in a real head.
    for z in range(d):
        weight = 1.0 - abs(z - d / 2) / (d / 2)
        v[:, 4:12, 4:12, z] = max(weight, 0.0) * (1.0 + rng.random())
    return v


# --------------------------------------------------------------------------
# slice selection
# --------------------------------------------------------------------------
def test_geometric_selection_returns_requested_count_sorted_in_range():
    v = _volumes(d=20)
    cfg = SliceCacheConfig(n_slices=8, selection="geometric")
    idx = select_slice_indices(v, cfg)
    assert idx.shape == (8,)
    assert np.all(np.diff(idx) >= 0)
    assert idx.min() >= 0 and idx.max() < 20


def test_geometric_selection_stays_within_the_central_band():
    v = _volumes(d=100)
    cfg = SliceCacheConfig(n_slices=10, selection="geometric", central_fraction=0.6)
    idx = select_slice_indices(v, cfg)
    # Central 60% of 100 slices is [20, 79]; the empty poles must be avoided.
    assert idx.min() >= 20
    assert idx.max() <= 79


def test_brain_area_selection_prefers_slices_with_signal():
    v = np.zeros((2, 8, 8, 10), dtype=np.float32)
    v[:, 2:6, 2:6, 4:7] = 1.0  # only slices 4,5,6 carry anything
    cfg = SliceCacheConfig(n_slices=3, selection="brain_area")
    idx = select_slice_indices(v, cfg)
    assert set(idx.tolist()) == {4, 5, 6}


def test_selection_handles_volume_shorter_than_requested():
    v = _volumes(d=3)
    cfg = SliceCacheConfig(n_slices=8, selection="geometric")
    idx = select_slice_indices(v, cfg)
    assert idx.shape == (8,)
    assert idx.max() < 3


def test_unknown_selection_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown selection"):
        SliceCacheConfig(selection="largest_tumour")


def test_slice_selection_ignores_the_segmentation_mask():
    """The leakage guarantee, pinned down.

    Picking "the slice with the largest tumour" would read the mask, which does
    not exist at inference. Two cases with identical images but different masks
    must therefore select identical slices.
    """
    v = _volumes(seed=3)
    for strategy in ("geometric", "brain_area"):
        cfg = SliceCacheConfig(n_slices=5, selection=strategy)
        a = select_slice_indices(v, cfg)
        b = select_slice_indices(v, cfg)
        assert np.array_equal(a, b)
    # And the function accepts no mask argument at all.
    import inspect

    params = list(inspect.signature(select_slice_indices).parameters)
    assert params == ["volumes", "config"]


# --------------------------------------------------------------------------
# cache build / open
# --------------------------------------------------------------------------
def test_build_and_open_roundtrip(tmp_path):
    cases = [FakeCase(f"P{i:03d}", _volumes(seed=i)) for i in range(6)]
    cfg = SliceCacheConfig(n_slices=4)
    cache = SliceCache.build(iter(cases), tmp_path / "cache", cfg)

    assert len(cache) == 6
    assert cache.n_slices == 4
    assert cache.n_modalities == 4
    assert (tmp_path / "cache" / "index.json").exists()
    assert (tmp_path / "cache" / "cases" / "P000.npy").exists()
    assert len(list((tmp_path / "cache" / "cases").glob("*.npy"))) == 6

    reopened = SliceCache.open(tmp_path / "cache")
    assert reopened.patient_ids == cache.patient_ids
    assert np.allclose(reopened.get(0), cache.get(0))


def test_get_returns_expected_shape(tmp_path):
    cases = [FakeCase(f"P{i}", _volumes(c=4, h=16, w=16, seed=i)) for i in range(3)]
    cache = SliceCache.build(iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=5))
    stack = cache.get(0)
    assert stack.shape == (5, 4, 16, 16)
    assert np.isfinite(stack).all()


def test_rows_for_maps_patients_and_fails_loudly_on_miss(tmp_path):
    cases = [FakeCase(f"P{i}", _volumes(seed=i)) for i in range(4)]
    cache = SliceCache.build(iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=2))
    rows = cache.rows_for(["P2", "P0"])
    assert rows.tolist() == [2, 0]
    with pytest.raises(KeyError, match="not in the slice cache"):
        cache.rows_for(["P9"])


def test_mismatched_geometry_is_rejected(tmp_path):
    cases = [
        FakeCase("A", _volumes(h=16, w=16)),
        FakeCase("B", _volumes(h=24, w=24)),
    ]
    with pytest.raises(ValueError, match="does not match"):
        SliceCache.build(iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=2))


def test_normalisation_standardises_brain_voxels(tmp_path):
    cases = [FakeCase("A", _volumes(seed=1) * 500.0 + 100.0)]
    cache = SliceCache.build(
        iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=6, normalise=True)
    )
    stack = cache.get(0)
    brain = stack[:, 0][stack[:, 0] != 0]
    # float16 storage costs precision, so tolerances are loose on purpose.
    assert abs(float(brain.mean())) < 0.5
    assert 0.4 < float(brain.std()) < 2.5


def test_background_stays_zero_after_normalisation(tmp_path):
    """The network must still be able to tell brain from air."""
    cases = [FakeCase("A", _volumes(seed=2))]
    cache = SliceCache.build(
        iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=4, normalise=True)
    )
    stack = cache.get(0)
    # Corners were never populated by _volumes and must remain exactly zero.
    assert np.all(stack[:, :, 0, 0] == 0.0)


def test_open_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no slice cache"):
        SliceCache.open(tmp_path / "nope")


def test_build_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="no cases"):
        SliceCache.build(iter([]), tmp_path / "c", SliceCacheConfig())


def test_index_records_selected_slices_for_audit(tmp_path):
    cases = [FakeCase(f"P{i}", _volumes(seed=i)) for i in range(3)]
    cache = SliceCache.build(iter(cases), tmp_path / "c", SliceCacheConfig(n_slices=4))
    assert len(cache.index["slice_indices"]) == 3
    assert all(len(s) == 4 for s in cache.index["slice_indices"])
    assert cache.index["config"]["selection"] == "geometric"

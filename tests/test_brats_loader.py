"""Case discovery and cohort labelling on real BraTS directory layouts.

These are the two things that failed silently against an actual download rather
than raising: the distributed BraTS-Africa package nests a tumour-type level
between the dataset root and the cases, which both hid the cases from a
fixed-depth walk and hid the Africa marker from a parent-name cohort lookup.

Most of these build empty `*-seg.nii.gz` files. Discovery and cohort labelling
read names only, so they need no NIfTI content and no nibabel — which keeps the
layout tests fast and installable-anywhere. One test at the end reads a real
volume to check the two halves meet.
"""

from pathlib import Path

import numpy as np
import pytest

from cgate.data.brats import (
    cohort_for_case,
    find_case_dirs,
    infer_cohort,
    load_brats_roots,
)

# BraTS 2021: flat, one directory per case, underscore suffixes.
BRATS2021_SUFFIXES = ("t1", "t1ce", "t2", "flair", "seg")
# BraTS-Africa / 2023: nested under tumour type, hyphen suffixes.
AFRICA_SUFFIXES = ("t1n", "t1c", "t2w", "t2f", "seg")


SHAPE = (8, 9, 10)
# LAS. The loader reads the left-right direction from the affine rather than
# assuming it: guessing mirrors every laterality label, and because the classes
# stay balanced that stays invisible in aggregate accuracy.
LAS_AFFINE = np.diag([-1.0, 1.0, 1.0, 1.0])


def _make_case(case_dir: Path, suffixes, separator: str) -> Path:
    """Empty files: enough for discovery and cohort labelling, which read names."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (case_dir / f"{case_dir.name}{separator}{suffix}.nii.gz").touch()
    return case_dir


def _write_case(
    case_dir: Path, suffixes, separator: str, seed: int = 0, affine=None
) -> Path:
    """Real NIfTI volumes, for anything that goes on to load the case."""
    nib = pytest.importorskip("nibabel")
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = LAS_AFFINE if affine is None else affine
    for suffix in suffixes:
        if suffix == "seg":
            data = np.zeros(SHAPE, dtype=np.int16)
            data[2:5, 2:5, 2:5] = 1
        else:
            data = rng.random(SHAPE).astype(np.float32)
        nib.save(
            nib.Nifti1Image(data, affine),
            case_dir / f"{case_dir.name}{separator}{suffix}.nii.gz",
        )
    return case_dir


def _real_africa_tree(root: Path) -> Path:
    base = root / "PKG - BraTS-Africa" / "BraTS-Africa" / "95_Glioma"
    for i, n in enumerate((2, 7, 9)):
        _write_case(base / f"BraTS-SSA-{n:05d}-000", AFRICA_SUFFIXES, "-", seed=i)
    return base


def _real_brats2021_tree(root: Path) -> Path:
    base = root / "BraTS2021_Training_Data"
    for n in range(3):
        _write_case(base / f"BraTS2021_{n:05d}", BRATS2021_SUFFIXES, "_", seed=100 + n)
    return base


def _africa_tree(root: Path) -> Path:
    """PKG - BraTS-Africa/BraTS-Africa/<type>/BraTS-SSA-XXXXX-000/ — as distributed."""
    base = root / "PKG - BraTS-Africa" / "BraTS-Africa"
    for group, ids in (("95_Glioma", (2, 7)), ("51_OtherNeoplasms", (9,))):
        for n in ids:
            _make_case(base / group / f"BraTS-SSA-{n:05d}-000", AFRICA_SUFFIXES, "-")
    return base


def _brats2021_tree(root: Path) -> Path:
    base = root / "BraTS2021_Training_Data"
    for n in range(3):
        _make_case(base / f"BraTS2021_{n:05d}", BRATS2021_SUFFIXES, "_")
    return base


def test_finds_cases_nested_below_a_tumour_type_directory(tmp_path):
    base = _africa_tree(tmp_path)
    found = find_case_dirs(base)
    assert len(found) == 3
    assert {d.name for d in found} == {
        "BraTS-SSA-00002-000",
        "BraTS-SSA-00007-000",
        "BraTS-SSA-00009-000",
    }


def test_finds_cases_in_a_flat_layout(tmp_path):
    found = find_case_dirs(_brats2021_tree(tmp_path))
    assert len(found) == 3


def test_a_case_directory_is_never_descended_into(tmp_path):
    """A stray subdirectory inside a case must not become a second case."""
    base = _brats2021_tree(tmp_path)
    nested = base / "BraTS2021_00000" / "derivatives"
    _make_case(nested, BRATS2021_SUFFIXES, "_")
    found = find_case_dirs(base)
    assert len(found) == 3
    assert nested not in found


def test_root_that_is_itself_a_case(tmp_path):
    case = _make_case(tmp_path / "BraTS2021_00000", BRATS2021_SUFFIXES, "_")
    assert find_case_dirs(case) == [case]


def test_search_depth_is_bounded(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
    _make_case(deep / "BraTS2021_00000", BRATS2021_SUFFIXES, "_")
    assert find_case_dirs(tmp_path, max_depth=3) == []
    assert len(find_case_dirs(tmp_path, max_depth=10)) == 1


def test_africa_cases_are_labelled_low_despite_the_tumour_type_level(tmp_path):
    """The regression: the case's parent is '95_Glioma', which carries no marker.

    Reading the cohort from that one name tagged all 146 real cases 'high',
    which makes --protocol external hold out the wrong cohort.
    """
    base = _africa_tree(tmp_path)
    for case_dir in find_case_dirs(base):
        assert cohort_for_case(case_dir, base) == "low"


def test_africa_cases_are_low_even_from_an_unmarked_root(tmp_path):
    """The case name carries the marker, so unpacking location cannot break it."""
    base = _africa_tree(tmp_path)
    neutral = base / "95_Glioma"
    for case_dir in find_case_dirs(neutral):
        assert cohort_for_case(case_dir, neutral) == "low"


def test_brats2021_cases_are_labelled_high(tmp_path):
    base = _brats2021_tree(tmp_path)
    for case_dir in find_case_dirs(base):
        assert cohort_for_case(case_dir, base) == "high"


def test_ancestors_above_the_root_do_not_decide_the_cohort(tmp_path):
    """Bounded at the root: an unrelated parent must not relabel a whole study."""
    base = _brats2021_tree(tmp_path / "africa_project")
    for case_dir in find_case_dirs(base):
        assert cohort_for_case(case_dir, base) == "high"


def test_ssa_is_matched_with_boundaries(tmp_path):
    """'assay' contains 'ssa' — a substring test would reclassify the dataset."""
    base = _brats2021_tree(tmp_path / "assay")
    for case_dir in find_case_dirs(base):
        assert cohort_for_case(case_dir, base) == "high"
    assert infer_cohort(Path("assay")) == "high"
    assert infer_cohort(Path("BraTS-SSA-00002-000")) == "low"
    assert infer_cohort(Path("BraTS-Africa")) == "low"
    assert infer_cohort(Path("Sub-Saharan")) == "low"


def test_several_roots_on_different_paths_are_loaded_together(tmp_path):
    """BraTS 2021 and BraTS-Africa are separate downloads, often on separate drives.

    Requiring one shared parent would mean copying tens of gigabytes purely to
    satisfy the loader.
    """
    source = _real_brats2021_tree(tmp_path / "drive_g")
    target = _real_africa_tree(tmp_path / "drive_c")

    cases = list(load_brats_roots([source, target]))
    assert len(cases) == 6
    cohorts = [c.cohort for c in cases]
    assert cohorts.count("high") == 3
    assert cohorts.count("low") == 3


def test_explicit_cohorts_override_inference(tmp_path):
    source = _real_brats2021_tree(tmp_path / "one")
    target = _real_africa_tree(tmp_path / "two")
    cases = list(load_brats_roots([source, target], cohorts=["low", "high"]))
    assert {c.cohort for c in cases[:3]} == {"low"}
    assert {c.cohort for c in cases[3:]} == {"high"}


def test_cohort_count_must_match_root_count(tmp_path):
    base = _brats2021_tree(tmp_path)
    with pytest.raises(ValueError, match="one per root"):
        list(load_brats_roots([base], cohorts=["high", "low"]))


def test_overlapping_roots_are_rejected(tmp_path):
    """Duplicated patients survive assert_no_leakage — both copies share an id."""
    base = _real_brats2021_tree(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        list(load_brats_roots([base, base / "BraTS2021_00000"]))


def test_empty_root_says_so(tmp_path):
    (tmp_path / "nothing").mkdir()
    with pytest.raises(FileNotFoundError, match="no case directories"):
        list(load_brats_roots([tmp_path / "nothing"]))


def test_left_is_low_index_matches_the_affine_arithmetic():
    """Derived from the affine rather than asserted, because a mirrored label
    keeps the classes balanced and moves no metric.

    An axcode names the direction an axis increases *toward*. World space is
    RAS+, so +x is anatomical right: code "R" puts the left hemisphere at the
    low indices. Reading it as "L" mirrors every laterality label, which is
    what the code did from the start -- and what this test now prevents by
    computing the answer instead of restating the convention.
    """
    from cgate.data.brats import left_is_low_index

    for affine in (np.eye(4), np.diag([-1.0, 1.0, 1.0, 1.0])):
        low = (affine @ np.array([0.0, 0, 0, 1]))[0]
        high = (affine @ np.array([10.0, 0, 0, 1]))[0]
        # Smaller world x is further left, so left is the low index when world
        # x increases with the index.
        assert left_is_low_index(affine) is bool(low < high)


# -- orientation: the cohorts ship in opposite storage orders ----------------


RAS_AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])


def _case_with_affine(case_dir: Path, affine, tumour_side: str) -> Path:
    """A case whose tumour sits on a given ANATOMICAL side.

    The voxel index of that tumour depends on the storage order, which is the
    whole point: the same anatomy lands at opposite indices under LPS and RAS.
    """
    nib = pytest.importorskip("nibabel")
    case_dir.mkdir(parents=True, exist_ok=True)
    # x increases toward L under an LPS affine and toward R under RAS.
    x_is_left = affine[0, 0] < 0
    high_index_is_left = x_is_left
    put_high = (tumour_side == "left") == high_index_is_left

    seg = np.zeros(SHAPE, dtype=np.int16)
    lo, hi = 0, SHAPE[0] // 2
    if put_high:
        lo, hi = SHAPE[0] // 2, SHAPE[0]
    seg[lo:hi, 2:5, 2:5] = 1

    for suffix in BRATS2021_SUFFIXES:
        data = seg if suffix == "seg" else seg.astype(np.float32) + 0.5
        nib.save(
            nib.Nifti1Image(data, affine),
            case_dir / f"{case_dir.name}_{suffix}.nii.gz",
        )
    return case_dir


def test_volumes_are_reoriented_to_a_common_frame(tmp_path):
    """BraTS 2021 ships LPS, BraTS-Africa mostly RAS.

    Reading the affine keeps both cohorts' *labels* right, which is what makes
    this dangerous: nothing downstream disagrees while the cached slices from
    one cohort are mirrored against the other. A network trained on one then has
    its left-right mapping inverted on the other.
    """
    from cgate.data.brats import load_case

    lps = _case_with_affine(tmp_path / "BraTS2021_00000", LAS_AFFINE, "left")
    ras = _case_with_affine(tmp_path / "BraTS2021_00001", RAS_AFFINE, "left")

    a, b = load_case(lps), load_case(ras)

    # Same anatomy, so the tumour must occupy the same voxel half in both.
    half = SHAPE[0] // 2
    a_high = a.mask[half:].sum() > a.mask[:half].sum()
    b_high = b.mask[half:].sum() > b.mask[:half].sum()
    assert a_high == b_high, "cohorts still stored in opposite left-right order"

    # And the images must be mirrored the same way as the masks, or the network
    # sees one cohort flipped against its own labels.
    a_img_high = a.volumes[0, half:].sum() > a.volumes[0, :half].sum()
    assert a_img_high == a_high
    assert a.truth["left_is_low_index"] == b.truth["left_is_low_index"]


def test_laterality_label_survives_reorientation(tmp_path):
    """The label must still describe anatomy, not storage order."""
    from cgate.data.brats import load_case
    from cgate.findings import measure

    left_lps = load_case(_case_with_affine(tmp_path / "a_BraTS2021_00000", LAS_AFFINE, "left"))
    left_ras = load_case(_case_with_affine(tmp_path / "b_BraTS2021_00001", RAS_AFFINE, "left"))
    right_ras = load_case(_case_with_affine(tmp_path / "c_BraTS2021_00002", RAS_AFFINE, "right"))

    m_ll = measure(left_lps.mask, left_is_low_index=left_lps.truth["left_is_low_index"])
    m_lr = measure(left_ras.mask, left_is_low_index=left_ras.truth["left_is_low_index"])
    m_rr = measure(right_ras.mask, left_is_low_index=right_ras.truth["left_is_low_index"])

    assert m_ll["laterality"] == m_lr["laterality"], "same anatomy, different label"
    assert m_rr["laterality"] != m_lr["laterality"], "opposite anatomy, same label"


# -- subsampling a large cohort ---------------------------------------------


def test_subsample_is_reproducible_and_sorted():
    from cgate.data.brats import subsample

    dirs = [Path(f"case-{i:03d}") for i in range(50)]
    a = subsample(dirs, 10, seed=0)
    assert len(a) == 10
    assert a == subsample(dirs, 10, seed=0), "not reproducible"
    assert a == sorted(a), "order depends on the draw"
    assert len(set(a)) == 10, "sampled with replacement"


def test_subsample_is_not_a_prefix():
    """BraTS ids are grouped by contributing institution.

    Taking the first N would select a subset of sites, and site is exactly what
    the quality cohorts are meant to vary.
    """
    from cgate.data.brats import subsample

    dirs = [Path(f"case-{i:03d}") for i in range(200)]
    assert subsample(dirs, 20, seed=0) != dirs[:20]


def test_subsample_passes_through_when_not_limiting():
    from cgate.data.brats import subsample

    dirs = [Path(f"case-{i:03d}") for i in range(5)]
    assert subsample(dirs, None) is dirs
    assert subsample(dirs, 99) is dirs


def test_subsample_rejects_a_nonsense_limit():
    from cgate.data.brats import subsample

    with pytest.raises(ValueError, match="max_cases must be >= 1"):
        subsample([Path("a"), Path("b")], 0)


def test_max_cases_limits_what_is_loaded(tmp_path):
    base = _real_brats2021_tree(tmp_path)
    assert len(list(load_brats_roots([base], max_cases=2))) == 2
    assert len(list(load_brats_roots([base]))) == 3


# -- the 2021 / 2023 label renumbering --------------------------------------


def test_modern_enhancing_tumour_label_is_canonicalised():
    """BraTS 2023 renumbered enhancing tumour from 4 to 3.

    Everything downstream is written against LABEL_ET = 4, so an unconverted
    2023 mask reports no enhancing tumour at all.
    """
    from cgate.data.brats import canonicalise_labels

    mask = np.array([0, 1, 2, 3], dtype=np.int16)
    out = canonicalise_labels(mask)
    assert out.tolist() == [0, 1, 2, 4]
    assert mask.tolist() == [0, 1, 2, 3], "input was mutated"


def test_legacy_labels_pass_through_untouched():
    from cgate.data.brats import canonicalise_labels

    mask = np.array([0, 1, 2, 4], dtype=np.int16)
    assert canonicalise_labels(mask).tolist() == [0, 1, 2, 4]


def test_a_mask_using_both_conventions_is_rejected():
    """Neither reading is safe, so guessing is worse than stopping."""
    from cgate.data.brats import canonicalise_labels

    with pytest.raises(ValueError, match="both label 3 and label 4"):
        canonicalise_labels(np.array([1, 2, 3, 4], dtype=np.int16), context="case-x")


def test_a_mask_without_enhancing_tumour_is_left_alone():
    from cgate.data.brats import canonicalise_labels

    mask = np.array([0, 1, 2], dtype=np.int16)
    assert canonicalise_labels(mask).tolist() == [0, 1, 2]


def test_enhancing_tumour_survives_loading_a_2023_style_case(tmp_path):
    """End to end: a 2023-format case must report ET through region_volumes."""
    from cgate.data.brats import load_case
    from cgate.findings import region_volumes

    nib = pytest.importorskip("nibabel")
    case_dir = tmp_path / "BraTS-SSA-00002-000"
    case_dir.mkdir()
    seg = np.zeros(SHAPE, dtype=np.int16)
    seg[1:3, 1:3, 1:3] = 1
    seg[3:5, 1:3, 1:3] = 2
    seg[5:7, 1:3, 1:3] = 3  # enhancing tumour, 2023 numbering
    for suffix in AFRICA_SUFFIXES:
        data = seg if suffix == "seg" else np.ones(SHAPE, dtype=np.float32)
        nib.save(
            nib.Nifti1Image(data, LAS_AFFINE),
            case_dir / f"{case_dir.name}-{suffix}.nii.gz",
        )

    case = load_case(case_dir)
    vols = region_volumes(case.mask)
    assert vols["ET"] == 8, "enhancing tumour was lost in translation"
    assert vols["WT"] == 24


# -- links, which the walk follows and the ancestor scan used to trip over ----


def _link_dir(link: Path, target: Path) -> None:
    """A directory link, however this platform makes one, or skip the test."""
    import os
    import subprocess

    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name != "nt":
        pytest.skip("cannot create a directory link here")
    # Junctions need no privilege; symlinks on Windows generally do.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    if result.returncode != 0 or not link.exists():
        pytest.skip("cannot create a directory link here")


def test_a_mirrored_link_does_not_duplicate_cases(tmp_path):
    """`is_dir()` follows links, so one case was reachable by several paths.

    The duplicates arrive as repeated patient ids and land in different splits —
    the one leak `assert_no_leakage` cannot see, because by then they are just
    two rows that happen to be the same patient.
    """
    study = tmp_path / "study"
    _brats2021_tree(study)
    _link_dir(study / "mirror", study / "BraTS2021_Training_Data")

    found = find_case_dirs(study)
    assert len(found) == 3, [str(d) for d in found]


def test_a_self_referential_link_terminates(tmp_path):
    study = tmp_path / "loop"
    _brats2021_tree(study)
    _link_dir(study / "self", study)

    assert len(find_case_dirs(study)) == 3


def test_a_linked_cohort_directory_still_decides_the_cohort(tmp_path):
    """The ancestor scan exists for cohorts whose marker is only a directory.

    Resolving both paths before `relative_to` put root and case on different
    branches whenever a link sat between them: the scan raised, was swallowed,
    and every case silently fell back to "high".
    """
    elsewhere = tmp_path / "elsewhere" / "scans"
    for n in range(2):
        _make_case(elsewhere / f"sub-{n:03d}", BRATS2021_SUFFIXES, "_")

    root = tmp_path / "data"
    root.mkdir()
    _link_dir(root / "africa_cohort", elsewhere)

    found = find_case_dirs(root)
    assert len(found) == 2
    for case_dir in found:
        assert cohort_for_case(case_dir, root) == "low"


# -- the halves meeting: a real volume, read end to end -----------------------


def test_loads_a_real_nifti_case_with_the_right_cohort(tmp_path):
    base = tmp_path / "PKG - BraTS-Africa" / "BraTS-Africa" / "95_Glioma"
    _write_case(base / "BraTS-SSA-00002-000", AFRICA_SUFFIXES, "-")

    (case,) = list(load_brats_roots([base]))
    assert case.patient_id == "BraTS-SSA-00002-000"
    assert case.cohort == "low"
    assert case.volumes.shape == (4, *SHAPE)
    assert case.mask.shape == SHAPE
    assert case.brain_mask.shape == SHAPE
    assert case.truth["lr_axis"] == 0


@pytest.mark.parametrize("x_scale", [-1.0, +1.0])
def test_storage_order_does_not_survive_loading(tmp_path, x_scale):
    """Either storage order must arrive in one canonical frame.

    This test used to pin the opposite contract -- that `left_is_low_index`
    tracked the sign of the affine -- which was right while the loader only
    read orientation. It now reorients to RAS+ as well, because reading the
    affine fixes the *labels* while leaving the *pixels* of one cohort mirrored
    against the other, and the network is trained on the pixels.

    So the flag is now a property of the canonical frame, and is the same for
    every case. What must still follow the file is the anatomical laterality
    label -- pinned in test_laterality_label_survives_reorientation.
    """
    affine = np.diag([x_scale, 1.0, 1.0, 1.0])
    base = tmp_path / "BraTS2021_Training_Data"
    _write_case(base / "BraTS2021_00000", BRATS2021_SUFFIXES, "_", affine=affine)

    (case,) = list(load_brats_roots([base]))
    assert case.truth["lr_axis"] == 0
    # Canonical RAS+: axis 0 runs toward anatomical right, so the left
    # hemisphere sits at the low indices.
    assert case.truth["left_is_low_index"] is True


def test_a_missing_modality_fails_loudly(tmp_path):
    """Silently dropping a sequence is worse than refusing to load the case."""
    case_dir = _write_case(
        tmp_path / "BraTS2021_00000", ("t1", "t2", "flair", "seg"), "_"
    )
    with pytest.raises(FileNotFoundError, match="t1ce"):
        list(load_brats_roots([case_dir.parent]))

"""Loader for real BraTS-style directory trees.

Handles both the BraTS 2021 naming convention and the BraTS 2023/Africa one,
because the training cohort and the low-quality evaluation cohort use different
suffixes and silently missing a modality is worse than failing loudly:

  BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz
  BraTS-SSA-00002-000/BraTS-SSA-00002-000-t1c.nii.gz

Laterality depends on image orientation, so the left-right direction is read
from each volume's affine rather than assumed. Getting this wrong silently
mirrors every laterality label, which would be invisible in aggregate accuracy
because the classes stay balanced.

Finding the cases, and labelling their cohort
---------------------------------------------
Both are done defensively, because both failed silently on a real download.

The distributed BraTS-Africa package nests an extra level for tumour type:

  PKG - BraTS-Africa/BraTS-Africa/95_Glioma/BraTS-SSA-00002-000/

An earlier version of this module walked exactly two levels and read the
cohort from the case's immediate parent directory. Against that tree it found
the cases but tagged all 146 of them ``"high"`` -- ``95_Glioma`` carries no
Africa marker -- which silently turns the target cohort into part of the source
cohort and makes the external protocol measure nothing. Pointed one level
higher it found zero cases instead.

So: case directories are discovered at *any* depth (a directory holding a
segmentation is a case, and is never descended into), and the cohort is decided
by scanning the case directory's own name and every ancestor name up to the
root the caller supplied. ``BraTS-SSA-00002-000`` carries the marker itself, so
the answer no longer depends on how the archive happened to be nested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Accepted suffixes per modality, in priority order.
MODALITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "t1": ("t1n", "t1"),
    "t1ce": ("t1c", "t1ce", "t1gd"),
    "t2": ("t2w", "t2"),
    "flair": ("t2f", "flair"),
}
SEG_PATTERNS: tuple[str, ...] = ("seg",)

MODALITY_ORDER: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


@dataclass
class BratsCase:
    """Same shape as `cgate.data.synthetic.SynthCase` so the pipeline is agnostic."""

    patient_id: str
    volumes: np.ndarray
    mask: np.ndarray
    brain_mask: np.ndarray
    cohort: str
    quality_level: float
    truth: dict = field(default_factory=dict)


def _require_nibabel():
    try:
        import nibabel  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "nibabel is required to read NIfTI volumes. "
            "Install it with:  pip install -e '.[io]'"
        ) from exc


def _find_one(case_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        for pattern in (f"*_{suffix}.nii*", f"*-{suffix}.nii*"):
            hits = sorted(case_dir.glob(pattern))
            if hits:
                return hits[0]
    return None


def left_is_low_index(affine: np.ndarray) -> bool:
    """True when increasing index along the L-R axis moves toward anatomical left.

    Read from the affine's orientation codes so the answer follows the file
    rather than an assumption about the dataset.
    """
    _require_nibabel()
    import nibabel as nib

    codes = nib.aff2axcodes(affine)
    for code in codes:
        if code in ("L", "R"):
            return code == "L"
    raise ValueError(f"no left-right axis found in orientation codes {codes}")


def lr_axis(affine: np.ndarray) -> int:
    _require_nibabel()
    import nibabel as nib

    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes):
        if code in ("L", "R"):
            return axis
    raise ValueError(f"no left-right axis found in orientation codes {codes}")


# Markers that identify the degraded cohort in a directory or case name.
# 'ssa' is matched with boundaries so an unrelated directory ('assay') cannot
# quietly reclassify a whole dataset; the others are distinctive enough as
# plain substrings.
_LOW_QUALITY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"africa"),
    re.compile(r"sub[-_ ]?saharan"),
    re.compile(r"(?<![a-z0-9])ssa(?![a-z0-9])"),
)

LOW, HIGH = "low", "high"


def _looks_low_quality(name: str) -> bool:
    lowered = name.lower()
    return any(p.search(lowered) for p in _LOW_QUALITY_PATTERNS)


def infer_cohort(root: Path) -> str:
    """Map a single directory name to a quality cohort.

    BraTS-Africa is 1.5 T data from resource-constrained settings and is the
    intended low-quality evaluation stratum; BraTS 2021 is predominantly 3 T.

    This inspects one name only. Use `cohort_for_case` when loading, which also
    considers the ancestors — see the module docstring for why one name is not
    enough.
    """
    return LOW if _looks_low_quality(Path(root).name) else HIGH


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _names_up_to(case_dir: Path, root: Path):
    """Yield `case_dir`'s name, then each ancestor's, up to and including `root`.

    Bounded at `root` on purpose: scanning all the way to the filesystem root
    would let an unrelated parent directory decide the cohort of a whole study.
    A case outside `root` yields its own name only, for the same reason.
    """
    case_dir, root = Path(case_dir), Path(root)
    yield case_dir.name
    # Tried unresolved first, then resolved. `find_case_dirs` returns paths
    # built by descending from `root`, so the unresolved form always relates
    # them -- resolving both first meant that a junction anywhere between root
    # and case put them on different branches, `relative_to` raised, and the
    # ancestor scan silently did nothing: every case under a symlinked cohort
    # directory fell back to "high", which is the bug this scan was added for.
    for base, target in ((root, case_dir), (_resolved(root), _resolved(case_dir))):
        try:
            relative = target.relative_to(base)
        except ValueError:
            continue
        case_dir, root = target, base
        break
    else:
        return
    current = case_dir
    for _ in range(len(relative.parts) - 1):
        current = current.parent
        yield current.name
    yield root.name


def cohort_for_case(case_dir: Path, root: Path) -> str:
    """Decide a case's cohort from its own name and its ancestors up to `root`.

    Any match wins, in either direction: the case's own name is the most
    portable signal (`BraTS-SSA-00002-000` says Africa whatever it was unpacked
    into), while the ancestors cover cohorts whose marker lives only in a
    directory name.

    The cost of `any` is that `root` votes too, so a root named for one cohort
    marks everything beneath it -- point `--root` at a mixed directory called
    `africa-study` and all of BraTS 2021 comes back "low". Pass one `--root`
    per cohort, or set `--cohort` explicitly; the run prints the resulting
    counts before fitting anything so the mistake is visible.
    """
    return LOW if any(_looks_low_quality(n) for n in _names_up_to(case_dir, root)) else HIGH


# BraTS 2021 labels enhancing tumour 4 and leaves 3 unused. The 2023 challenge
# renumbered it to 3, and BraTS-Africa ships in the 2023 format. Both are
# canonicalised to the 2021 convention on load, because everything downstream --
# REGION_LABELS, region_mask, the ET and TC composites, `et_present`,
# `et_fraction` -- is written against LABEL_ET = 4.
LEGACY_ET, MODERN_ET = 4, 3


def canonicalise_labels(mask: np.ndarray, context: str = "") -> np.ndarray:
    """Renumber a 2023-convention segmentation to the 2021 convention.

    Left unhandled this is silent and severe rather than loud: every
    2023-format case reports no enhancing tumour at all, so `et_present`
    collapses to absent and `et_fraction` to its lowest tercile for that entire
    cohort. Since BraTS-Africa is the low-quality stratum and BraTS 2021 the
    high-quality one, the artefact aligns exactly with the contrast this
    project measures -- it would read as a catastrophic quality effect, and
    every metric would agree with it.
    """
    has_modern = bool(np.any(mask == MODERN_ET))
    has_legacy = bool(np.any(mask == LEGACY_ET))
    if has_modern and has_legacy:
        raise ValueError(
            f"{context or 'segmentation'}: contains both label 3 and label 4, so "
            f"the convention cannot be determined. BraTS 2021 uses 4 for "
            f"enhancing tumour and does not use 3; BraTS 2023 uses 3."
        )
    if not has_modern:
        return mask
    out = mask.copy()
    out[mask == MODERN_ET] = LEGACY_ET
    return out


def load_case(case_dir: Path, cohort: str | None = None) -> BratsCase:
    _require_nibabel()
    import nibabel as nib

    volumes = []
    affine = None
    for modality in MODALITY_ORDER:
        path = _find_one(case_dir, MODALITY_PATTERNS[modality])
        if path is None:
            raise FileNotFoundError(
                f"{case_dir.name}: no file matching modality {modality!r} "
                f"(tried suffixes {MODALITY_PATTERNS[modality]})"
            )
        img = nib.load(str(path))
        volumes.append(np.asarray(img.dataobj, dtype=np.float32))
        if affine is None:
            affine = img.affine

    seg_path = _find_one(case_dir, SEG_PATTERNS)
    if seg_path is None:
        raise FileNotFoundError(f"{case_dir.name}: no segmentation file found")
    mask = np.asarray(nib.load(str(seg_path)).dataobj).astype(np.int16)
    mask = canonicalise_labels(mask, context=case_dir.name)

    vols = np.stack(volumes, axis=0)
    if vols.shape[1:] != mask.shape:
        raise ValueError(
            f"{case_dir.name}: volume shape {vols.shape[1:]} does not match "
            f"segmentation shape {mask.shape}"
        )

    # Skull-stripped data: brain is wherever any modality has signal.
    brain_mask = np.any(vols > 0, axis=0)

    return BratsCase(
        patient_id=case_dir.name,
        volumes=vols,
        mask=mask,
        brain_mask=brain_mask,
        cohort=cohort or infer_cohort(case_dir.parent),
        quality_level=float("nan"),  # unknown for real data; IQMs measure it
        truth={
            "affine": affine,
            "lr_axis": lr_axis(affine),
            "left_is_low_index": left_is_low_index(affine),
        },
    )


MAX_SEARCH_DEPTH = 6


def find_case_dirs(root: Path, max_depth: int = MAX_SEARCH_DEPTH) -> list[Path]:
    """Every case directory at any depth under `root`, sorted.

    A directory holding a segmentation *is* a case and is never descended into,
    so the search cannot mistake a case's contents for nested datasets. Depth is
    bounded so a mistyped root cannot start walking an entire drive.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    found: list[Path] = []
    # `is_dir()` follows symlinks and NTFS junctions, so a mirrored or
    # self-referential link makes the same case reachable by several paths. The
    # duplicates would be yielded as repeated patient ids and land in different
    # splits -- the one form of leakage `assert_no_leakage` cannot see, because
    # by then they are simply two rows. Resolved paths are what actually
    # identify a directory, so dedupe on those.
    seen: set[Path] = set()

    def walk(directory: Path, depth: int) -> None:
        try:
            key = directory.resolve()
        except OSError:
            key = directory
        if key in seen:
            return
        seen.add(key)

        if _find_one(directory, SEG_PATTERNS) is not None:
            found.append(directory)
            return
        if depth >= max_depth:
            return
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                walk(child, depth + 1)

    walk(root, 0)
    return found


def subsample(case_dirs: list[Path], max_cases: int | None, seed: int = 0) -> list[Path]:
    """A reproducible random subset of the discovered cases.

    Random rather than the first N: BraTS ids are grouped by contributing
    institution, so a prefix is a subset of sites rather than a subset of the
    study -- and site is exactly what the quality cohorts are meant to vary.
    Seeded and re-sorted, so the same --max-cases always picks the same cases.
    """
    if max_cases is None or max_cases >= len(case_dirs):
        return case_dirs
    if max_cases < 1:
        raise ValueError(f"max_cases must be >= 1, got {max_cases}")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(case_dirs), size=max_cases, replace=False)
    return [case_dirs[i] for i in sorted(picked)]


def load_brats_cases(
    root: Path,
    cohort: str | None = None,
    max_depth: int = MAX_SEARCH_DEPTH,
    max_cases: int | None = None,
    sample_seed: int = 0,
):
    """Yield every case found under `root`, at any nesting depth.

    `cohort` forces every case to that cohort. Left as None, each case is
    labelled from its own name and its ancestors up to `root` — see
    `cohort_for_case`.
    """
    root = Path(root)
    case_dirs = find_case_dirs(root, max_depth=max_depth)
    if not case_dirs:
        raise FileNotFoundError(
            f"no case directories under {root} (searched {max_depth} levels for a "
            f"file matching {SEG_PATTERNS}). Is the archive still packed?"
        )
    for case_dir in subsample(case_dirs, max_cases, sample_seed):
        yield load_case(case_dir, cohort=cohort or cohort_for_case(case_dir, root))


def load_brats_roots(
    roots,
    cohorts=None,
    max_depth: int = MAX_SEARCH_DEPTH,
    max_cases: int | None = None,
    sample_seed: int = 0,
):
    """Yield cases from several dataset roots in turn.

    The real experiment needs this: BraTS 2021 and BraTS-Africa are separate
    downloads and can easily sit on different drives, so requiring one shared
    parent directory would mean copying tens of gigabytes to satisfy the loader.

    `cohorts` may be None (infer every root), a single string (force all), or a
    sequence aligned with `roots` whose entries may individually be None.
    """
    roots = [Path(r) for r in roots]
    if not roots:
        raise ValueError("no dataset roots supplied")

    if cohorts is None or isinstance(cohorts, str):
        per_root = [cohorts] * len(roots)
    else:
        per_root = list(cohorts)
        if len(per_root) != len(roots):
            raise ValueError(
                f"got {len(per_root)} cohort labels for {len(roots)} roots; pass "
                f"one per root or none at all"
            )

    seen: dict[str, Path] = {}
    for root, cohort in zip(roots, per_root):
        for case in load_brats_cases(
            root,
            cohort=cohort,
            max_depth=max_depth,
            max_cases=max_cases,
            sample_seed=sample_seed,
        ):
            # Two roots that overlap would duplicate patients across splits,
            # which assert_no_leakage cannot detect because both copies carry
            # the same id and would land in the same split.
            if case.patient_id in seen:
                raise ValueError(
                    f"patient {case.patient_id!r} found under both "
                    f"{seen[case.patient_id]} and {root}; dataset roots must not overlap"
                )
            seen[case.patient_id] = root
            yield case

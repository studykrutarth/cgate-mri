"""Derive the case-level finding vector from a segmentation mask.

This is the bridge between voxel-level expert output and the fused target defined
in `cgate.schema`. Ground-truth findings come from the reference mask; a CNN/ViT
expert's findings come from running the same reduction on its *predicted* mask,
which is what makes voxel-level and case-level experts comparable at all.

Anything fitted from data (the tercile edges) is fitted on the training split
only and then frozen -- see `FindingExtractor.fit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import (
    FINDING_SCHEMA,
    LABEL_ET,
    LABEL_NCR,
    N_FINDINGS,
    REGION_LABELS,
    finding_index,
)

# 0.5 mL at 1 mm isotropic. Below this a "finding" is indistinguishable from
# annotation jitter, so calling it present would inject label noise.
DEFAULT_MIN_VOXELS = 500

# Fraction of whole-tumour volume that must sit on the minority side before the
# mass is called bilateral rather than unilateral-with-spill.
DEFAULT_BILATERAL_FRACTION = 0.15


def region_mask(mask: np.ndarray, region: str) -> np.ndarray:
    """Boolean mask for a BraTS composite region ('WT', 'TC', 'ET')."""
    try:
        labels = REGION_LABELS[region]
    except KeyError:
        raise KeyError(
            f"unknown region {region!r}; expected one of {sorted(REGION_LABELS)}"
        ) from None
    return np.isin(mask, labels)


def region_volumes(mask: np.ndarray) -> dict[str, int]:
    """Voxel counts for each composite region plus the raw sub-labels."""
    return {
        "WT": int(region_mask(mask, "WT").sum()),
        "TC": int(region_mask(mask, "TC").sum()),
        "ET": int((mask == LABEL_ET).sum()),
        "NCR": int((mask == LABEL_NCR).sum()),
    }


def laterality_split(
    mask: np.ndarray,
    spatial_axis: int = 0,
    left_is_low_index: bool = True,
    bilateral_fraction: float = DEFAULT_BILATERAL_FRACTION,
) -> int:
    """Classify whole-tumour laterality -> 0=left, 1=right, 2=bilateral.

    `spatial_axis` is the left-right (sagittal) axis. For BraTS volumes
    registered to SRI24 this is axis 0, but the anatomical direction of
    increasing index depends on the image orientation code, so
    `left_is_low_index` must be checked against the affine for real data --
    `cgate.data.brats` does that and passes the correct value through.
    """
    wt = region_mask(mask, "WT")
    total = int(wt.sum())
    if total == 0:
        # No tumour: laterality is undefined. Caller filters these out, but be
        # deterministic rather than raising mid-batch.
        return 2

    mid = wt.shape[spatial_axis] // 2
    lo = wt.take(indices=range(0, mid), axis=spatial_axis).sum()
    hi = total - lo

    lo_frac = lo / total
    hi_frac = hi / total
    if min(lo_frac, hi_frac) >= bilateral_fraction:
        return 2

    low_side_dominant = lo_frac > hi_frac
    if left_is_low_index:
        return 0 if low_side_dominant else 1
    return 1 if low_side_dominant else 0


def _tercile_class(value: float, edges: tuple[float, float]) -> int:
    lo, hi = edges
    if value <= lo:
        return 0
    if value <= hi:
        return 1
    return 2


def measure(
    mask: np.ndarray,
    spatial_axis: int = 0,
    left_is_low_index: bool = True,
    bilateral_fraction: float = DEFAULT_BILATERAL_FRACTION,
) -> dict[str, float]:
    """Reduce a mask to the few scalars the finding vector needs.

    Kept separate from label assignment so a pipeline can stream volumes,
    record these handful of numbers per case, and discard the mask -- the
    tercile edges cannot be fitted until the training split is known, which is
    after the streaming pass. Holding every mask in memory just to defer that
    decision costs hundreds of megabytes for no benefit.
    """
    vols = region_volumes(mask)
    wt = vols["WT"]
    return {
        "wt": float(wt),
        "tc": float(vols["TC"]),
        "et": float(vols["ET"]),
        "ncr": float(vols["NCR"]),
        "et_fraction": float(vols["ET"] / wt) if wt > 0 else 0.0,
        "laterality": float(
            laterality_split(
                mask,
                spatial_axis=spatial_axis,
                left_is_low_index=left_is_low_index,
                bilateral_fraction=bilateral_fraction,
            )
        ),
    }


@dataclass
class FindingExtractor:
    """Mask -> finding-vector reduction with train-fitted tercile edges."""

    min_voxels: int = DEFAULT_MIN_VOXELS
    spatial_axis: int = 0
    left_is_low_index: bool = True
    bilateral_fraction: float = DEFAULT_BILATERAL_FRACTION
    burden_edges: tuple[float, float] | None = None
    et_fraction_edges: tuple[float, float] | None = None
    _fitted: bool = field(default=False, repr=False)

    # -- fitting -----------------------------------------------------------
    def fit(self, masks: list[np.ndarray]) -> FindingExtractor:
        """Fit tercile edges on the training split only.

        Call this with training masks and nothing else. Fitting on all data
        would leak the test distribution into the label definition.
        """
        if not masks:
            raise ValueError("cannot fit FindingExtractor on zero masks")

        burdens: list[float] = []
        et_fracs: list[float] = []
        for m in masks:
            vols = region_volumes(m)
            wt = vols["WT"]
            if wt == 0:
                continue
            burdens.append(float(wt))
            et_fracs.append(vols["ET"] / wt)

        if not burdens:
            raise ValueError("every training mask was empty; cannot fit terciles")

        self.burden_edges = _edges_from(burdens)
        self.et_fraction_edges = _edges_from(et_fracs)
        self._fitted = True
        return self

    def fit_from_measurements(
        self, measurements: list[dict[str, float]]
    ) -> FindingExtractor:
        """Fit tercile edges from `measure` output for the training split only."""
        burdens = [m["wt"] for m in measurements if m["wt"] > 0]
        et_fracs = [m["et_fraction"] for m in measurements if m["wt"] > 0]
        if not burdens:
            raise ValueError("every training case was empty; cannot fit terciles")
        self.burden_edges = _edges_from(burdens)
        self.et_fraction_edges = _edges_from(et_fracs)
        self._fitted = True
        return self

    def labels_from_measurements(
        self, measurements: list[dict[str, float]]
    ) -> np.ndarray:
        """Integer label matrix (n_cases, K) from `measure` output."""
        if not self.is_fitted:
            raise RuntimeError(
                "FindingExtractor must be fitted on the training split first"
            )
        assert self.burden_edges is not None
        assert self.et_fraction_edges is not None

        out = np.zeros((len(measurements), N_FINDINGS), dtype=np.int64)
        for i, m in enumerate(measurements):
            out[i, finding_index("et_present")] = int(m["et"] >= self.min_voxels)
            out[i, finding_index("ncr_present")] = int(m["ncr"] >= self.min_voxels)
            out[i, finding_index("laterality")] = int(m["laterality"])
            out[i, finding_index("burden")] = _tercile_class(m["wt"], self.burden_edges)
            out[i, finding_index("et_fraction")] = _tercile_class(
                m["et_fraction"], self.et_fraction_edges
            )
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted or (
            self.burden_edges is not None and self.et_fraction_edges is not None
        )

    # -- application -------------------------------------------------------
    def extract(self, mask: np.ndarray) -> np.ndarray:
        """Return the integer finding vector, shape (K,), for one case."""
        if not self.is_fitted:
            raise RuntimeError(
                "FindingExtractor.fit() must be called on the training split "
                "before extracting findings"
            )
        assert self.burden_edges is not None
        assert self.et_fraction_edges is not None

        vols = region_volumes(mask)
        wt = vols["WT"]
        out = np.zeros(N_FINDINGS, dtype=np.int64)

        out[finding_index("et_present")] = int(vols["ET"] >= self.min_voxels)
        out[finding_index("ncr_present")] = int(vols["NCR"] >= self.min_voxels)
        out[finding_index("laterality")] = laterality_split(
            mask,
            spatial_axis=self.spatial_axis,
            left_is_low_index=self.left_is_low_index,
            bilateral_fraction=self.bilateral_fraction,
        )
        out[finding_index("burden")] = _tercile_class(float(wt), self.burden_edges)
        et_frac = (vols["ET"] / wt) if wt > 0 else 0.0
        out[finding_index("et_fraction")] = _tercile_class(
            et_frac, self.et_fraction_edges
        )
        return out

    def extract_many(self, masks: list[np.ndarray]) -> np.ndarray:
        """Return the integer label matrix, shape (n_cases, K)."""
        if not masks:
            return np.zeros((0, N_FINDINGS), dtype=np.int64)
        return np.stack([self.extract(m) for m in masks], axis=0)


def _edges_from(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.quantile(arr, [1 / 3, 2 / 3])
    if hi <= lo:
        # Degenerate distribution (e.g. tiny synthetic cohort). Nudge the upper
        # edge so the three classes stay distinguishable instead of collapsing.
        hi = float(lo) + 1e-6
    return float(lo), float(hi)


def label_prevalence(labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-finding class prevalence -- used to check no finding is degenerate.

    A finding whose majority class exceeds ~95% makes accuracy meaningless and
    calibration trivial, so this is reported alongside every experiment.
    """
    out: dict[str, dict[str, float]] = {}
    n = max(len(labels), 1)
    for j, spec in enumerate(FINDING_SCHEMA):
        counts = np.bincount(labels[:, j], minlength=spec.n_classes)
        out[spec.name] = {
            cls: float(counts[i] / n) for i, cls in enumerate(spec.classes)
        }
    return out

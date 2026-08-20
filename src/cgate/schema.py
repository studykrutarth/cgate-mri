"""The prediction target that every expert must agree on.

This module exists to answer one question precisely: *what exactly gets fused?*

A CNN and a ViT emit voxel-wise class maps; an XGBoost-on-radiomics model emits a
case-level logit. Those live at different granularities and cannot be averaged
together. So the system defines a single common target that all three experts are
required to produce, and fusion happens only there:

    a case-level **structured finding vector** -- a fixed list of K categorical
    findings, each expert emitting one probability simplex per finding.

Voxel-level segmentation is *upstream* of this target (the CNN/ViT experts reduce
their masks to findings via `cgate.findings.findings_from_mask`) and is reported
as an auxiliary metric. It is never the gated quantity.

The gate therefore operates on calibrated case-level finding probabilities, which
is also the granularity a radiologist would triage at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# BraTS label convention (as published: 4 is enhancing tumour, 3 is unused).
LABEL_BACKGROUND = 0
LABEL_NCR = 1  # necrotic / non-enhancing tumour core
LABEL_ED = 2  # peritumoural edema
LABEL_ET = 4  # enhancing tumour

# Composite regions used by the BraTS challenge.
REGION_LABELS: dict[str, tuple[int, ...]] = {
    "WT": (LABEL_NCR, LABEL_ED, LABEL_ET),  # whole tumour
    "TC": (LABEL_NCR, LABEL_ET),  # tumour core
    "ET": (LABEL_ET,),  # enhancing tumour
}


@dataclass(frozen=True)
class FindingSpec:
    """One categorical finding in the fused prediction target."""

    name: str
    classes: tuple[str, ...]
    critical: bool = True
    description: str = ""

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# ---------------------------------------------------------------------------
# The canonical target.
#
# Every finding below is derivable from the BraTS segmentation mask alone, with
# no extra annotation and no ambiguity. Each was *intended* to have non-degenerate
# class prevalence -- e.g. "edema present" is deliberately excluded because it is
# true in almost every BraTS case and would be trivially predictable, inflating
# accuracy while telling us nothing about calibration.
#
# On real BraTS data that intention is not met, and this comment used to claim it
# was. Measured over 146 BraTS-Africa masks (F13): `et_present` is true in 0.973
# of cases at the current threshold and in 0.986 at a threshold of one voxel, so
# it is precisely the "edema present" case the paragraph above says was excluded.
# `ncr_present` is 0.781, degenerate for a different reason -- 500 voxels is a
# phantom-era constant an order of magnitude below the real NCR distribution.
#
# Resolved differently for each, because they fail differently. `ncr_present`
# responds to the threshold, so `FindingExtractor` now fits it on the training
# split. `et_present` responds to no threshold, so it is demoted out of the
# critical set rather than redefined: it stays a reported finding and stops
# gating cases.
# ---------------------------------------------------------------------------
FINDING_SCHEMA: tuple[FindingSpec, ...] = (
    FindingSpec(
        name="et_present",
        classes=("absent", "present"),
        # Not critical: measured at 0.973 prevalence on real BraTS masks and
        # 0.986 at a threshold of one voxel (F13). Still reported, because
        # "does this tumour enhance" is a real question and the answer is
        # occasionally no -- but a near-constant finding cannot gate a case.
        critical=False,
        description="Enhancing tumour above volume threshold. Intended to vary "
        "with grade, and does on the phantoms; on real BraTS gliomas it is "
        "present in almost every case -- see F13.",
    ),
    FindingSpec(
        name="ncr_present",
        classes=("absent", "present"),
        critical=True,
        description="Necrotic / non-enhancing core above volume threshold.",
    ),
    FindingSpec(
        name="laterality",
        classes=("left", "right", "bilateral"),
        critical=True,
        description="Hemisphere of the whole-tumour mass relative to the "
        "mid-sagittal plane.",
    ),
    FindingSpec(
        name="burden",
        classes=("small", "medium", "large"),
        critical=False,
        description="Whole-tumour volume tercile. Tercile edges are fitted on the "
        "training split only.",
    ),
    FindingSpec(
        name="et_fraction",
        classes=("low", "mid", "high"),
        critical=False,
        description="ET/WT volume ratio tercile -- how much of the mass enhances. "
        "Tercile edges fitted on the training split only.",
    ),
)

FINDING_NAMES: tuple[str, ...] = tuple(f.name for f in FINDING_SCHEMA)
N_FINDINGS: int = len(FINDING_SCHEMA)
CLASS_COUNTS: tuple[int, ...] = tuple(f.n_classes for f in FINDING_SCHEMA)
CRITICAL_MASK: tuple[bool, ...] = tuple(f.critical for f in FINDING_SCHEMA)


def spec_by_name(name: str) -> FindingSpec:
    for f in FINDING_SCHEMA:
        if f.name == name:
            return f
    raise KeyError(f"unknown finding {name!r}; expected one of {FINDING_NAMES}")


def finding_index(name: str) -> int:
    return FINDING_NAMES.index(name)


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


@dataclass
class ExpertPrediction:
    """One expert's output for a batch of cases, on the common target.

    `probs` is a list of length K (one entry per finding). Entry j has shape
    (n_cases, n_classes_j) and each row sums to 1.

    Keeping findings as a list rather than one padded array avoids inventing
    probability mass for findings with fewer classes, which would silently
    corrupt entropy and calibration measurements.
    """

    expert: str
    probs: list[np.ndarray]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_probs(self.probs, context=f"expert {self.expert!r}")

    @property
    def n_cases(self) -> int:
        return int(self.probs[0].shape[0])

    def subset(self, idx: np.ndarray) -> ExpertPrediction:
        return ExpertPrediction(
            expert=self.expert,
            probs=[p[idx] for p in self.probs],
            meta=dict(self.meta),
        )


def validate_probs(probs: list[np.ndarray], context: str = "") -> None:
    """Fail loudly on malformed probability output.

    Calibration and selective-prediction metrics are silently meaningless on
    unnormalised or wrongly-shaped input, so this is checked at every boundary
    rather than trusted.
    """
    where = f" ({context})" if context else ""
    if len(probs) != N_FINDINGS:
        raise ValueError(
            f"expected {N_FINDINGS} findings, got {len(probs)}{where}"
        )
    n = probs[0].shape[0]
    for spec, p in zip(FINDING_SCHEMA, probs):
        if p.ndim != 2:
            raise ValueError(f"finding {spec.name!r} must be 2-D, got {p.ndim}-D{where}")
        if p.shape[0] != n:
            raise ValueError(
                f"finding {spec.name!r} has {p.shape[0]} rows, expected {n}{where}"
            )
        if p.shape[1] != spec.n_classes:
            raise ValueError(
                f"finding {spec.name!r} expects {spec.n_classes} classes, "
                f"got {p.shape[1]}{where}"
            )
        if not np.all(np.isfinite(p)):
            raise ValueError(f"finding {spec.name!r} contains non-finite values{where}")
        if np.any(p < -1e-6):
            raise ValueError(f"finding {spec.name!r} contains negative values{where}")
        sums = p.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            bad = int(np.argmax(np.abs(sums - 1.0)))
            raise ValueError(
                f"finding {spec.name!r} row {bad} sums to {sums[bad]:.6f}, "
                f"expected 1.0{where}"
            )


def empty_labels(n_cases: int) -> np.ndarray:
    """Integer label matrix of shape (n_cases, K)."""
    return np.zeros((n_cases, N_FINDINGS), dtype=np.int64)


def validate_labels(labels: np.ndarray) -> None:
    if labels.ndim != 2 or labels.shape[1] != N_FINDINGS:
        raise ValueError(
            f"labels must have shape (n_cases, {N_FINDINGS}), got {labels.shape}"
        )
    for j, spec in enumerate(FINDING_SCHEMA):
        col = labels[:, j]
        if col.size and (col.min() < 0 or col.max() >= spec.n_classes):
            raise ValueError(
                f"finding {spec.name!r} labels out of range [0, {spec.n_classes - 1}]"
            )

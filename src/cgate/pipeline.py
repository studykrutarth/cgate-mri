"""End-to-end system assembly: ingest -> experts -> calibrate -> route -> gate.

Split discipline
----------------
Every fitted component gets data that no later component reuses for a decision
it would benefit from having seen:

  train      experts
  calib_fit  per-expert temperatures, and the router
  calib_gate final fused temperature, and the gate threshold
  test       reporting only, touched once

`calib` is halved because the router is *optimised* on calib_fit; choosing the
gate threshold on the same cases would let the gate exploit the router's
in-sample fit and overstate coverage at a given risk. This is the difference
between a gate that holds up on new scans and one that only looks good in the
table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import MultiFindingTemperatureScaler
from .data.features import FEATURE_VIEWS, extract_views
from .findings import FindingExtractor, measure
from .fusion import combine, confidence_weights, uniform_weights
from .quality import ROUTER_FEATURES, QualityStandardizer, compute_iqms, iqm_vector
from .routing import QualityRouter, RouterConfig
from .schema import (
    CRITICAL_MASK,
    FINDING_NAMES,
    FINDING_SCHEMA,
    N_FINDINGS,
    validate_probs,
)
from .selective import ConfidenceGate


# Bumped whenever the on-disk table layout changes, so a stale cache fails
# loudly instead of being read with the wrong column meanings.
# v2 added slice_cache_build_id. A v1 table has no way to prove which cache it
# was built against, so it is rejected rather than trusted.
TABLE_FORMAT_VERSION = 2


@dataclass
class CaseTable:
    """Everything the pipeline needs, after volumes have been discarded."""

    patient_ids: np.ndarray
    cohorts: np.ndarray
    quality_level: np.ndarray
    iqms: np.ndarray  # (n, len(ROUTER_FEATURES))
    views: dict[str, np.ndarray]
    measurements: list[dict[str, float]]
    labels: np.ndarray | None = None
    iqm_names: tuple[str, ...] = ROUTER_FEATURES
    # Present only when ingest built one. The torch experts read pixels through
    # this; the tabular experts ignore it entirely.
    slice_cache: object | None = None
    # Where the cases came from. Travels with a saved table so a reused one
    # cannot be reported as synthetic just because --data was left at its
    # default -- the same reason the expert family is recorded in every run.
    source: dict = field(default_factory=dict)

    @property
    def n_cases(self) -> int:
        return int(len(self.patient_ids))

    def assign_labels(self, extractor: FindingExtractor) -> None:
        self.labels = extractor.labels_from_measurements(self.measurements)

    # -- persistence -------------------------------------------------------
    # Ingest reads every NIfTI in the study, computes IQMs and feature views,
    # then throws the volumes away. That pass costs far more than fitting the
    # grid does, and it is repeated identically for every seed and every router
    # configuration. Saving the table turns "re-read ~1400 volumes" into "read
    # a few megabytes", which is also what makes it practical to ingest on one
    # machine and train on another: the raw archive is ~15 GB, the table and its
    # slice cache are a couple of GB.
    #
    # Labels are deliberately not saved. Their tercile edges are fitted on the
    # train split, so they belong to a run, not to the dataset; persisting them
    # would let one run's split silently define another run's labels.
    def save(self, root, source: dict | None = None) -> Path:
        """Write the table to `root` as arrays.npz + meta.json. Returns `root`."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if source is not None:
            self.source = source

        arrays = {
            "patient_ids": np.asarray(self.patient_ids, dtype=str),
            "cohorts": np.asarray(self.cohorts, dtype=str),
            "quality_level": np.asarray(self.quality_level, dtype=np.float64),
            "iqms": np.asarray(self.iqms, dtype=np.float64),
        }
        for name, matrix in self.views.items():
            arrays[f"view__{name}"] = np.asarray(matrix)
        np.savez_compressed(root / "arrays.npz", **arrays)

        cache_root = getattr(self.slice_cache, "root", None)
        meta = {
            "format_version": TABLE_FORMAT_VERSION,
            "n_cases": self.n_cases,
            "iqm_names": list(self.iqm_names),
            "view_names": list(self.views),
            "measurements": self.measurements,
            # Recorded so a reloaded table can find its pixels again. Stored as
            # written; --slice-cache overrides it, because the path that was
            # right on the ingesting machine is rarely right on the training one.
            "slice_cache_root": str(cache_root) if cache_root is not None else None,
            # Pinned so `load` can tell "the cache this table was built against"
            # from "some other cache that happens to hold the same patients".
            "slice_cache_build_id": getattr(self.slice_cache, "build_id", None),
            # Paths here are provenance only and are never validated: the whole
            # point is that the table can be built on one machine and used on
            # another, where they do not exist. `data` is load-bearing though —
            # it is what stops a reused BraTS table being reported as synthetic.
            "source": self.source,
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
        return root

    @staticmethod
    def load(root, slice_cache_root=None, require_slices: bool = True) -> "CaseTable":
        """Read back a table written by `save`.

        `slice_cache_root` overrides the recorded location of the slice cache.

        `require_slices=False` lets the table load when its recorded cache is
        gone. The surrogate experts read only the tabular views and never touch
        a pixel, so demanding the multi-GB cache would block the very workflow
        the table exists for: ingest beside the ~15 GB archive, copy the few-MB
        table elsewhere, train from that. An explicit `slice_cache_root` is
        always honoured strictly — asking for a specific cache and silently
        getting none would be worse than the error.
        """
        root = Path(root)
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no saved case table at {root} (expected {meta_path.name}). "
                f"Run an ingest with --table-cache {root} to build one."
            )
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        version = meta.get("format_version")
        if version != TABLE_FORMAT_VERSION:
            raise ValueError(
                f"case table at {root} is format version {version}, this build "
                f"reads version {TABLE_FORMAT_VERSION}. Re-ingest with --rebuild-table."
            )

        with np.load(root / "arrays.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}

        views = {
            key[len("view__") :]: arrays[key]
            for key in arrays
            if key.startswith("view__")
        }
        patient_ids = arrays["patient_ids"]
        measurements = meta["measurements"]
        if len(measurements) != len(patient_ids):
            raise ValueError(
                f"case table at {root} is inconsistent: {len(patient_ids)} patients "
                f"but {len(measurements)} measurement rows"
            )

        cache = None
        cache_root = slice_cache_root or meta.get("slice_cache_root")
        if cache_root is not None:
            from .data.slices import SliceCache

            try:
                cache = SliceCache.open(Path(cache_root))
            except FileNotFoundError:
                # Only tolerated for the recorded path: an explicitly requested
                # cache that is absent stays an error.
                if require_slices or slice_cache_root is not None:
                    raise
                cache = None

        if cache is not None:
            expected = meta.get("slice_cache_build_id")
            if expected is not None and cache.build_id != expected:
                raise ValueError(
                    f"the slice cache at {cache_root} is not the one this table was "
                    f"built against (build {cache.build_id} vs {expected}). Its "
                    f"patient ids may still match while its pixels are from a "
                    f"different ingest, which would train the network on the wrong "
                    f"images with the right labels. Point --slice-cache at the "
                    f"matching cache, or re-ingest with --rebuild-table."
                )
            # Re-derive the row indices instead of trusting the saved column.
            # rows_for raises on a patient the cache does not hold, which turns
            # a mismatched table/cache pair into an error here rather than into
            # a network trained on the wrong images with the right labels.
            views["volume"] = cache.rows_for(patient_ids).reshape(-1, 1)
        else:
            # The saved column indexes a cache that is not open; leaving it would
            # hand the torch experts row numbers pointing at nothing.
            views.pop("volume", None)

        return CaseTable(
            patient_ids=patient_ids,
            cohorts=arrays["cohorts"],
            quality_level=arrays["quality_level"],
            iqms=arrays["iqms"],
            views=views,
            measurements=measurements,
            iqm_names=tuple(meta["iqm_names"]),
            slice_cache=cache,
            source=meta.get("source", {}),
        )


def ingest_cases(
    cases,
    modality_for_quality: int = 3,
    slice_cache_root=None,
    slice_config=None,
) -> CaseTable:
    """Stream cases, reduce each to features/IQMs/measurements, discard volumes.

    IQMs are computed on one modality (FLAIR by default) rather than averaged
    across all four: the four sequences of one acquisition share a motion and
    shim profile, so their IQMs are highly correlated, and averaging them buys
    almost nothing while making the feature harder to interpret.
    """
    patient_ids: list[str] = []
    cohorts: list[str] = []
    quality: list[float] = []
    iqm_rows: list[np.ndarray] = []
    view_rows: dict[str, list[np.ndarray]] = {v: [] for v in FEATURE_VIEWS}
    measurements: list[dict[str, float]] = []

    # Slices are written during this same pass: the case stream is consumable
    # once, and volumes are discarded immediately afterwards.
    writer = None
    if slice_cache_root is not None:
        from .data.slices import SliceCacheWriter

        writer = SliceCacheWriter(slice_cache_root, slice_config)

    for case in cases:
        vols = case.volumes
        brain = case.brain_mask
        iqms = compute_iqms(vols[modality_for_quality].astype(np.float64), brain)
        iqm_rows.append(iqm_vector(iqms, ROUTER_FEATURES))

        views = extract_views(vols, brain)
        for name in FEATURE_VIEWS:
            view_rows[name].append(views[name])

        if writer is not None:
            writer.add(case)

        # Orientation travels with the case. `measure` defaults to axis 0 with
        # left at the low index, which is right for canonical RAS and for the
        # phantoms -- but the loader goes to the trouble of reading each
        # volume's affine and nothing was reading the answer, so a cohort
        # stored the other way round had its laterality silently mirrored
        # against the cohort it was compared with.
        truth = getattr(case, "truth", None) or {}
        measurements.append(
            measure(
                case.mask,
                spatial_axis=int(truth.get("lr_axis", 0)),
                left_is_low_index=bool(truth.get("left_is_low_index", True)),
            )
        )
        patient_ids.append(case.patient_id)
        cohorts.append(case.cohort)
        quality.append(float(case.quality_level))

    if not patient_ids:
        raise ValueError("no cases were ingested")

    views_out = {k: np.vstack(v) for k, v in view_rows.items()}
    cache = writer.close() if writer is not None else None
    if cache is not None:
        # The 'volume' view is a single column of cache row indices. That is how
        # pixels reach an interface built around tabular features without
        # changing the interface every other component depends on.
        views_out["volume"] = cache.rows_for(patient_ids).reshape(-1, 1)

    return CaseTable(
        patient_ids=np.asarray(patient_ids),
        cohorts=np.asarray(cohorts),
        quality_level=np.asarray(quality, dtype=np.float64),
        iqms=np.vstack(iqm_rows),
        views=views_out,
        measurements=measurements,
        slice_cache=cache,
    )


def case_confidence(fused: list[np.ndarray], critical_only: bool = True) -> np.ndarray:
    """Case-level confidence: the weakest link across findings.

    A case is only safe to release if *every* critical finding is trustworthy,
    so the aggregate is a minimum, not a mean. Averaging would let three
    confident findings mask one the model is guessing at -- precisely the case
    that should reach a radiologist.
    """
    validate_probs(fused, context="case_confidence")
    per_finding = []
    for j in range(N_FINDINGS):
        if critical_only and not CRITICAL_MASK[j]:
            continue
        per_finding.append(fused[j].max(axis=1))
    if not per_finding:
        raise ValueError("no findings selected for case confidence")
    return np.min(np.stack(per_finding, axis=1), axis=1)


def case_correct(
    fused: list[np.ndarray], labels: np.ndarray, critical_only: bool = True
) -> np.ndarray:
    """A case counts as correct only if every selected finding is correct."""
    oks = []
    for j in range(N_FINDINGS):
        if critical_only and not CRITICAL_MASK[j]:
            continue
        oks.append(np.argmax(fused[j], axis=1) == labels[:, j])
    return np.all(np.stack(oks, axis=1), axis=1)


@dataclass
class SystemConfig:
    name: str
    fusion: str = "router"  # one of: single, uniform, confidence, router
    calibrate_experts: bool = True
    calibrate_fused: bool = True
    router: RouterConfig = field(default_factory=RouterConfig)
    confidence_beta: float = 1.0
    seed: int = 0


class TrainedSystem:
    """A fitted system: experts + calibration + fusion rule + gate."""

    def __init__(self, config: SystemConfig, experts: list):
        if not experts:
            raise ValueError("a system needs at least one expert")
        if config.fusion == "single" and len(experts) != 1:
            raise ValueError("fusion='single' requires exactly one expert")
        self.config = config
        self.experts = experts
        self.expert_names = tuple(e.name for e in experts)
        self.standardizer = QualityStandardizer()
        self.expert_calibrators: list[MultiFindingTemperatureScaler] = []
        self.fused_calibrator: MultiFindingTemperatureScaler | None = None
        self.router: QualityRouter | None = None
        self.gate = ConfidenceGate()
        self.fit_log: dict = {}

    # -- internals ---------------------------------------------------------
    def _expert_probs(self, table: CaseTable, idx: np.ndarray, calibrated: bool):
        out = []
        for e, expert in enumerate(self.experts):
            x = table.views[expert.view][idx]
            probs = expert.predict_proba(x)
            if calibrated and self.expert_calibrators:
                probs = self.expert_calibrators[e].transform(probs)
            out.append(probs)
        return out

    def _fuse(self, expert_probs, q: np.ndarray) -> list[np.ndarray]:
        mode = self.config.fusion
        n = expert_probs[0][0].shape[0]
        if mode == "single":
            return expert_probs[0]
        if mode == "uniform":
            return combine(expert_probs, uniform_weights(n, len(expert_probs)))
        if mode == "confidence":
            return combine(
                expert_probs,
                confidence_weights(expert_probs, beta=self.config.confidence_beta),
            )
        if mode == "router":
            if self.router is None:
                raise RuntimeError("router fusion selected but router is not fitted")
            return self.router.fuse(q, expert_probs)
        raise ValueError(f"unknown fusion mode {mode!r}")

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        table: CaseTable,
        train_idx: np.ndarray,
        calib_idx: np.ndarray,
        target_coverage: float = 0.8,
    ) -> "TrainedSystem":
        if table.labels is None:
            raise RuntimeError("table.assign_labels() must be called before fitting")
        labels = table.labels

        # Quality standardiser: training statistics only, so a shift at test
        # time shows up as out-of-range features instead of being absorbed.
        self.standardizer.fit(table.iqms[train_idx])

        for expert in self.experts:
            expert.fit(table.views[expert.view][train_idx], labels[train_idx])

        # Halve the calibration split -- see module docstring.
        rng = np.random.default_rng(self.config.seed)
        shuffled = rng.permutation(calib_idx)
        half = max(1, len(shuffled) // 2)
        calib_fit, calib_gate = shuffled[:half], shuffled[half:]
        if len(calib_gate) == 0:
            calib_gate = calib_fit
        self.fit_log["n_calib_fit"] = int(len(calib_fit))
        self.fit_log["n_calib_gate"] = int(len(calib_gate))

        # 1. per-expert temperature on calib_fit
        raw_fit = self._expert_probs(table, calib_fit, calibrated=False)
        if self.config.calibrate_experts:
            self.expert_calibrators = [
                MultiFindingTemperatureScaler().fit(probs, labels[calib_fit])
                for probs in raw_fit
            ]
            self.fit_log["expert_temperatures"] = {
                name: [round(t, 4) for t in cal.temperatures]
                for name, cal in zip(self.expert_names, self.expert_calibrators)
            }

        # 2. router on calib_fit, using calibrated expert outputs
        if self.config.fusion == "router":
            cal_fit = self._expert_probs(table, calib_fit, calibrated=True)
            q_fit = self.standardizer.transform(table.iqms[calib_fit])
            self.router = QualityRouter(
                n_experts=len(self.experts),
                n_features=q_fit.shape[1],
                config=self.config.router,
            )
            # calib_gate doubles as the router's early-stopping monitor: it is
            # not used to fit router weights, only to decide when to stop.
            cal_gate = self._expert_probs(table, calib_gate, calibrated=True)
            q_gate = self.standardizer.transform(table.iqms[calib_gate])
            self.router.fit(
                q_fit,
                cal_fit,
                labels[calib_fit],
                q_val=q_gate,
                expert_probs_val=cal_gate,
                labels_val=labels[calib_gate],
            )
            self.fit_log["router_capacity"] = dict(self.router.fit_log_)
            self.fit_log["router_mean_weights"] = {
                FINDING_NAMES[j]: {
                    self.expert_names[e]: round(float(w), 4)
                    for e, w in enumerate(self.router.mean_weights(q_fit)[j])
                }
                for j in range(N_FINDINGS)
            }

        # 3. final fused temperature on calib_gate
        gate_probs = self._raw_fused(table, calib_gate)
        if self.config.calibrate_fused:
            self.fused_calibrator = MultiFindingTemperatureScaler().fit(
                gate_probs, labels[calib_gate]
            )
            gate_probs = self.fused_calibrator.transform(gate_probs)
            self.fit_log["fused_temperatures"] = [
                round(t, 4) for t in self.fused_calibrator.temperatures
            ]

        # 4. gate threshold on calib_gate
        conf = case_confidence(gate_probs)
        self.gate.fit_for_coverage(conf, target_coverage)
        self.fit_log["gate_threshold"] = round(float(self.gate.threshold_), 6)
        return self

    def _raw_fused(self, table: CaseTable, idx: np.ndarray) -> list[np.ndarray]:
        probs = self._expert_probs(table, idx, calibrated=True)
        q = self.standardizer.transform(table.iqms[idx])
        return self._fuse(probs, q)

    # -- inference ---------------------------------------------------------
    def predict(self, table: CaseTable, idx: np.ndarray) -> list[np.ndarray]:
        fused = self._raw_fused(table, idx)
        if self.fused_calibrator is not None:
            fused = self.fused_calibrator.transform(fused)
        return fused

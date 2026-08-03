"""The ablation grid.

Purpose: make the novelty claim falsifiable.

Three heterogeneous models fused together are a deep ensemble, and deep
ensembles are already a strong calibration and selective-prediction baseline. So
"our system is well calibrated" proves nothing on its own. Each row below adds
exactly one factor to the row above, which is what lets a specific claim be
attributed to a specific mechanism:

  B0 single expert, uncalibrated      -- the floor
  B1  + temperature scaling           -- isolates post-hoc calibration
  B2 sampling-based uncertainty       -- isolates uncertainty-by-sampling
  B3 homogeneous 3x deep ensemble     -- isolates *ensembling*
  B4 heterogeneous, uniform weights   -- isolates *heterogeneity* (B4 vs B3)
  B5 heterogeneous, confidence-weighted -- isolates self-reported confidence
  B6 heterogeneous, quality-routed    -- isolates *routing* (B6 vs B4/B5)

The claim this project can defend is B6 > B4 on the low-quality stratum. If that
gap is absent or its bootstrap CI crosses zero, quality-aware routing does not
help and the honest result is to report that. B6 is initialised to be exactly B4
(see `cgate.routing`), so the comparison is not confounded by a change of
fusion family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..experts.sklearn_experts import (
    DropoutEnsembleExpert,
    GlobalContextExpert,
    HomogeneousEnsemble,
    LocalPatternExpert,
    TextureRadiomicsExpert,
)
from ..pipeline import CaseTable, SystemConfig, TrainedSystem
from ..routing import RouterConfig


@dataclass
class AblationSpec:
    key: str
    label: str
    fusion: str
    experts: str  # 'single' | 'dropout' | 'homogeneous' | 'heterogeneous'
    calibrate_experts: bool = True
    isolates: str = ""


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        key="B0",
        label="Single expert, raw softmax",
        fusion="single",
        experts="single",
        calibrate_experts=False,
        isolates="uncalibrated floor",
    ),
    AblationSpec(
        key="B1",
        label="Single expert + temperature scaling",
        fusion="single",
        experts="single",
        isolates="post-hoc calibration",
    ),
    AblationSpec(
        key="B2",
        label="Sampling-based uncertainty (MC-dropout surrogate)",
        fusion="single",
        experts="dropout",
        isolates="uncertainty by sampling",
    ),
    AblationSpec(
        key="B3",
        label="Homogeneous 3x deep ensemble",
        fusion="single",
        experts="homogeneous",
        isolates="ensembling alone",
    ),
    AblationSpec(
        key="B4",
        label="Heterogeneous experts, uniform fusion",
        fusion="uniform",
        experts="heterogeneous",
        isolates="expert heterogeneity",
    ),
    AblationSpec(
        key="B5",
        label="Heterogeneous experts, confidence-weighted fusion",
        fusion="confidence",
        experts="heterogeneous",
        isolates="self-reported confidence",
    ),
    AblationSpec(
        key="B6",
        label="Heterogeneous experts, quality-aware routing (ours)",
        fusion="router",
        experts="heterogeneous",
        isolates="quality-conditioned routing",
    ),
)


def _build_experts(kind: str, seed: int) -> list:
    if kind == "single":
        return [LocalPatternExpert(seed=seed)]
    if kind == "dropout":
        return [DropoutEnsembleExpert(seed=seed, n_samples=8)]
    if kind == "homogeneous":
        return [HomogeneousEnsemble(n_members=3, seed=seed)]
    if kind == "heterogeneous":
        return [
            TextureRadiomicsExpert(seed=seed),
            LocalPatternExpert(seed=seed),
            GlobalContextExpert(seed=seed),
        ]
    raise ValueError(f"unknown expert kind {kind!r}")


def build_system(
    spec: AblationSpec, seed: int = 0, router: RouterConfig | None = None
) -> TrainedSystem:
    config = SystemConfig(
        name=f"{spec.key}: {spec.label}",
        fusion=spec.fusion,
        calibrate_experts=spec.calibrate_experts,
        # B0 is the uncalibrated floor: calibrating the fused output would
        # defeat the purpose of having a floor at all.
        calibrate_fused=spec.calibrate_experts,
        router=router or RouterConfig(),
        seed=seed,
    )
    return TrainedSystem(config, _build_experts(spec.experts, seed))


def run_ablations(
    table: CaseTable,
    splits: dict[str, np.ndarray],
    specs: tuple[AblationSpec, ...] = ABLATIONS,
    seed: int = 0,
    target_coverage: float = 0.8,
    router: RouterConfig | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Fit and evaluate every ablation on identical splits.

    All rows share one set of splits so differences between them cannot be
    explained by a luckier partition.
    """
    from .evaluate import evaluate_system

    results: dict[str, dict] = {}
    for spec in specs:
        if verbose:
            print(f"  [{spec.key}] {spec.label} ... ", end="", flush=True)
        system = build_system(spec, seed=seed, router=router)
        system.fit(
            table,
            splits["train"],
            splits["calib"],
            target_coverage=target_coverage,
        )
        res = evaluate_system(system, table, splits["test"], target_coverage=target_coverage)
        res["spec"] = {
            "key": spec.key,
            "label": spec.label,
            "isolates": spec.isolates,
            "fusion": spec.fusion,
        }
        res["fit_log"] = system.fit_log
        results[spec.key] = res
        if verbose:
            print(
                f"acc={res['overall']['case_accuracy']:.3f} "
                f"AURC={res['overall']['aurc']:.4f}"
            )
    return results

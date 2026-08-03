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
        label="Sampling-based uncertainty (MC dropout)",
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


# What each row is actually running, per expert family. The generic labels
# ("Single expert") would otherwise hide that B0-B3 are a CNN under --experts
# torch and an MLP under --experts surrogate, and that B2 is real MC dropout in
# one case and bagging in the other.
_TORCH_LABELS: dict[str, str] = {
    "B0": "Single CNN, raw softmax",
    "B1": "Single CNN + temperature scaling",
    "B2": "MC dropout (dropout active at inference)",
    "B3": "Homogeneous 3x CNN deep ensemble",
    "B4": "Radiomics + CNN + ViT, uniform fusion",
    "B5": "Radiomics + CNN + ViT, confidence-weighted",
    "B6": "Radiomics + CNN + ViT, quality-routed (ours)",
}
_SURROGATE_SUFFIX = {
    "B0": " [MLP surrogate]",
    "B1": " [MLP surrogate]",
    "B2": " [bagging surrogate, not MC dropout]",
    "B3": " [MLP surrogate]",
    "B4": " [MLP surrogates]",
    "B5": " [MLP surrogates]",
    "B6": " [MLP surrogates]",
}


def _label(spec: "AblationSpec", use_torch: bool) -> str:
    if use_torch:
        return _TORCH_LABELS.get(spec.key, spec.label)
    return spec.label + _SURROGATE_SUFFIX.get(spec.key, "")


def _build_experts(kind: str, seed: int, cache=None, torch_config=None) -> list:
    """Construct the experts for one ablation row.

    With `cache` supplied, the CNN/ViT/MC-dropout rows use the real torch
    backbones reading pixels from the slice cache. Without it they fall back to
    the scikit-learn surrogates, so the grid still runs on a machine with no
    torch — but the run log records which was used, because the two are not
    interchangeable as evidence.
    """
    use_torch = cache is not None
    if use_torch:
        from ..experts.torch_experts import CNNExpert, MCDropoutExpert, ViTExpert

    if kind == "single":
        return [CNNExpert(cache, torch_config, seed=seed)] if use_torch else [
            LocalPatternExpert(seed=seed)
        ]
    if kind == "dropout":
        return [MCDropoutExpert(cache, torch_config, seed=seed)] if use_torch else [
            DropoutEnsembleExpert(seed=seed, n_samples=8)
        ]
    if kind == "homogeneous":
        if use_torch:
            # Same architecture, different seeds -- the deep-ensemble control.
            return [
                _TorchEnsemble(
                    [CNNExpert(cache, torch_config, seed=seed + 1000 * m) for m in range(3)]
                )
            ]
        return [HomogeneousEnsemble(n_members=3, seed=seed)]
    if kind == "heterogeneous":
        if use_torch:
            return [
                TextureRadiomicsExpert(seed=seed),
                CNNExpert(cache, torch_config, seed=seed),
                ViTExpert(cache, torch_config, seed=seed),
            ]
        return [
            TextureRadiomicsExpert(seed=seed),
            LocalPatternExpert(seed=seed),
            GlobalContextExpert(seed=seed),
        ]
    raise ValueError(f"unknown expert kind {kind!r}")


class _TorchEnsemble:
    """Averages several torch experts into one, for the B3 deep-ensemble row."""

    def __init__(self, members: list):
        self.members = members
        self.name = "deep_ensemble"
        self.view = members[0].view

    def fit(self, x, labels):
        for m in self.members:
            m.fit(x, labels)
        return self

    def predict_proba(self, x):
        from ..schema import FINDING_SCHEMA, validate_probs
        from ..experts.base import PROB_FLOOR

        acc = None
        for m in self.members:
            probs = m.predict_proba(x)
            acc = probs if acc is None else [a + p for a, p in zip(acc, probs)]
        out = []
        for p in acc:
            p = np.maximum(p / len(self.members), PROB_FLOOR)
            out.append(p / p.sum(axis=1, keepdims=True))
        validate_probs(out, context="deep_ensemble")
        return out


def build_system(
    spec: AblationSpec,
    seed: int = 0,
    router: RouterConfig | None = None,
    cache=None,
    torch_config=None,
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
    return TrainedSystem(
        config, _build_experts(spec.experts, seed, cache=cache, torch_config=torch_config)
    )


def run_ablations(
    table: CaseTable,
    splits: dict[str, np.ndarray],
    specs: tuple[AblationSpec, ...] = ABLATIONS,
    seed: int = 0,
    target_coverage: float = 0.8,
    router: RouterConfig | None = None,
    verbose: bool = True,
    use_torch: bool = False,
    torch_config=None,
) -> dict[str, dict]:
    """Fit and evaluate every ablation on identical splits.

    All rows share one set of splits so differences between them cannot be
    explained by a luckier partition.
    """
    from .evaluate import evaluate_system

    cache = None
    if use_torch:
        cache = getattr(table, "slice_cache", None)
        if cache is None:
            raise ValueError(
                "use_torch=True but the table has no slice cache. Re-ingest with "
                "ingest_cases(..., slice_cache_root=...) so the CNN/ViT experts "
                "have pixels to read."
            )

    results: dict[str, dict] = {}
    for spec in specs:
        if verbose:
            print(f"  [{spec.key}] {spec.label} ... ", end="", flush=True)
        system = build_system(
            spec, seed=seed, router=router, cache=cache, torch_config=torch_config
        )
        system.fit(
            table,
            splits["train"],
            splits["calib"],
            target_coverage=target_coverage,
        )
        res = evaluate_system(system, table, splits["test"], target_coverage=target_coverage)
        res["spec"] = {
            "key": spec.key,
            "label": _label(spec, use_torch),
            "isolates": spec.isolates,
            "fusion": spec.fusion,
            # Which expert family produced this row. A surrogate B2 is bagging,
            # not MC dropout, and a surrogate B4 is MLPs, not a CNN+ViT -- the
            # label alone must never be the only place that distinction lives.
            "experts": "torch" if use_torch else "surrogate",
        }
        res["fit_log"] = system.fit_log
        results[spec.key] = res
        if verbose:
            print(
                f"acc={res['overall']['case_accuracy']:.3f} "
                f"AURC={res['overall']['aurc']:.4f}"
            )
    return results

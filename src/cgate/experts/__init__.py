"""Expert models.

Every expert, whatever its internals, exposes the same contract: fit on a
feature view, then emit one probability simplex per finding in
`cgate.schema.FINDING_SCHEMA`. That contract is what lets a gradient-boosted
radiomics model and a neural network be fused at all.
"""

from .base import Expert, MultiFindingExpert, align_proba
from .sklearn_experts import (
    EXPERT_REGISTRY,
    DropoutEnsembleExpert,
    GlobalContextExpert,
    LocalPatternExpert,
    TextureRadiomicsExpert,
    build_default_experts,
    build_homogeneous_ensemble,
)

__all__ = [
    "Expert",
    "MultiFindingExpert",
    "align_proba",
    "EXPERT_REGISTRY",
    "TextureRadiomicsExpert",
    "LocalPatternExpert",
    "GlobalContextExpert",
    "DropoutEnsembleExpert",
    "build_default_experts",
    "build_homogeneous_ensemble",
]

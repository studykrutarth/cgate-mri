"""Combining expert predictions on the common finding target.

Data layout used throughout
---------------------------
expert_probs : list of length E (experts)
               each entry is a list of length K (findings)
               each of those is an array of shape (n_cases, n_classes_j)

weights      : array of shape (n_cases, K, E), rows over E summing to 1.
               Per-case *and* per-finding, because an expert can be reliable for
               one finding and not another -- a radiomics model is decent at
               tumour burden and poor at laterality.

Fusion is a linear opinion pool, sum_e w_e * p_e. The alternative, a logarithmic
pool (weighted geometric mean), is sharper but assigns zero probability to any
class that any expert rules out, which is exactly the failure mode a safety gate
is supposed to prevent. Both are implemented; linear is the default.
"""

from __future__ import annotations

import numpy as np

from .schema import N_FINDINGS, validate_probs

_EPS = 1e-12


def _n_cases(expert_probs: list[list[np.ndarray]]) -> int:
    if not expert_probs:
        raise ValueError("no experts supplied")
    return int(expert_probs[0][0].shape[0])


def check_expert_probs(expert_probs: list[list[np.ndarray]]) -> tuple[int, int]:
    """Validate the nested structure; return (n_experts, n_cases)."""
    if not expert_probs:
        raise ValueError("no experts supplied")
    n = _n_cases(expert_probs)
    for e, probs in enumerate(expert_probs):
        validate_probs(probs, context=f"expert index {e}")
        if probs[0].shape[0] != n:
            raise ValueError(
                f"expert index {e} has {probs[0].shape[0]} cases, expected {n}"
            )
    return len(expert_probs), n


def uniform_weights(n_cases: int, n_experts: int) -> np.ndarray:
    """Equal weight to every expert -- the homogeneous-ensembling control."""
    return np.full((n_cases, N_FINDINGS, n_experts), 1.0 / n_experts)


def normalised_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy scaled to [0, 1]; 0 = certain, 1 = uniform."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    n_classes = p.shape[1]
    if n_classes <= 1:
        return np.zeros(p.shape[0])
    return h / np.log(n_classes)


def confidence_weights(
    expert_probs: list[list[np.ndarray]], beta: float = 1.0
) -> np.ndarray:
    """Weight each expert by its own certainty on each case and finding.

    This is the classic "confidence-weighted fusion" baseline. Note it is
    *self-reported* certainty: an overconfident expert earns weight it has not
    justified. That is precisely why the router below conditions on measured
    image quality instead, and why the ablation table separates the two.
    """
    n_experts, n = check_expert_probs(expert_probs)
    w = np.zeros((n, N_FINDINGS, n_experts))
    for e, probs in enumerate(expert_probs):
        for j in range(N_FINDINGS):
            w[:, j, e] = (1.0 - normalised_entropy(probs[j])) ** beta
    total = w.sum(axis=2, keepdims=True)
    # If every expert is maximally uncertain the weights collapse to zero;
    # fall back to uniform rather than dividing by zero.
    degenerate = total[:, :, 0] <= _EPS
    w[degenerate] = 1.0 / n_experts
    total = w.sum(axis=2, keepdims=True)
    return w / total


def combine(
    expert_probs: list[list[np.ndarray]],
    weights: np.ndarray,
    pool: str = "linear",
) -> list[np.ndarray]:
    """Fuse experts into one prediction per finding."""
    n_experts, n = check_expert_probs(expert_probs)
    weights = np.asarray(weights, dtype=np.float64)
    expected = (n, N_FINDINGS, n_experts)
    if weights.shape != expected:
        raise ValueError(f"weights must have shape {expected}, got {weights.shape}")
    if np.any(weights < -1e-9):
        raise ValueError("weights must be non-negative")
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("weights must sum to 1 over the expert axis")

    fused: list[np.ndarray] = []
    for j in range(N_FINDINGS):
        stack = np.stack([expert_probs[e][j] for e in range(n_experts)], axis=2)
        w = weights[:, j, :][:, None, :]
        if pool == "linear":
            f = np.sum(stack * w, axis=2)
        elif pool == "log":
            f = np.exp(np.sum(w * np.log(np.clip(stack, _EPS, None)), axis=2))
        else:
            raise ValueError(f"unknown pool {pool!r}; expected 'linear' or 'log'")
        f = np.clip(f, _EPS, None)
        fused.append(f / f.sum(axis=1, keepdims=True))

    validate_probs(fused, context=f"fused ({pool} pool)")
    return fused


def fuse_uniform(expert_probs: list[list[np.ndarray]]) -> list[np.ndarray]:
    n_experts, n = check_expert_probs(expert_probs)
    return combine(expert_probs, uniform_weights(n, n_experts))


def fuse_by_confidence(
    expert_probs: list[list[np.ndarray]], beta: float = 1.0
) -> list[np.ndarray]:
    return combine(expert_probs, confidence_weights(expert_probs, beta=beta))

"""Calibration metrics and post-hoc calibrators.

ECE alone does not evaluate a gate -- it says whether stated confidence matches
observed accuracy on average, not whether thresholding on that confidence
actually removes errors. Those questions live in `cgate.selective`. Both are
reported for every system.

Everything here is numpy-only and works on probabilities rather than logits,
because heterogeneous experts (a gradient-boosted radiomics model, a CNN) do not
share a logit space. Temperature scaling on log-probabilities is equivalent to
the usual logit formulation up to an additive constant, which softmax removes.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, n_classes), got {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]}"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= probs.shape[1]):
        raise ValueError("labels out of range for the given number of classes")
    return probs, labels


def confidence_and_correct(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 confidence and a boolean correctness indicator."""
    probs, labels = _check(probs, labels)
    if probs.shape[0] == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    pred = np.argmax(probs, axis=1)
    conf = probs[np.arange(probs.shape[0]), pred]
    return conf, pred == labels


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    _, correct = confidence_and_correct(probs, labels)
    return float(correct.mean()) if correct.size else float("nan")


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    probs, labels = _check(probs, labels)
    if probs.shape[0] == 0:
        return float("nan")
    p = np.clip(probs[np.arange(probs.shape[0]), labels], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target."""
    probs, labels = _check(probs, labels)
    if probs.shape[0] == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins."""
    conf, correct = confidence_and_correct(probs, labels)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by confidence; the first bin owns its left edge so nothing is dropped.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    n = conf.size
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        total += (cnt / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(total)


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-mass bins.

    Equal-width bins are dominated by whichever bin happens to hold most of the
    mass -- typically the top one for an overconfident model -- which understates
    miscalibration elsewhere. Equal-mass bins give every bin the same weight.

    Caveat on ties: when many cases share a confidence value, rank-based binning
    splits them by their original order, which is arbitrary. If that order
    happens to correlate with correctness, this reports non-zero ECE for a model
    that is in fact perfectly calibrated. Report it alongside `ece`, not instead
    of it, and treat a large gap between the two as a sign of a heavily-tied
    confidence distribution rather than of miscalibration.
    """
    conf, correct = confidence_and_correct(probs, labels)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(conf, kind="stable")
    conf, correct = conf[order], correct[order]
    splits = np.array_split(np.arange(n), min(n_bins, n))
    total = 0.0
    for chunk in splits:
        if chunk.size == 0:
            continue
        total += (chunk.size / n) * abs(
            float(correct[chunk].mean()) - float(conf[chunk].mean())
        )
    return float(total)


def confidence_ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """ECE computed directly from a confidence score and a correctness flag.

    Needed for the case-level gate, where "confidence" is the minimum across
    findings rather than a row of a probability matrix, so the usual
    `ece(probs, labels)` signature does not apply.
    """
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct).ravel().astype(bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    n = confidence.size
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        total += (cnt / n) * abs(float(correct[m].mean()) - float(confidence[m].mean()))
    return float(total)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, np.ndarray]:
    """Per-bin mean confidence, empirical accuracy and count, for plotting."""
    conf, correct = confidence_and_correct(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    mean_conf = np.full(n_bins, np.nan)
    mean_acc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            mean_conf[b] = float(conf[m].mean())
            mean_acc[b] = float(correct[m].mean())
    return {
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "confidence": mean_conf,
        "accuracy": mean_acc,
        "count": counts,
    }


# ---------------------------------------------------------------------------
# Post-hoc calibrators
# ---------------------------------------------------------------------------


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log p / T), computed stably."""
    logits = np.log(np.clip(probs, _EPS, None)) / max(temperature, _EPS)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / np.clip(e.sum(axis=1, keepdims=True), _EPS, None)


def _golden_section_min(f, lo: float, hi: float, tol: float = 1e-5, max_iter: int = 200):
    """Minimise a unimodal 1-D function without pulling in scipy."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., ICML 2017).

    T is fitted by minimising NLL on a held-out calibration split -- never on
    train (the model is overfitted there, so T would come out near 1) and never
    on test (that leaks). `cgate.pipeline` enforces the split.
    """

    def __init__(self, t_min: float = 0.05, t_max: float = 20.0) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.temperature_: float = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        probs, labels = _check(probs, labels)
        if probs.shape[0] == 0:
            self.temperature_ = 1.0
            return self

        # Search in log-space: T is a scale parameter, so the objective is much
        # closer to symmetric in log T than in T.
        def objective(log_t: float) -> float:
            return nll(_apply_temperature(probs, float(np.exp(log_t))), labels)

        best = _golden_section_min(
            objective, float(np.log(self.t_min)), float(np.log(self.t_max))
        )
        self.temperature_ = float(np.exp(best))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return _apply_temperature(np.asarray(probs, dtype=np.float64), self.temperature_)

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(probs, labels).transform(probs)


class MultiFindingTemperatureScaler:
    """One temperature per finding.

    Findings differ in difficulty and class count, so a single shared
    temperature would under-correct the hard ones and over-correct the easy
    ones. Fitting per finding costs one scalar each and is strictly more
    expressive.
    """

    def __init__(self) -> None:
        self.scalers_: list[TemperatureScaler] = []

    def fit(self, probs: list[np.ndarray], labels: np.ndarray) -> MultiFindingTemperatureScaler:
        self.scalers_ = []
        for j, p in enumerate(probs):
            self.scalers_.append(TemperatureScaler().fit(p, labels[:, j]))
        return self

    def transform(self, probs: list[np.ndarray]) -> list[np.ndarray]:
        if not self.scalers_:
            raise RuntimeError("MultiFindingTemperatureScaler.fit must be called first")
        if len(probs) != len(self.scalers_):
            raise ValueError(
                f"expected {len(self.scalers_)} findings, got {len(probs)}"
            )
        return [s.transform(p) for s, p in zip(self.scalers_, probs)]

    @property
    def temperatures(self) -> list[float]:
        return [s.temperature_ for s in self.scalers_]

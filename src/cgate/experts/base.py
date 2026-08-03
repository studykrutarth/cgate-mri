"""The expert contract and the class-alignment logic every expert needs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schema import CLASS_COUNTS, FINDING_SCHEMA, N_FINDINGS, validate_probs

# No expert may emit exactly zero for any class. A hard zero makes NLL infinite
# and silently poisons temperature scaling and every calibration metric, and it
# is never justified: an expert that has never seen a class has no evidence it
# is impossible, only that it is rare.
PROB_FLOOR = 1e-6


@runtime_checkable
class Expert(Protocol):
    """Minimum interface the pipeline depends on."""

    name: str
    view: str

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "Expert": ...

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]: ...


def align_proba(
    proba: np.ndarray,
    known_classes: np.ndarray,
    n_classes: int,
    floor: float = PROB_FLOOR,
) -> np.ndarray:
    """Expand a classifier's output to the finding's full class set.

    scikit-learn returns one column per class *seen during training*. If a class
    is absent from the training split -- entirely possible for a rare finding on
    a small cohort -- the returned array has too few columns and would misalign
    against the label indices, quietly scoring the wrong class as correct.

    This maps columns back to their true positions, gives unseen classes the
    floor rather than zero, and renormalises.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"expected 2-D proba, got shape {proba.shape}")
    n = proba.shape[0]
    out = np.full((n, n_classes), floor, dtype=np.float64)
    for col, cls in enumerate(np.asarray(known_classes).ravel()):
        c = int(cls)
        if not 0 <= c < n_classes:
            raise ValueError(f"classifier produced class {c}, outside [0, {n_classes})")
        out[:, c] = np.maximum(proba[:, col], floor)
    return out / out.sum(axis=1, keepdims=True)


class MultiFindingExpert:
    """Wraps K independent per-finding classifiers into one expert.

    Findings are modelled independently rather than as one joint label. A joint
    model over the product space would have 2*2*3*3*3 = 108 classes, most of
    them unobserved on a cohort of a few hundred, and its probabilities would be
    unusable for per-finding calibration. Independence costs some structure
    (laterality and burden are not truly independent) but keeps every finding's
    probability well-estimated, which is what the gate needs.
    """

    name: str = "base"
    view: str = "texture"

    def __init__(self, name: str, view: str, estimator_factory, seed: int = 0):
        self.name = name
        self.view = view
        self._factory = estimator_factory
        self.seed = seed
        self.models_: list = []
        self.classes_: list[np.ndarray] = []
        self.fitted_ = False

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "MultiFindingExpert":
        x = np.asarray(x, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 2 or labels.shape[1] != N_FINDINGS:
            raise ValueError(
                f"labels must have shape (n, {N_FINDINGS}), got {labels.shape}"
            )
        if x.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features have {x.shape[0]} rows, labels have {labels.shape[0]}"
            )

        self.models_ = []
        self.classes_ = []
        for j, spec in enumerate(FINDING_SCHEMA):
            y = labels[:, j]
            model = self._factory(self.seed + j)
            uniq = np.unique(y)
            if uniq.size < 2:
                # Degenerate finding on this split: a classifier cannot be fitted.
                # Fall back to the observed constant, recorded explicitly so the
                # run log shows it rather than the expert appearing to work.
                self.models_.append(None)
                self.classes_.append(uniq)
            else:
                model.fit(x, y)
                self.models_.append(model)
                self.classes_.append(np.asarray(model.classes_))
        self.fitted_ = True
        return self

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"expert {self.name!r} must be fitted before predicting")
        x = np.asarray(x, dtype=np.float64)
        out: list[np.ndarray] = []
        for j, spec in enumerate(FINDING_SCHEMA):
            model = self.models_[j]
            if model is None:
                # Constant finding: put almost all mass on the single observed
                # class, but not all of it -- see PROB_FLOOR.
                p = np.full((x.shape[0], spec.n_classes), PROB_FLOOR)
                observed = self.classes_[j]
                if observed.size:
                    p[:, int(observed[0])] = 1.0
                else:
                    p[:] = 1.0 / spec.n_classes
                p = p / p.sum(axis=1, keepdims=True)
            else:
                p = align_proba(
                    model.predict_proba(x), self.classes_[j], spec.n_classes
                )
            out.append(p)
        validate_probs(out, context=f"expert {self.name!r}")
        return out

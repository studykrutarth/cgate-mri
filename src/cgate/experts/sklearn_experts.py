"""Concrete experts built on scikit-learn.

Mapping to the architecture on the deck:

  TextureRadiomicsExpert -- gradient-boosted trees on first-order + gradient
      statistics. This is the PyRadiomics + XGBoost expert; scikit-learn's
      HistGradientBoosting is the same algorithm family and avoids a compiled
      dependency, and swapping in xgboost.XGBClassifier is a one-line change.

  LocalPatternExpert  -- MLP on the block-mean grid. Stands in for the CNN.
  GlobalContextExpert -- MLP on axis marginals and moments. Stands in for the ViT.

The last two are explicitly *surrogates*: they consume hand-built views designed
to mimic what a CNN and a ViT respectively attend to. They exist so the router,
fusion, calibration and gate can be developed and validated end-to-end before
the GPU work lands, and so this repository runs on a machine with no torch. They
are not CNNs and nothing here claims they are. Replacing them with real
backbones means implementing `fit`/`predict_proba` in cgate/experts/torch_experts.py;
no other module changes.
"""

from __future__ import annotations

import numpy as np

from ..schema import FINDING_SCHEMA, N_FINDINGS, validate_probs
from .base import PROB_FLOOR, MultiFindingExpert, align_proba


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for the built-in experts. "
            "Install it with:  pip install -e '.[sci]'"
        ) from exc


def _boosted_factory(seed: int):
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=3,
        # Small cohorts overfit trees fast; these two are the main guards.
        min_samples_leaf=8,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed,
    )


def _mlp_factory(hidden: tuple[int, ...], alpha: float, seed: int):
    _require_sklearn()
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Scaling is not optional for an MLP here: the views mix raw intensities
    # with normalised ratios, spanning several orders of magnitude.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            max_iter=1200,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.15,
            random_state=seed,
        ),
    )


class TextureRadiomicsExpert(MultiFindingExpert):
    """Gradient-boosted trees on radiomics-style texture features."""

    def __init__(self, seed: int = 0, name: str = "radiomics"):
        super().__init__(name=name, view="texture", estimator_factory=_boosted_factory, seed=seed)


class LocalPatternExpert(MultiFindingExpert):
    """MLP on the block-mean grid (CNN surrogate)."""

    def __init__(self, seed: int = 0, name: str = "cnn_surrogate"):
        super().__init__(
            name=name,
            view="local",
            estimator_factory=lambda s: _mlp_factory((64, 32), 1e-2, s),
            seed=seed,
        )


class GlobalContextExpert(MultiFindingExpert):
    """MLP on global marginals and moments (ViT surrogate)."""

    def __init__(self, seed: int = 0, name: str = "vit_surrogate"):
        super().__init__(
            name=name,
            view="global",
            estimator_factory=lambda s: _mlp_factory((48,), 3e-2, s),
            seed=seed,
        )


class DropoutEnsembleExpert:
    """Sampling-based uncertainty baseline (MC-dropout surrogate).

    True MC dropout needs a network with dropout layers kept active at inference,
    which requires torch. This approximates the same idea with the tools
    available: M models, each fitted on a bootstrap resample with a random
    subset of input features masked out, then averaged. Like MC dropout it
    produces predictive uncertainty by sampling a distribution over models
    rather than reading a single softmax.

    It is labelled a surrogate everywhere it is reported. When torch is
    installed, the real MC-dropout baseline replaces it and the ablation table
    keeps the same row.
    """

    def __init__(
        self,
        base: str = "local",
        n_samples: int = 10,
        drop_rate: float = 0.25,
        seed: int = 0,
        name: str = "mc_dropout_surrogate",
    ):
        self.name = name
        self.view = base
        self.n_samples = n_samples
        self.drop_rate = drop_rate
        self.seed = seed
        self.members_: list[MultiFindingExpert] = []
        self.masks_: list[np.ndarray] = []
        self.fitted_ = False

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "DropoutEnsembleExpert":
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        n, d = x.shape
        self.members_ = []
        self.masks_ = []
        for m in range(self.n_samples):
            keep = rng.random(d) >= self.drop_rate
            if not keep.any():
                keep[rng.integers(0, d)] = True
            idx = rng.integers(0, n, size=n)
            member = LocalPatternExpert(seed=self.seed + 100 * m, name=f"{self.name}_m{m}")
            member.fit(x[idx][:, keep], labels[idx])
            self.members_.append(member)
            self.masks_.append(keep)
        self.fitted_ = True
        return self

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"expert {self.name!r} must be fitted before predicting")
        x = np.asarray(x, dtype=np.float64)
        acc = [
            np.zeros((x.shape[0], spec.n_classes), dtype=np.float64)
            for spec in FINDING_SCHEMA
        ]
        for member, keep in zip(self.members_, self.masks_):
            probs = member.predict_proba(x[:, keep])
            for j in range(N_FINDINGS):
                acc[j] += probs[j]
        out = []
        for j in range(N_FINDINGS):
            p = acc[j] / len(self.members_)
            p = np.maximum(p, PROB_FLOOR)
            out.append(p / p.sum(axis=1, keepdims=True))
        validate_probs(out, context=f"expert {self.name!r}")
        return out


class HomogeneousEnsemble:
    """M copies of one architecture, differing only in seed -- a deep ensemble.

    This is the control that matters most. Deep ensembles are a strong
    calibration baseline in their own right (Lakshminarayanan et al., 2017), so
    if the heterogeneous router cannot beat *this*, the reported gain is coming
    from ensembling and not from routing, and the novelty claim fails. Keeping
    the architecture fixed and varying only the seed isolates that.
    """

    def __init__(
        self,
        member_cls=LocalPatternExpert,
        n_members: int = 3,
        seed: int = 0,
        name: str = "deep_ensemble",
    ):
        self.name = name
        self.member_cls = member_cls
        self.n_members = n_members
        self.seed = seed
        self.members_: list[MultiFindingExpert] = []
        self.view = member_cls().view
        self.fitted_ = False

    def fit(self, x: np.ndarray, labels: np.ndarray) -> "HomogeneousEnsemble":
        self.members_ = []
        for m in range(self.n_members):
            member = self.member_cls(seed=self.seed + 1000 * m)
            member.fit(x, labels)
            self.members_.append(member)
        self.fitted_ = True
        return self

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"expert {self.name!r} must be fitted before predicting")
        acc = [
            np.zeros((np.asarray(x).shape[0], spec.n_classes), dtype=np.float64)
            for spec in FINDING_SCHEMA
        ]
        for member in self.members_:
            probs = member.predict_proba(x)
            for j in range(N_FINDINGS):
                acc[j] += probs[j]
        out = []
        for j in range(N_FINDINGS):
            p = np.maximum(acc[j] / len(self.members_), PROB_FLOOR)
            out.append(p / p.sum(axis=1, keepdims=True))
        validate_probs(out, context=f"expert {self.name!r}")
        return out


EXPERT_REGISTRY = {
    "radiomics": TextureRadiomicsExpert,
    "cnn_surrogate": LocalPatternExpert,
    "vit_surrogate": GlobalContextExpert,
}


def build_default_experts(seed: int = 0) -> list:
    """The three heterogeneous experts used by the full system."""
    return [
        TextureRadiomicsExpert(seed=seed),
        LocalPatternExpert(seed=seed),
        GlobalContextExpert(seed=seed),
    ]


def build_homogeneous_ensemble(n_members: int = 3, seed: int = 0) -> HomogeneousEnsemble:
    return HomogeneousEnsemble(n_members=n_members, seed=seed)

"""Smoke tests for the real CNN/ViT experts.

Deliberately tiny: a small architecture, no pretrained download, few epochs.
The point is to prove the contract holds end to end -- the network trains, and
what comes out is a valid per-finding probability simplex the rest of the
pipeline can consume. Accuracy on eight phantoms is meaningless and is not
asserted.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from cgate.data.slices import SliceCache, SliceCacheConfig  # noqa: E402
from cgate.data.synthetic import SyntheticConfig, build_synthetic_dataset  # noqa: E402
from cgate.experts.torch_experts import (  # noqa: E402
    CNNExpert,
    MCDropoutExpert,
    TorchExpertConfig,
    ViTExpert,
)
from cgate.schema import FINDING_SCHEMA, N_FINDINGS, validate_probs  # noqa: E402

SHAPE = (24, 24, 16)
N_CASES = 8


def _tiny_config(**kw):
    base = dict(
        arch="resnet18",
        img_size=64,
        pretrained=False,  # no network access, and weights are irrelevant here
        epochs=1,
        batch_size=4,
        val_fraction=0.25,
        verbose=False,
    )
    base.update(kw)
    return TorchExpertConfig(**base)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("slices")
    cfg = SyntheticConfig(shape=SHAPE, n_cases=N_CASES, seed=0)
    return SliceCache.build(
        build_synthetic_dataset(cfg), root, SliceCacheConfig(n_slices=2)
    )


@pytest.fixture(scope="module")
def labels():
    rng = np.random.default_rng(0)
    return np.stack(
        [rng.integers(0, s.n_classes, size=N_CASES) for s in FINDING_SCHEMA], axis=1
    )


def _rows(cache):
    return np.arange(len(cache)).reshape(-1, 1)


def test_cache_has_four_modalities(cache):
    assert cache.n_modalities == 4
    assert cache.n_slices == 2
    assert cache.get(0).shape == (2, 4, *SHAPE[:2])


@pytest.mark.slow
def test_cnn_expert_trains_and_emits_valid_simplexes(cache, labels):
    expert = CNNExpert(cache, _tiny_config())
    expert.fit(_rows(cache), labels)
    probs = expert.predict_proba(_rows(cache))

    # The contract: K arrays, right widths, rows summing to 1.
    validate_probs(probs, context="cnn smoke test")
    assert len(probs) == N_FINDINGS
    for spec, p in zip(FINDING_SCHEMA, probs):
        assert p.shape == (N_CASES, spec.n_classes)
        assert np.all(p > 0)  # PROB_FLOOR: never a hard zero
    assert expert.history_, "training recorded no epochs"


@pytest.mark.slow
def test_vit_expert_emits_valid_simplexes(cache, labels):
    # ViT needs its native 224 input; keep everything else minimal.
    expert = ViTExpert(
        cache, _tiny_config(arch="vit_tiny_patch16_224", img_size=224, batch_size=2)
    )
    expert.fit(_rows(cache), labels)
    probs = expert.predict_proba(_rows(cache))
    validate_probs(probs, context="vit smoke test")


@pytest.mark.slow
def test_mc_dropout_is_actually_stochastic(cache, labels):
    """Two MC passes must differ, or dropout was not left active."""
    expert = MCDropoutExpert(cache, _tiny_config(dropout=0.5), n_passes=4)
    expert.fit(_rows(cache), labels)

    torch.manual_seed(1)
    a = expert.predict_proba(_rows(cache))
    torch.manual_seed(2)
    b = expert.predict_proba(_rows(cache))

    validate_probs(a, context="mc dropout")
    differs = any(not np.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    assert differs, "MC dropout produced identical output; dropout was not active"


@pytest.mark.slow
def test_deterministic_expert_is_not_stochastic(cache, labels):
    """The plain CNN must be repeatable -- only B2 is meant to sample."""
    expert = CNNExpert(cache, _tiny_config())
    expert.fit(_rows(cache), labels)
    a = expert.predict_proba(_rows(cache))
    b = expert.predict_proba(_rows(cache))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-6)


def test_expert_rejects_a_feature_matrix_that_is_not_cache_rows(cache, labels):
    expert = CNNExpert(cache, _tiny_config())
    with pytest.raises(ValueError, match="slice-cache rows"):
        expert.fit(np.zeros((N_CASES, 12)), labels)


def test_predict_before_fit_raises(cache):
    expert = CNNExpert(cache, _tiny_config())
    with pytest.raises(RuntimeError, match="must be fitted"):
        expert.predict_proba(_rows(cache))

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
    # The ViT reads vit_arch/vit_img_size, not arch/img_size.
    expert = ViTExpert(
        cache,
        _tiny_config(vit_arch="vit_tiny_patch16_224", vit_img_size=224, batch_size=2),
    )
    assert expert.config.arch == "vit_tiny_patch16_224"
    expert.fit(_rows(cache), labels)
    probs = expert.predict_proba(_rows(cache))
    validate_probs(probs, context="vit smoke test")


def test_vit_keeps_its_own_architecture_when_arch_is_a_cnn(cache):
    """--arch resnet50 must not turn the ViT into a second ResNet.

    It silently did, which left B4-B6 fusing radiomics with two copies of one
    architecture while still being reported as CNN+ViT — the heterogeneity those
    rows exist to isolate, gone with nothing in the output to show it.
    """
    shared = _tiny_config(arch="resnet50", vit_arch="vit_small_patch16_224")
    cnn, vit = CNNExpert(cache, shared), ViTExpert(cache, shared)

    assert cnn.config.arch == "resnet50"
    assert vit.config.arch == "vit_small_patch16_224"
    assert cnn.config.arch != vit.config.arch


def test_experts_do_not_mutate_the_shared_config(cache):
    """Every expert in a row is handed the same config instance.

    Assigning to it wrote through to the others, so the last expert constructed
    decided the seed for all of them — which reduced B3's three-seed deep
    ensemble to three identical models.
    """
    shared = _tiny_config(arch="resnet18", seed=0)
    members = [CNNExpert(cache, shared, seed=1000 * m) for m in range(3)]
    ViTExpert(cache, shared, seed=7)

    assert shared.seed == 0, "the shared config was mutated"
    assert shared.arch == "resnet18", "the shared config was mutated"
    assert [m.config.seed for m in members] == [0, 1000, 2000]


def test_mc_dropout_keeps_its_own_rate_when_a_shared_config_is_passed(cache):
    """B2 is the row that isolates sampling-based uncertainty.

    `config or TorchExpertConfig(dropout=0.5)` only reached 0.5 when no config
    was supplied, and run.py always supplies one — so B2 trained at the shared
    0.3, which is too small for the passes to move apart, and came out
    indistinguishable from B0 with nothing in the run log naming the rate.
    """
    shared = _tiny_config(dropout=0.3, mc_dropout=0.5)
    cnn, mcd = CNNExpert(cache, shared), MCDropoutExpert(cache, shared)

    assert cnn.config.dropout == 0.3
    assert mcd.config.dropout == 0.5
    assert shared.dropout == 0.3, "the shared config was mutated"


def test_amp_defaults_follow_the_device(cache):
    """On for CUDA, off for CPU, and overridable either way.

    Left on for CPU it would cost accuracy for no memory saving; left off for
    CUDA it would OOM a 4 GB card on the stock configuration.
    """
    import torch

    expert = CNNExpert(cache, _tiny_config())
    expert.device = torch.device("cpu")
    assert expert._amp_enabled() is False
    expert.device = torch.device("cuda")
    assert expert._amp_enabled() is True

    forced_off = CNNExpert(cache, _tiny_config(amp=False))
    forced_off.device = torch.device("cuda")
    assert forced_off._amp_enabled() is False

    forced_on = CNNExpert(cache, _tiny_config(amp=True))
    forced_on.device = torch.device("cpu")
    assert forced_on._amp_enabled() is True


def test_disabled_autocast_leaves_the_cpu_path_alone(cache, labels):
    """Every result recorded so far came from the CPU path.

    A disabled autocast and a disabled GradScaler must both be pass-throughs,
    or introducing AMP would silently move numbers that predate it.
    """
    import torch

    a = CNNExpert(cache, _tiny_config(amp=False, epochs=1), seed=3)
    a.fit(_rows(cache), labels)
    b = CNNExpert(cache, _tiny_config(amp=None, epochs=1), seed=3)
    b.fit(_rows(cache), labels)
    assert b.device == torch.device("cpu"), "test assumes no CUDA here"

    for pa, pb in zip(a.predict_proba(_rows(cache)), b.predict_proba(_rows(cache))):
        np.testing.assert_array_equal(pa, pb)


def test_out_of_memory_names_the_flags_that_fix_it(cache, labels, monkeypatch):
    """Torch reports the failed allocation, not which knob to turn."""
    expert = CNNExpert(cache, _tiny_config(epochs=1))

    def boom(*_a, **_k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(expert, "_fit", boom)
    with pytest.raises(RuntimeError, match="--batch-size") as exc:
        expert.fit(_rows(cache), labels)
    assert "--img-size" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_an_unrelated_failure_is_not_relabelled_as_memory(cache, labels, monkeypatch):
    expert = CNNExpert(cache, _tiny_config(epochs=1))

    def boom(*_a, **_k):
        raise ValueError("features and labels disagree on case count")

    monkeypatch.setattr(expert, "_fit", boom)
    with pytest.raises(ValueError, match="disagree on case count"):
        expert.fit(_rows(cache), labels)


def test_batch_size_reaches_the_expert_from_the_cli():
    from cgate.experiments.run import build_parser

    args = build_parser().parse_args(["--experts", "torch", "--batch-size", "4", "--no-amp"])
    assert args.batch_size == 4
    assert args.amp is False
    assert build_parser().parse_args([]).amp is None
    assert build_parser().parse_args(["--amp"]).amp is True


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


@pytest.mark.slow
def test_b3_ensemble_members_are_actually_different_models(cache, labels):
    """B3 exists to show what ensembling *alone* buys, which needs real diversity.

    With the shared config being mutated, all three members ended up on one seed:
    identical initialisation, identical batch order, identical models. The row
    still reported as a 3x deep ensemble while being a single model.
    """
    from cgate.experiments.ablations import _build_experts

    (ensemble,) = _build_experts(
        "homogeneous", seed=0, cache=cache, torch_config=_tiny_config()
    )
    assert [m.config.seed for m in ensemble.members] == [0, 1000, 2000]

    ensemble.fit(_rows(cache), labels)
    first, second = (m.predict_proba(_rows(cache)) for m in ensemble.members[:2])
    assert any(
        not np.allclose(a, b, atol=1e-6) for a, b in zip(first, second)
    ), "two ensemble members are identical; the deep-ensemble row is a single model"


def test_expert_rejects_a_feature_matrix_that_is_not_cache_rows(cache, labels):
    expert = CNNExpert(cache, _tiny_config())
    with pytest.raises(ValueError, match="slice-cache rows"):
        expert.fit(np.zeros((N_CASES, 12)), labels)


def test_predict_before_fit_raises(cache):
    expert = CNNExpert(cache, _tiny_config())
    with pytest.raises(RuntimeError, match="must be fitted"):
        expert.predict_proba(_rows(cache))

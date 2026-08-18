"""Real CNN and ViT experts on 2-D slices.

These replace the MLP surrogates in `sklearn_experts.py`. They obey the same
contract (fit / predict_proba over `cgate.schema.FINDING_SCHEMA`), so the
router, calibration, gate and the B0-B6 grid need no changes.

Shape of the model
------------------
One shared backbone, five linear heads — one per finding, sized to that
finding's class count (2, 2, 3, 3, 3) — trained on the summed cross-entropy.
Modelling the findings jointly instead would mean 2*2*3*3*3 = 108 classes, most
unobserved on a cohort of this size.

2.5-D, not 3-D
--------------
Training is per-slice: each cached slice becomes a sample carrying its case's
label, and predictions are averaged back to case level at inference. A full 3-D
ViT on 240x240x155x4 does not fit the available compute; that exclusion is a
design decision, not an oversight. Slices of a case never straddle a split
because splitting happens at patient level upstream.

Why `timm` and pretrained weights
---------------------------------
~1200 cases is far too few to train a ViT from scratch. `in_chans=4` makes timm
adapt the pretrained 3-channel stem to the four MRI sequences by replicating
weights, which is both standard and much better than a random stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..schema import FINDING_SCHEMA, N_FINDINGS, validate_probs
from .base import PROB_FLOOR


_THREADS_LIMITED = False


def _require_torch(num_threads: int | None = 1):
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required for the CNN/ViT experts. Install with: "
            "pip install -e '.[torch]'"
        ) from exc

    # Windows OpenMP clash. scikit-learn/scipy load Intel MKL's OpenMP runtime
    # and torch loads its own; when both spawn thread teams in one process,
    # torch's backward pass can die with an access violation. Since this
    # pipeline runs a scikit-learn radiomics expert and a torch CNN side by
    # side, that combination is the normal case, not an edge case.
    #
    # Capping torch's thread count avoids it without the global
    # KMP_DUPLICATE_LIB_OK=TRUE workaround, which Intel documents as able to
    # crash or silently produce wrong results. Only torch is slowed; numpy and
    # scikit-learn keep their threads.
    global _THREADS_LIMITED
    if num_threads is not None and not _THREADS_LIMITED:
        import torch

        torch.set_num_threads(int(num_threads))
        _THREADS_LIMITED = True


def _require_timm():
    try:
        import timm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "timm is required for the CNN/ViT experts. Install with: "
            "pip install -e '.[torch]'"
        ) from exc


@dataclass
class TorchExpertConfig:
    # `arch`/`img_size` drive the CNN and MC-dropout experts. The ViT expert
    # carries its own pair, because one shared `arch` field meant that passing
    # --arch resnet50 silently rebuilt the ViT as a second ResNet: B4-B6 then
    # fused radiomics with two copies of one architecture, and the heterogeneity
    # those rows exist to isolate was gone with nothing in the output to show it.
    arch: str = "resnet50"
    img_size: int = 224
    vit_arch: str = "vit_small_patch16_224"
    vit_img_size: int = 224
    pretrained: bool = True
    dropout: float = 0.3
    # The MC-dropout expert carries its own rate for the same reason the ViT
    # carries its own arch. `config or TorchExpertConfig(dropout=0.5)` only
    # reached 0.5 when no config was passed at all, and run.py always passes
    # one -- so B2, the row that exists to isolate sampling-based uncertainty,
    # ran at the shared 0.3 and came out indistinguishable from B0.
    mc_dropout: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    # Mixed precision. None means "on for CUDA, off for CPU", which is the only
    # sensible default: it roughly halves activation memory -- the difference
    # between a 4 GB card running the ResNet-50/224px/batch-32 configuration and
    # OOMing on it -- while on CPU it buys nothing and costs accuracy.
    #
    # It does change the numerics, so it is recorded in the run log. Comparing a
    # row trained with AMP against one trained without is comparing two
    # different experiments.
    amp: bool | None = None
    epochs: int = 12
    patience: int = 4
    val_fraction: float = 0.15
    num_workers: int = 0
    device: str | None = None
    seed: int = 0
    # Vertical (anterior-posterior) flips only. A left-right flip would mirror
    # the hemisphere and silently invert the `laterality` label while the
    # target stayed unchanged -- corrupting the one finding that depends on
    # spatial orientation, invisibly.
    augment: bool = True
    verbose: bool = False
    # Capped to dodge the Windows OpenMP clash with scikit-learn — see
    # _require_torch. Set to None to leave torch's threading untouched.
    num_threads: int | None = 1


def build_backbone(config: TorchExpertConfig, in_chans: int):
    """Shared encoder + one head per finding."""
    _require_torch()
    _require_timm()
    import timm
    import torch.nn as nn

    def create_encoder():
        kwargs = dict(
            pretrained=config.pretrained,
            in_chans=in_chans,
            num_classes=0,  # feature extractor
        )
        try:
            # ViTs need img_size to interpolate their position embeddings when
            # run at anything other than the size they were pretrained at.
            return timm.create_model(config.arch, img_size=config.img_size, **kwargs)
        except TypeError:
            # Convolutional backbones do not take img_size and are size-agnostic.
            return timm.create_model(config.arch, **kwargs)

    class MultiHeadBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = create_encoder()
            n_feat = self.encoder.num_features
            self.dropout = nn.Dropout(config.dropout)
            self.heads = nn.ModuleList(
                [nn.Linear(n_feat, spec.n_classes) for spec in FINDING_SCHEMA]
            )

        def forward(self, x):
            feats = self.dropout(self.encoder(x))
            return [head(feats) for head in self.heads]

    return MultiHeadBackbone()


class TorchSliceExpert:
    """Base class for slice-based torch experts.

    `fit`/`predict_proba` receive a feature matrix whose single column holds
    row indices into a `SliceCache` — that is how pixels reach an interface
    otherwise built around tabular views, without changing the interface.
    """

    view = "volume"

    def __init__(
        self,
        cache,
        config: TorchExpertConfig | None = None,
        name: str = "cnn",
    ):
        self.cache = cache
        self.config = config or TorchExpertConfig()
        self.name = name
        self.model = None
        self.device = None
        self.history_: list[dict] = []
        self.fitted_ = False

    # -- helpers -----------------------------------------------------------
    def _rows(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(
                f"{self.name}: expected an (n, 1) matrix of slice-cache rows, "
                f"got shape {x.shape}. Did the pipeline build a SliceCache?"
            )
        return x[:, 0].astype(np.int64)

    def _resolve_device(self):
        import torch

        if self.config.device:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _amp_enabled(self) -> bool:
        if self.config.amp is not None:
            return bool(self.config.amp)
        return self.device is not None and self.device.type == "cuda"

    def _autocast(self):
        """Mixed precision around the forward pass only.

        Disabled it is a no-op, so the CPU path stays numerically identical to
        what it was before AMP existed -- which matters, because every result
        recorded so far came from it.
        """
        import torch

        return torch.amp.autocast(
            device_type=self.device.type, enabled=self._amp_enabled()
        )

    def _batch_tensor(self, images: np.ndarray):
        """(B, C, H, W) float32 on device, resized to the backbone's input."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(images)).float()
        if t.shape[-1] != self.config.img_size or t.shape[-2] != self.config.img_size:
            t = F.interpolate(
                t,
                size=(self.config.img_size, self.config.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return t.to(self.device)

    # -- training ----------------------------------------------------------
    def fit(self, x: np.ndarray, labels: np.ndarray) -> "TorchSliceExpert":
        """Train, translating an out-of-memory death into what to do about it.

        Torch reports the allocation it failed to make, which does not tell you
        which knob to turn. On a 4 GB card the stock ResNet-50 / 224px /
        batch-32 configuration does not fit, and that is a configuration
        question, not a bug.
        """
        try:
            return self._fit(x, labels)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            raise RuntimeError(
                f"expert {self.name!r} ran out of memory on {self.device} with "
                f"arch={self.config.arch} img_size={self.config.img_size} "
                f"batch_size={self.config.batch_size} amp={self._amp_enabled()}.\n"
                f"In rough order of cost to the result: lower --batch-size "
                f"(changes optimisation slightly), enable --amp if it is off, "
                f"lower --img-size (discards detail), or pick a smaller --arch. "
                f"--device cpu trades the memory problem for a much slower run."
            ) from exc

    def _fit(self, x: np.ndarray, labels: np.ndarray) -> "TorchSliceExpert":
        cfg = self.config
        _require_torch(cfg.num_threads)
        import torch
        import torch.nn as nn

        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        rows = self._rows(x)
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape[0] != rows.shape[0]:
            raise ValueError("features and labels disagree on case count")

        self.device = self._resolve_device()
        self.model = build_backbone(cfg, in_chans=self.cache.n_modalities).to(self.device)

        # Hold out whole cases for early stopping. Splitting by slice would put
        # neighbouring slices of one tumour on both sides.
        n = len(rows)
        perm = rng.permutation(n)
        n_val = max(1, int(round(cfg.val_fraction * n))) if n > 4 else 0
        val_cases, train_cases = perm[:n_val], perm[n_val:]
        if train_cases.size == 0:
            train_cases, val_cases = perm, np.array([], dtype=int)

        n_slices = self.cache.n_slices
        # Each (case, slice) pair is one training sample.
        train_pairs = np.array(
            [(c, s) for c in train_cases for s in range(n_slices)], dtype=np.int64
        )

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        # A disabled scaler passes gradients through untouched.
        scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp_enabled())

        best_val = np.inf
        best_state = None
        bad = 0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            rng.shuffle(train_pairs)
            total, seen = 0.0, 0

            for start in range(0, len(train_pairs), cfg.batch_size):
                chunk = train_pairs[start : start + cfg.batch_size]
                imgs = np.stack(
                    [self.cache.get(rows[c])[s] for c, s in chunk], axis=0
                )
                if cfg.augment:
                    # Anterior-posterior flips only -- see TorchExpertConfig.
                    flip = rng.random(len(chunk)) < 0.5
                    imgs[flip] = imgs[flip][..., ::-1, :]
                batch = self._batch_tensor(imgs)
                y = torch.from_numpy(labels[chunk[:, 0]]).to(self.device)

                with self._autocast():
                    logits = self.model(batch)
                # Losses in fp32: summing five cross-entropies in fp16 loses
                # small gradients that matter for the rarer finding classes.
                loss = sum(
                    loss_fn(logits[j].float(), y[:, j]) for j in range(N_FINDINGS)
                )

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total += float(loss.item()) * len(chunk)
                seen += len(chunk)

            train_loss = total / max(seen, 1)
            val_loss = (
                self._evaluate_loss(rows, labels, val_cases)
                if val_cases.size
                else train_loss
            )
            self.history_.append(
                {"epoch": epoch, "train": train_loss, "val": val_loss}
            )
            if cfg.verbose:
                print(f"[{self.name}] epoch {epoch} train={train_loss:.4f} val={val_loss:.4f}")

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fitted_ = True
        return self

    def _evaluate_loss(self, rows, labels, cases) -> float:
        import torch
        import torch.nn as nn

        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        n_slices = self.cache.n_slices
        pairs = np.array([(c, s) for c in cases for s in range(n_slices)], dtype=np.int64)
        total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(pairs), self.config.batch_size):
                chunk = pairs[start : start + self.config.batch_size]
                imgs = np.stack([self.cache.get(rows[c])[s] for c, s in chunk], axis=0)
                batch = self._batch_tensor(imgs)
                y = torch.from_numpy(labels[chunk[:, 0]]).to(self.device)
                with self._autocast():
                    logits = self.model(batch)
                loss = sum(
                    loss_fn(logits[j].float(), y[:, j]) for j in range(N_FINDINGS)
                )
                total += float(loss.item()) * len(chunk)
                seen += len(chunk)
        return total / max(seen, 1)

    # -- inference ---------------------------------------------------------
    def _forward_probs(self, rows: np.ndarray, mc_passes: int = 1) -> list[np.ndarray]:
        """Case-level probabilities, averaged over slices (and MC passes)."""
        import torch

        n = len(rows)
        n_slices = self.cache.n_slices
        acc = [np.zeros((n, spec.n_classes), dtype=np.float64) for spec in FINDING_SCHEMA]

        self.model.eval()
        if mc_passes > 1:
            # Real MC dropout: keep dropout stochastic while everything else
            # (notably BatchNorm) stays in eval mode.
            for module in self.model.modules():
                if module.__class__.__name__.startswith("Dropout"):
                    module.train()

        with torch.no_grad():
            for _ in range(mc_passes):
                for start in range(0, n, self.config.batch_size):
                    case_idx = np.arange(start, min(start + self.config.batch_size, n))
                    for s in range(n_slices):
                        imgs = np.stack([self.cache.get(rows[c])[s] for c in case_idx], axis=0)
                        batch = self._batch_tensor(imgs)
                        with self._autocast():
                            logits = self.model(batch)
                        for j in range(N_FINDINGS):
                            p = torch.softmax(logits[j].float(), dim=1).cpu().numpy()
                            acc[j][case_idx] += p

        denom = n_slices * mc_passes
        out = []
        for j in range(N_FINDINGS):
            p = np.maximum(acc[j] / denom, PROB_FLOOR)
            out.append(p / p.sum(axis=1, keepdims=True))
        validate_probs(out, context=f"expert {self.name!r}")
        return out

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"expert {self.name!r} must be fitted before predicting")
        return self._forward_probs(self._rows(x), mc_passes=1)


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM, however this torch build reports it."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):  # pragma: no cover - build dependent
        pass
    return "out of memory" in str(exc).lower()


def _derive(config: TorchExpertConfig | None, seed: int, **overrides) -> TorchExpertConfig:
    """A private per-expert copy of the shared config.

    `replace`, never mutation: every expert in a row is handed the same config
    instance, so assigning `cfg.seed = seed` wrote through to all of them. The
    last one constructed won, which quietly reduced B3's three-seed deep
    ensemble to three identical models — the row whose entire purpose is to show
    what ensembling alone buys.
    """
    return replace(config or TorchExpertConfig(), seed=seed, **overrides)


class CNNExpert(TorchSliceExpert):
    """ResNet-50 on 2-D slices — local spatial abnormality."""

    def __init__(self, cache, config: TorchExpertConfig | None = None, seed: int = 0):
        super().__init__(cache, _derive(config, seed), name="cnn")


class ViTExpert(TorchSliceExpert):
    """ViT-S/16 on 2-D slices — global context and long-range structure."""

    def __init__(self, cache, config: TorchExpertConfig | None = None, seed: int = 0):
        cfg = config or TorchExpertConfig()
        super().__init__(
            cache,
            _derive(cfg, seed, arch=cfg.vit_arch, img_size=cfg.vit_img_size),
            name="vit",
        )


class MCDropoutExpert(TorchSliceExpert):
    """Genuine MC dropout — the real B2 baseline.

    Replaces `DropoutEnsembleExpert`, which approximated the idea with bagging
    because torch was unavailable. Here dropout stays active at inference and
    N stochastic passes are averaged, which is the method as published
    (Gal & Ghahramani, ICML 2016).
    """

    def __init__(
        self,
        cache,
        config: TorchExpertConfig | None = None,
        n_passes: int = 20,
        seed: int = 0,
    ):
        # Dropout has to be substantial for MC sampling to say anything; the
        # default 0.3 used elsewhere barely moves the passes apart. Taken from
        # its own field so a caller-supplied config cannot silently reinstate
        # the shared rate -- see `mc_dropout` on TorchExpertConfig.
        cfg = config or TorchExpertConfig()
        super().__init__(cache, _derive(cfg, seed, dropout=cfg.mc_dropout), name="mc_dropout")
        self.n_passes = n_passes

    def predict_proba(self, x: np.ndarray) -> list[np.ndarray]:
        if not self.fitted_:
            raise RuntimeError(f"expert {self.name!r} must be fitted before predicting")
        return self._forward_probs(self._rows(x), mc_passes=self.n_passes)

"""Three deliberately different views of the same scan, one per expert.

The point of a heterogeneous ensemble is that its members fail differently. That
only holds if they genuinely see different things, so each view is built to be
sensitive to a different image property and therefore to break under a different
acquisition defect:

  texture -- first-order and gradient statistics inside the brain. This is the
             PyRadiomics-style view. Noise corrupts it first, because noise is
             indistinguishable from fine texture.

  local   -- a coarse grid of block means, i.e. "what is bright, and where".
             This is the CNN's view: localised spatial abnormality. Block
             averaging suppresses noise, but a bias field shifts block means
             directly, so this view breaks under intensity nonuniformity.

  global  -- marginal intensity profiles along each axis plus low-order spatial
             moments: long-range structure and left-right asymmetry. This is the
             ViT's view. It is robust to both noise and smooth bias, and breaks
             under blur and ghosting, which is what actually destroys long-range
             structure.

When torch becomes available, `local` and `global` are replaced by real CNN and
ViT backbones behind the same expert interface (see cgate/experts/README.md);
nothing downstream changes. Until then they are surrogates, and the code says so
rather than claiming a CNN is running.
"""

from __future__ import annotations

import numpy as np

FEATURE_VIEWS: tuple[str, ...] = ("texture", "local", "global")

_EPS = 1e-8


def _safe_stats(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(x.mean()), float(x.std())


def _moments(x: np.ndarray) -> tuple[float, float]:
    """Skewness and excess kurtosis, guarded against zero variance."""
    if x.size < 4:
        return 0.0, 0.0
    mu = x.mean()
    sd = x.std()
    if sd <= _EPS:
        return 0.0, 0.0
    z = (x - mu) / sd
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def texture_view(volumes: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """First-order + gradient statistics per modality (radiomics surrogate)."""
    feats: list[float] = []
    for c in range(volumes.shape[0]):
        v = volumes[c].astype(np.float64)
        fg = v[brain_mask]
        if fg.size == 0:
            feats.extend([0.0] * 13)
            continue
        mean, std = _safe_stats(fg)
        skew, kurt = _moments(fg)
        qs = np.percentile(fg, [10, 25, 50, 75, 90])
        hist, _ = np.histogram(fg, bins=32)
        p = hist.astype(np.float64)
        p = p / max(p.sum(), 1.0)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p))) if p.size else 0.0

        grads = [np.gradient(v, axis=a) for a in range(3)]
        gmag = np.sqrt(sum(g**2 for g in grads))[brain_mask]
        gmean, gstd = _safe_stats(gmag)

        feats.extend(
            [mean, std, skew, kurt, *[float(q) for q in qs], entropy, gmean, gstd,
             float(np.mean(fg**2))]
        )
    return np.asarray(feats, dtype=np.float64)


def local_view(
    volumes: np.ndarray, brain_mask: np.ndarray, grid: tuple[int, int, int] = (3, 3, 2)
) -> np.ndarray:
    """Block-mean grid per modality (CNN surrogate: localised spatial pattern)."""
    gx, gy, gz = grid
    feats: list[float] = []
    shape = volumes.shape[1:]
    edges = [np.linspace(0, shape[a], (grid[a] + 1)).astype(int) for a in range(3)]

    for c in range(volumes.shape[0]):
        v = volumes[c].astype(np.float64)
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    sl = (
                        slice(edges[0][i], edges[0][i + 1]),
                        slice(edges[1][j], edges[1][j + 1]),
                        slice(edges[2][k], edges[2][k + 1]),
                    )
                    m = brain_mask[sl]
                    block = v[sl]
                    feats.append(float(block[m].mean()) if np.any(m) else 0.0)
    return np.asarray(feats, dtype=np.float64)


def global_view(
    volumes: np.ndarray, brain_mask: np.ndarray, n_bins: int = 6
) -> np.ndarray:
    """Axis marginals + spatial moments per modality (ViT surrogate)."""
    feats: list[float] = []
    shape = volumes.shape[1:]
    coords = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")

    for c in range(volumes.shape[0]):
        v = volumes[c].astype(np.float64) * brain_mask

        # Marginal intensity profile along each axis, resampled to n_bins.
        for axis in range(3):
            other = tuple(a for a in range(3) if a != axis)
            prof = v.sum(axis=other)
            edges = np.linspace(0, prof.size, n_bins + 1).astype(int)
            total = max(float(prof.sum()), _EPS)
            for b in range(n_bins):
                seg = prof[edges[b] : edges[b + 1]]
                feats.append(float(seg.sum() / total))

        # Intensity-weighted centroid and spread: where the mass sits and how
        # far it spreads. Left-right centroid is what carries laterality.
        total = max(float(v.sum()), _EPS)
        for axis in range(3):
            centroid = float((coords[axis] * v).sum() / total)
            feats.append(centroid / max(shape[axis], 1))
            var = float((((coords[axis] - centroid) ** 2) * v).sum() / total)
            feats.append(np.sqrt(var) / max(shape[axis], 1))

        # Left-right energy asymmetry, computed directly because it is the
        # single most informative global cue for laterality.
        mid = shape[0] // 2
        lo = float(v[:mid].sum())
        hi = float(v[mid:].sum())
        feats.append((lo - hi) / max(lo + hi, _EPS))

    return np.asarray(feats, dtype=np.float64)


def extract_views(
    volumes: np.ndarray, brain_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """All three views for one case."""
    if volumes.ndim != 4:
        raise ValueError(
            f"expected volumes with shape (n_modalities, H, W, D), got {volumes.shape}"
        )
    return {
        "texture": texture_view(volumes, brain_mask),
        "local": local_view(volumes, brain_mask),
        "global": global_view(volumes, brain_mask),
    }


def view_dimensions(n_modalities: int = 4) -> dict[str, int]:
    """Feature count per view, for sanity checks and model sizing."""
    return {
        "texture": 13 * n_modalities,
        "local": 3 * 3 * 2 * n_modalities,
        "global": (6 * 3 + 6 + 1) * n_modalities,
    }

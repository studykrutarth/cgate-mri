"""No-reference image quality metrics (IQMs) -- the router's input features.

Why this module exists
----------------------
The routing hypothesis is that different experts degrade differently as scan
quality drops, so the fusion weights should be conditioned on measured quality.
That requires a quality descriptor computed *without* ground truth, from the
image alone. These are the standard no-reference IQMs used by MRIQC and MRQy.

The skull-stripping trap
------------------------
Most published IQMs (SNR, CNR, FBER, QI1) are defined against the *air* region
outside the head: they measure signal relative to background noise, and QI1
counts artefact-corrupted background voxels. BraTS volumes are skull-stripped
and background-zeroed, so that region is identically zero and every air-based
metric collapses to a constant or a divide-by-zero.

Computing them anyway and feeding constants to the router would make the router
look useless for reasons that have nothing to do with the hypothesis. So
`compute_iqms` detects a degenerate background and reports which metrics are
valid, and `ROUTER_FEATURES` lists only the ones that survive skull-stripping.
The air-based metrics are still implemented, because they are informative on
raw (pre-skull-strip) data such as the unprocessed BraTS-Africa scans.

References for the metric definitions are in docs/quality-metrics.md.
"""

from __future__ import annotations

import numpy as np

# Metrics that remain meaningful on skull-stripped, background-zeroed volumes.
# This is the feature vector the router consumes.
ROUTER_FEATURES: tuple[str, ...] = (
    "cjv",
    "efc",
    "snr",
    "sharpness",
    "inu",
    "contrast",
    "entropy",
)

# Metrics that require a non-degenerate air region (raw / non-skull-stripped).
AIR_FEATURES: tuple[str, ...] = ("snr_air", "fber", "qi1")

ALL_FEATURES: tuple[str, ...] = ROUTER_FEATURES + AIR_FEATURES

# MAD -> sigma conversion for a Gaussian.
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817
_EPS = 1e-8


def otsu_threshold(values: np.ndarray, nbins: int = 128) -> float:
    """Otsu's between-class-variance threshold. Pure numpy, no scipy."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return vmin

    hist, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    total = hist.sum()
    if total <= 0:
        return vmin

    w0 = np.cumsum(hist)
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    total_mean = csum[-1]

    # Only split points where both classes are non-empty are admissible.
    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        return vmin

    mu0 = np.zeros_like(w0)
    mu1 = np.zeros_like(w1)
    np.divide(csum, w0, out=mu0, where=valid)
    np.divide(total_mean - csum, w1, out=mu1, where=valid)

    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~valid] = -np.inf
    return float(centers[int(np.argmax(between))])


def _noise_sigma(volume: np.ndarray, mask: np.ndarray) -> float:
    """Robust noise estimate from first differences inside the brain.

    Differencing adjacent voxels cancels the smooth anatomical signal and leaves
    noise scaled by sqrt(2); the median absolute deviation makes it insensitive
    to the tumour and to edges. This works on skull-stripped data, where the
    usual approach of measuring the standard deviation of air is unavailable.
    """
    diffs: list[np.ndarray] = []
    for axis in range(volume.ndim):
        d = np.diff(volume, axis=axis)
        m = np.logical_and(
            np.take(mask, range(0, mask.shape[axis] - 1), axis=axis),
            np.take(mask, range(1, mask.shape[axis]), axis=axis),
        )
        if np.any(m):
            diffs.append(d[m])
    if not diffs:
        return 0.0
    d = np.concatenate(diffs)
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    return mad * _MAD_TO_SIGMA / np.sqrt(2.0)


def _gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, same shape as input."""
    grads = [np.gradient(volume, axis=ax) for ax in range(volume.ndim)]
    return np.sqrt(sum(g**2 for g in grads))


def _block_means(volume: np.ndarray, mask: np.ndarray, blocks: int = 4) -> np.ndarray:
    """Mean intensity over a coarse grid of blocks, ignoring empty blocks.

    Used as a bias-field / intensity-nonuniformity proxy: a well-corrected
    volume has similar tissue means everywhere, an uncorrected one drifts.
    """
    out: list[float] = []
    slices_per_axis = []
    for ax in range(volume.ndim):
        edges = np.linspace(0, volume.shape[ax], blocks + 1).astype(int)
        slices_per_axis.append(
            [slice(edges[i], edges[i + 1]) for i in range(blocks)]
        )

    def walk(idx: list[slice], axis: int) -> None:
        if axis == volume.ndim:
            sl = tuple(idx)
            m = mask[sl]
            if np.count_nonzero(m) >= 32:
                out.append(float(volume[sl][m].mean()))
            return
        for s in slices_per_axis[axis]:
            walk(idx + [s], axis + 1)

    walk([], 0)
    return np.asarray(out, dtype=np.float64)


def _shannon_entropy(values: np.ndarray, nbins: int = 64) -> float:
    hist, _ = np.histogram(values, bins=nbins)
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def efc(values: np.ndarray) -> float:
    """Entropy Focus Criterion (Atkinson et al., 1997).

    Measures how concentrated the image energy is. Ghosting and blurring spread
    energy out and raise EFC, so *lower is better*. Normalised by the EFC of a
    uniform image of the same voxel count so volumes of different sizes compare.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[x > 0]
    n = x.size
    if n < 2:
        return 0.0

    b_max = np.sqrt((x**2).sum())
    if b_max <= _EPS:
        return 0.0

    efc_max = n * (1.0 / np.sqrt(n)) * np.log(1.0 / np.sqrt(n))
    if abs(efc_max) <= _EPS:
        return 0.0

    r = x / b_max
    return float((1.0 / abs(efc_max)) * np.sum(r * np.log(r)))


def cjv(values: np.ndarray) -> float:
    """Coefficient of Joint Variation between the two dominant tissue classes.

    CJV = (sigma_a + sigma_b) / |mu_a - mu_b|. Rises with intensity
    nonuniformity and head motion, so *lower is better*. MRIQC computes it over
    white and grey matter from a tissue segmentation; here the two classes come
    from an Otsu split of the brain, which needs no atlas or segmentation and
    tracks the same quantity closely enough to rank scans by quality.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    thr = otsu_threshold(x)
    lo = x[x <= thr]
    hi = x[x > thr]
    if lo.size < 2 or hi.size < 2:
        return 0.0
    denom = abs(float(hi.mean()) - float(lo.mean()))
    if denom <= _EPS:
        return float("inf") if (lo.std() + hi.std()) > 0 else 0.0
    return float((lo.std() + hi.std()) / denom)


def compute_iqms(
    volume: np.ndarray,
    brain_mask: np.ndarray | None = None,
    air_threshold_percentile: float = 5.0,
) -> dict[str, float]:
    """Compute all IQMs for one 3-D volume.

    Returns a dict with every key in ALL_FEATURES plus:
      background_degenerate : 1.0 if the air region is empty/zero (skull-stripped)
      brain_fraction        : fraction of voxels inside the brain mask

    Air-based metrics are set to NaN when the background is degenerate rather
    than to a fake number, so downstream code cannot silently use them.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {volume.shape}")

    if brain_mask is None:
        # Skull-stripped data: anything non-zero is brain. On raw data this
        # falls back to a low-percentile threshold.
        if np.any(volume > 0):
            thr = max(0.0, float(np.percentile(volume[volume > 0], air_threshold_percentile)))
        else:
            thr = 0.0
        brain_mask = volume > thr
    brain_mask = np.asarray(brain_mask, dtype=bool)

    fg = volume[brain_mask]
    bg = volume[~brain_mask]

    out: dict[str, float] = {k: float("nan") for k in ALL_FEATURES}
    out["brain_fraction"] = float(brain_mask.mean())

    if fg.size < 8:
        out["background_degenerate"] = 1.0
        for k in ROUTER_FEATURES:
            out[k] = 0.0
        return out

    # An air region is usable only if it exists and actually carries signal.
    bg_degenerate = bg.size == 0 or float(np.abs(bg).max()) <= _EPS
    out["background_degenerate"] = float(bg_degenerate)

    fg_mean = float(fg.mean())
    fg_std = float(fg.std())

    # --- metrics valid on skull-stripped data ---------------------------
    out["cjv"] = cjv(fg)
    out["efc"] = efc(fg)

    sigma = _noise_sigma(volume, brain_mask)
    out["snr"] = float(fg_mean / sigma) if sigma > _EPS else 0.0

    grad = _gradient_magnitude(volume)[brain_mask]
    out["sharpness"] = float(grad.mean() / fg_mean) if abs(fg_mean) > _EPS else 0.0

    blocks = _block_means(volume, brain_mask)
    if blocks.size >= 2 and abs(blocks.mean()) > _EPS:
        out["inu"] = float(blocks.std() / abs(blocks.mean()))
    else:
        out["inu"] = 0.0

    out["contrast"] = float(fg_std / fg_mean) if abs(fg_mean) > _EPS else 0.0
    out["entropy"] = _shannon_entropy(fg)

    # --- metrics needing a real air region -------------------------------
    if not bg_degenerate:
        bg_std = float(bg.std())
        out["snr_air"] = float(fg_mean / bg_std) if bg_std > _EPS else float("nan")

        bg_energy = float(np.median(bg**2))
        fg_energy = float(np.median(fg**2))
        out["fber"] = (
            float(fg_energy / bg_energy) if bg_energy > _EPS else float("nan")
        )

        # QI1 proxy: share of air voxels carrying signal well above the air
        # noise floor, i.e. ghosting/spikes leaking outside the head.
        bg_mad = float(np.median(np.abs(bg - np.median(bg)))) * _MAD_TO_SIGMA
        if bg_mad > _EPS:
            out["qi1"] = float(np.mean(np.abs(bg) > 5.0 * bg_mad))
        else:
            out["qi1"] = float("nan")

    return out


def iqm_vector(iqms: dict[str, float], features: tuple[str, ...] = ROUTER_FEATURES) -> np.ndarray:
    """Pull a fixed-order feature vector out of an IQM dict."""
    return np.asarray([float(iqms.get(f, np.nan)) for f in features], dtype=np.float64)


class QualityStandardizer:
    """Z-scores IQM features using training-split statistics only.

    The router is a linear-softmax model, so unscaled features (EFC ~1e-1,
    entropy ~4) would make its weights incomparable and its optimisation
    ill-conditioned. Statistics are fitted once on train and reused everywhere,
    which also means a distribution shift at test time (BraTS-Africa) shows up
    as genuinely out-of-range features rather than being normalised away.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> QualityStandardizer:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected 2-D feature matrix, got shape {x.shape}")
        finite = np.isfinite(x)
        # Column means/stds over finite entries only; non-finite IQMs (e.g. air
        # metrics on skull-stripped data) must not poison the statistics.
        counts = finite.sum(axis=0)
        safe = np.where(finite, x, 0.0)
        mean = np.where(counts > 0, safe.sum(axis=0) / np.maximum(counts, 1), 0.0)
        var = np.where(
            counts > 0,
            (np.where(finite, (x - mean) ** 2, 0.0)).sum(axis=0) / np.maximum(counts, 1),
            0.0,
        )
        scale = np.sqrt(var)
        scale[scale < 1e-6] = 1.0
        self.mean_, self.scale_ = mean, scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("QualityStandardizer.fit must be called first")
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean_) / self.scale_
        # Replace non-finite features with the training mean (0 after scaling)
        # so a missing air metric degrades to "average" instead of NaN-ing the
        # whole router.
        return np.where(np.isfinite(z), z, 0.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

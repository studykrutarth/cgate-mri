"""Synthetic BraTS-like phantoms with controllable acquisition quality.

Purpose and limits
------------------
This exists so the whole pipeline -- IQMs, experts, router, calibration, gate,
ablations -- can be run, tested and demonstrated without the multi-gigabyte
BraTS download, and so CI can assert end-to-end behaviour deterministically.

It is NOT evidence for the scientific hypothesis. Numbers produced on phantoms
show that the plumbing is correct and that the metrics respond in the expected
direction; they say nothing about whether quality-aware routing helps on real
gliomas. Every reported result must state which data it came from, and
`cgate.experiments.run` prints a banner to that effect. Real conclusions require
BraTS 2021 for training and BraTS-Africa for the low-quality stratum.

Design
------
Each phantom has a three-compartment tumour matching the BraTS label convention
(necrotic core, enhancing rim, peritumoural edema) placed in one hemisphere or
across the midline, on a coarse brain-like background with distinct tissue
intensities per pseudo-modality (T1, T1ce, T2, FLAIR).

Degradations are applied in acquisition order -- bias field, then blur, then
ghosting, then Rician noise -- because that ordering is what makes the IQMs
respond the way they do on real scanners (e.g. noise added before blurring would
be smoothed away and EFC would not move).

The corruptions are chosen to hit different image properties: noise destroys
fine texture while leaving gross shape intact, blur destroys edges, bias field
distorts absolute intensity but not local texture. That is what makes the three
feature views degrade at different rates, which is the phenomenon the router is
supposed to exploit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema import LABEL_ED, LABEL_ET, LABEL_NCR

MODALITIES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


@dataclass
class SyntheticConfig:
    shape: tuple[int, int, int] = (48, 48, 32)
    n_cases: int = 240
    # Fraction of cases drawn from the degraded ("field-hospital") cohort. The
    # default mirrors the real design: train mostly on high-quality data, then
    # evaluate on a low-quality stratum.
    low_quality_fraction: float = 0.35
    seed: int = 0
    skull_strip: bool = True


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(vol: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding, pure numpy.

    Implemented as a sum of shifted slices rather than `np.apply_along_axis`,
    which would run one Python-level 1-D convolution per line -- tens of
    thousands per volume, and by far the dominant cost of dataset generation.
    Every tap here is a single vectorised array operation.
    """
    if sigma <= 0:
        return vol.astype(np.float64)
    k = _gaussian_kernel1d(sigma)
    r = (len(k) - 1) // 2
    out = vol.astype(np.float64)
    for axis in range(out.ndim):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += w * padded[tuple(sl)]
        out = acc
    return out


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int, int], scale: float) -> np.ndarray:
    """Low-frequency multiplicative field, used as a bias-field model."""
    coarse = rng.normal(size=(4, 4, 4))
    # Trilinear upsample by index broadcasting, then smooth.
    idx = [np.linspace(0, 3, s) for s in shape]
    g = np.stack(np.meshgrid(*idx, indexing="ij"), axis=-1)
    i0 = np.floor(g).astype(int)
    i1 = np.minimum(i0 + 1, 3)
    frac = g - i0
    out = np.zeros(shape)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wx = frac[..., 0] if dx else 1 - frac[..., 0]
                wy = frac[..., 1] if dy else 1 - frac[..., 1]
                wz = frac[..., 2] if dz else 1 - frac[..., 2]
                ix = i1[..., 0] if dx else i0[..., 0]
                iy = i1[..., 1] if dy else i0[..., 1]
                iz = i1[..., 2] if dz else i0[..., 2]
                out += wx * wy * wz * coarse[ix, iy, iz]
    out = _blur(out, 1.5)
    m = np.abs(out).max()
    if m > 0:
        out = out / m
    return 1.0 + scale * out


def _ellipsoid(shape: tuple[int, int, int], center, radii) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
    d = sum(((g - c) / max(r, 1e-6)) ** 2 for g, c, r in zip(grids, center, radii))
    return d <= 1.0


@dataclass
class SynthCase:
    patient_id: str
    volumes: np.ndarray  # (n_modalities, H, W, D)
    mask: np.ndarray  # (H, W, D) with BraTS labels
    brain_mask: np.ndarray
    cohort: str
    quality_level: float
    truth: dict = field(default_factory=dict)


def synth_case(
    rng: np.random.Generator,
    shape: tuple[int, int, int] = (48, 48, 32),
    quality_level: float = 1.0,
    cohort: str = "high",
    patient_id: str = "SYN-0000",
    skull_strip: bool = True,
) -> SynthCase:
    """Generate one phantom.

    `quality_level` in [0, 1]: 1.0 is a pristine 3-T-like acquisition, values
    toward 0 add bias field, blur, ghosting and Rician noise.
    """
    H, W, D = shape
    q = float(np.clip(quality_level, 0.0, 1.0))

    # --- anatomy ---------------------------------------------------------
    brain = _ellipsoid(shape, (H / 2, W / 2, D / 2), (H * 0.42, W * 0.44, D * 0.42))
    ventricle = _ellipsoid(shape, (H / 2, W / 2, D / 2), (H * 0.10, W * 0.16, D * 0.12))

    # Grey matter as an outer rim, white matter inside.
    inner = _ellipsoid(shape, (H / 2, W / 2, D / 2), (H * 0.34, W * 0.36, D * 0.34))
    grey = brain & ~inner
    white = inner & ~ventricle

    # --- tumour ----------------------------------------------------------
    # Latent grade drives whether an enhancing rim exists, so `et_present` is
    # genuinely uncertain rather than constant.
    high_grade = rng.random() < 0.6
    bilateral = rng.random() < 0.25

    # All radii and offsets are expressed as fractions of the volume size, so
    # changing `shape` rescales the anatomy instead of distorting it. With
    # absolute voxel counts a smaller volume made every tumour straddle the
    # midline and the laterality finding collapsed to ~all-bilateral.
    r_core = H * rng.uniform(0.045, 0.095)
    r_et = r_core * (1.0 + (rng.uniform(0.25, 0.6) if high_grade else rng.uniform(0.0, 0.06)))
    r_ed = r_et + H * rng.uniform(0.035, 0.075)

    if bilateral:
        cx = H / 2 + rng.normal(0, H * 0.015)
    else:
        side = 1 if rng.random() < 0.5 else -1
        # Offset far enough that the edema rim stays on one side of the
        # midplane, otherwise "unilateral" cases would be labelled bilateral --
        # but not so far that the mass leaves the brain, which would clip the
        # tumour against the ellipsoid and shrink whole-tumour volume instead.
        offset = r_ed + H * rng.uniform(0.03, 0.10)
        offset = min(offset, H * 0.40 - r_ed)
        cx = H / 2 + side * max(offset, r_ed * 0.5)
    cy = W / 2 + rng.normal(0, W * 0.07)
    cz = D / 2 + rng.normal(0, D * 0.06)

    core = _ellipsoid(shape, (cx, cy, cz), (r_core, r_core, r_core * 0.8)) & brain
    et = (_ellipsoid(shape, (cx, cy, cz), (r_et, r_et, r_et * 0.8)) & ~core) & brain
    ed = (
        _ellipsoid(shape, (cx, cy, cz), (r_ed, r_ed * 1.1, r_ed * 0.85))
        & ~core
        & ~et
        & brain
    )

    mask = np.zeros(shape, dtype=np.int16)
    mask[ed] = LABEL_ED
    mask[core] = LABEL_NCR
    if high_grade:
        mask[et] = LABEL_ET

    # --- intensities per pseudo-modality ---------------------------------
    # (background, csf, grey, white, ncr, ed, et)
    intensity_table = {
        "t1": (0.0, 0.25, 0.60, 0.85, 0.30, 0.50, 0.62),
        "t1ce": (0.0, 0.25, 0.60, 0.85, 0.28, 0.50, 1.00),
        "t2": (0.0, 1.00, 0.70, 0.45, 0.85, 0.95, 0.70),
        "flair": (0.0, 0.20, 0.68, 0.45, 0.75, 1.00, 0.78),
    }

    # Corruption emphasis, drawn once per case and shared across modalities: a
    # single acquisition has one motion profile and one shim, so its four
    # sequences are degraded coherently.
    #
    # Emphasis is drawn independently of overall severity so that *which* defect
    # dominates varies case to case. Without this every corruption would scale
    # with a single hidden variable, all three feature views would degrade in
    # lockstep, and the router could not learn anything the uniform ensemble
    # does not already have -- the testbed would be rigged against the very
    # hypothesis it is meant to exercise.
    emphasis = rng.dirichlet([1.4, 1.4, 1.4, 1.4]) * 4.0
    emph_bias, emph_blur, emph_ghost, emph_noise = (float(x) for x in emphasis)
    severity = 1.0 - q
    bias_field = _smooth_field(rng, shape, 0.05 + 0.45 * severity * emph_bias)

    volumes = np.zeros((len(MODALITIES), H, W, D), dtype=np.float32)
    for mi, mod in enumerate(MODALITIES):
        bg, csf, gm, wm, ncr_i, ed_i, et_i = intensity_table[mod]
        v = np.full(shape, bg, dtype=np.float64)
        v[ventricle] = csf
        v[grey] = gm
        v[white] = wm
        v[ed] = ed_i
        v[core] = ncr_i
        if high_grade:
            v[et] = et_i

        # Fine tissue texture. This is the signal the texture-sensitive expert
        # relies on and the first thing noise destroys.
        texture = rng.normal(0, 0.035, size=shape)
        v = v + texture * brain

        # --- degradations, in acquisition order --------------------------
        v = v * bias_field

        blur_sigma = 0.3 + 1.5 * severity * emph_blur
        v = _blur(v, blur_sigma)

        if severity > 0.05:
            # Ghosting: a shifted, attenuated copy along the phase-encode axis.
            shift = max(1, int(W * 0.18))
            ghost = np.roll(v, shift, axis=1) * (0.12 * severity * emph_ghost)
            v = v + ghost

        noise_sigma = 0.01 + 0.10 * severity * emph_noise
        # Rician: magnitude of a complex Gaussian, which is what magnitude MRI
        # actually produces and why low-signal regions get a positive bias.
        real = v + rng.normal(0, noise_sigma, size=shape)
        imag = rng.normal(0, noise_sigma, size=shape)
        v = np.sqrt(real**2 + imag**2)

        if skull_strip:
            v = v * brain

        volumes[mi] = v.astype(np.float32)

    return SynthCase(
        patient_id=patient_id,
        volumes=volumes,
        mask=mask,
        brain_mask=brain,
        cohort=cohort,
        quality_level=q,
        truth={
            "high_grade": bool(high_grade),
            "bilateral": bool(bilateral),
            # Recorded so tests can assert the IQMs actually track the defect
            # that was injected, rather than merely correlating with severity.
            "emph_bias": emph_bias,
            "emph_blur": emph_blur,
            "emph_ghost": emph_ghost,
            "emph_noise": emph_noise,
        },
    )


def build_synthetic_dataset(config: SyntheticConfig | None = None):
    """Yield phantoms one at a time.

    A generator rather than a list on purpose: at 48x48x32x4 a few hundred
    volumes would be hundreds of megabytes held for no reason, since the
    pipeline reduces each case to features and labels immediately.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n_low = int(round(cfg.n_cases * cfg.low_quality_fraction))
    cohorts = ["low"] * n_low + ["high"] * (cfg.n_cases - n_low)
    rng.shuffle(cohorts)

    for i, cohort in enumerate(cohorts):
        if cohort == "high":
            # Tight, good-quality band: a well-run 3 T protocol.
            q = float(np.clip(rng.normal(0.88, 0.07), 0.6, 1.0))
        else:
            # Wide, poor band: 1.5 T in a resource-constrained setting, where
            # the spread of quality is itself much larger.
            q = float(np.clip(rng.normal(0.32, 0.16), 0.0, 0.7))
        yield synth_case(
            rng,
            shape=cfg.shape,
            quality_level=q,
            cohort=cohort,
            patient_id=f"SYN-{i:04d}",
            skull_strip=cfg.skull_strip,
        )

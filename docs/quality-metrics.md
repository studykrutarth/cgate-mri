# Image quality metrics (IQMs)

Reference for `cgate/quality.py`. These are the no-reference metrics that form
the router's input vector — computed from the image alone, with no ground truth.

## Which metrics are used, and why only seven

`ROUTER_FEATURES` contains the metrics that survive skull-stripping:

| Metric | Measures | Direction | Notes |
|---|---|---|---|
| `cjv` | Intensity nonuniformity + head motion | lower better | Coefficient of Joint Variation |
| `efc` | Ghosting and blurring | lower better | Entropy Focus Criterion |
| `snr` | Noise level | higher better | From within-brain differences, not air |
| `sharpness` | Edge definition | higher better | Mean gradient / mean intensity |
| `inu` | Bias-field drift | lower better | Block-mean coefficient of variation |
| `contrast` | Tissue separation | — | Foreground std / mean |
| `entropy` | Histogram spread | — | Shannon entropy of foreground |

`AIR_FEATURES` (`snr_air`, `fber`, `qi1`) are implemented but **excluded from
routing**. See below.

## The skull-stripping trap

Most published IQMs are defined against the **air** region outside the head:

- **SNR** (classic) = mean(tissue) / std(air)
- **CNR** = |mean(WM) − mean(GM)| / std(air)
- **FBER** = median(foreground²) / median(background²)
- **QI1** = proportion of *background* voxels corrupted by artefact

BraTS volumes are skull-stripped and background-zeroed. That region is
identically zero, so:

- `std(air) = 0` → SNR and CNR divide by zero
- `median(background²) = 0` → FBER divides by zero
- QI1 counts artefact voxels in a region that has been erased → always 0

Feeding constants (or silently-clipped infinities) to the router would make it
look useless for reasons unrelated to the hypothesis under test.

**What the code does:** `compute_iqms` detects a degenerate background and
returns `NaN` for every air-based metric, plus a `background_degenerate` flag.
`QualityStandardizer` maps non-finite features to the training mean, so a
missing metric degrades to "average" rather than NaN-ing the whole router.

The air metrics stay implemented because they *are* valid on raw,
pre-skull-strip data — including the unprocessed BraTS-Africa scans used in the
published quality comparison.

## Metric definitions

### EFC — Entropy Focus Criterion

Atkinson et al., IEEE TMI 16(6), 1997. Measures how concentrated image energy
is; ghosting and blurring spread energy out and raise EFC.

```
B_max = sqrt(Σ x_i²)
EFC   = (1 / |EFC_max|) · Σ (x_i / B_max) · ln(x_i / B_max)
EFC_max = N · (1/√N) · ln(1/√N)
```

Normalising by `EFC_max` (the EFC of a uniform image of the same voxel count)
makes volumes of different sizes comparable. Zero and negative voxels are
excluded before the log.

**Caveat:** EFC's sensitivity to ghosting comes largely from energy appearing
*outside* the head. Skull-stripping removes that, so on BraTS the metric is
weaker than on raw data. It is retained because it still responds to
within-brain blurring.

### CJV — Coefficient of Joint Variation

```
CJV = (σ_a + σ_b) / |μ_a − μ_b|
```

MRIQC computes this over white and grey matter from a tissue segmentation.
Here the two classes come from an **Otsu split of the brain**, which needs no
atlas or segmentation step and tracks the same quantity closely enough to rank
scans by quality. Documented as a proxy, not as the segmentation-based value.

CJV is the headline metric in the BraTS-Africa quality comparison: higher CJV
means more intensity nonuniformity and motion.

### SNR without air

Noise is estimated from **first differences inside the brain**:

```
σ ≈ MAD(diff) · 1.4826 / √2
SNR = mean(foreground) / σ
```

Differencing adjacent voxels cancels smooth anatomical signal and leaves noise
scaled by √2. The median absolute deviation makes the estimate insensitive to
the tumour and to tissue edges. This is what makes an SNR-like measure available
on skull-stripped data at all.

### INU — bias-field proxy

Mean intensity over a 4×4×4 grid of blocks (blocks with fewer than 32 brain
voxels are skipped), then the coefficient of variation across blocks. A
well-corrected volume has similar tissue means everywhere; an uncorrected one
drifts across the field of view.

### QI1 proxy

The published QI1 (Mortamet et al., 2009) fits a distribution to background
voxels. The implementation here is simpler: the fraction of air voxels whose
magnitude exceeds 5×MAD of the air noise floor — i.e. ghosting or spikes leaking
outside the head. Labelled a proxy in the code.

## Standardisation

`QualityStandardizer` z-scores features using **training statistics only**.

This is deliberate: if statistics were refitted on the test set, a genuine
distribution shift (BraTS 2021 → BraTS-Africa) would be normalised away and the
router would never see that it is operating out of range. Keeping training
statistics means a degraded cohort produces genuinely out-of-range features,
which is the signal the router is meant to act on.

## References

- Atkinson D., et al. "Automatic correction of motion artifacts in MRI using an
  entropy focus criterion." IEEE TMI 16(6), 1997.
- Mortamet B., et al. "Automatic quality assessment in structural brain MRI."
  Magnetic Resonance in Medicine 62(2), 2009.
- Esteban O., et al. "MRIQC: Advancing the automatic prediction of image quality
  in MRI from unseen sites." PLoS ONE 12(9), 2017.
- Muthusivarajan R., et al. "Evaluating the relationship between magnetic
  resonance image quality metrics and deep learning-based segmentation accuracy
  of brain tumors." Medical Physics, 2024. — empirical link between IQMs and
  Dice; the closest published support for the routing hypothesis.
- Adewole M., et al. "The BraTS-Africa Dataset." Radiology: Artificial
  Intelligence, 2025. doi:10.1148/ryai.240528 — CJV/EFC/QI1 computed with MRIQC
  22.0.6 on 50 BraTS-Africa (1.5 T) vs 50 TCGA-GBM (3 T) scans.

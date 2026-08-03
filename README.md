# cgate-mri

**Quality-aware expert routing with calibrated confidence gating for brain tumour MRI.**

Project-I (B.Tech. VII, Group 6D1). Implements the routing + calibration + gate
half of the proposal. Grounded report generation is deliberately **out of scope**
— see [Scope](#scope) for why.

---

## The one-line claim

> Fusion weights conditioned on *measured image quality* produce better-ranked
> confidence than fixed-weight fusion, and that improvement concentrates in the
> low-quality stratum.

Stated as a falsifiable test: **B6 beats B4 on AURC in the low-quality cohort,
with a paired-bootstrap CI that excludes zero.** If the CI crosses zero, routing
did not help and that is the result we report.

---

## Quick start

```bash
pip install -e ".[sci,dev]"
pytest -q                                       # 65 tests, ~45s
cgate --n-cases 400 --figures runs/figs         # ablation grid + plots
```

On Windows the console script may not land on PATH; use
`py -3 -m cgate.experiments.run ...` instead.

No dataset download is required: the default runs on synthetic phantoms.
`torch` is **not** needed — see [Status](#status).

---

## What gets fused

The question a review panel asks first. A CNN emits voxel masks; an XGBoost
radiomics model emits case-level logits. Those cannot be averaged. So every
expert is required to produce the **same** target:

> a case-level **structured finding vector** — K categorical findings, each
> expert emitting one probability simplex per finding.

| Finding | Classes | Critical | Derived from |
|---|---|---|---|
| `et_present` | absent / present | yes | ET volume ≥ threshold |
| `ncr_present` | absent / present | yes | NCR volume ≥ threshold |
| `laterality` | left / right / bilateral | yes | WT mass about the mid-sagittal plane |
| `burden` | small / medium / large | no | WT volume tercile |
| `et_fraction` | low / mid / high | no | ET/WT ratio tercile |

Every finding is derivable from the BraTS mask alone — no extra annotation.
"Edema present" is deliberately excluded: it is true in nearly every case, so it
would inflate accuracy while telling us nothing about calibration.

Segmentation is **upstream** of this target (voxel experts reduce their predicted
mask through the same reduction) and is reported as an auxiliary metric. It is
never the gated quantity. Tercile edges are fitted on **train only**.

---

## Architecture

```
MRI (4 sequences)
      │
      ├─► IQM extractor ──────────────► quality vector q
      │   CJV, EFC, SNR, sharpness,     (7 skull-strip-safe metrics)
      │   INU, contrast, entropy
      │
      ├─► texture view ─► radiomics expert (gradient-boosted trees) ─┐
      ├─► local view   ─► CNN-surrogate expert  (MLP)               ─┤
      └─► global view  ─► ViT-surrogate expert  (MLP)               ─┤
                                                                     │
                     per-expert temperature scaling ◄────────────────┘
                                    │
                     w = softmax(W·q + b)   ◄── router, conditioned on quality
                                    │
                          confidence-weighted fusion
                                    │
                        final temperature scaling
                                    │
                    case confidence = min over critical findings
                                    │
                     ┌──────────────┴──────────────┐
              ≥ threshold                    < threshold
              release                    defer to radiologist
```

Case confidence is a **minimum**, not a mean: a case is only safe to release if
*every* critical finding is trustworthy. Averaging would let three confident
findings mask one the model is guessing at.

---

## Why the ablation grid exists

Three heterogeneous models fused together **are a deep ensemble**, and deep
ensembles are already a strong calibration baseline. "Our system is well
calibrated" therefore proves nothing. Each row adds exactly one factor:

| | System | Isolates |
|---|---|---|
| B0 | Single expert, raw softmax | uncalibrated floor |
| B1 | + temperature scaling | post-hoc calibration |
| B2 | Sampling-based uncertainty (MC-dropout surrogate) | uncertainty by sampling |
| B3 | Homogeneous 3× deep ensemble | **ensembling alone** |
| B4 | Heterogeneous experts, uniform fusion | **heterogeneity** (vs B3) |
| B5 | Heterogeneous, confidence-weighted fusion | self-reported confidence |
| B6 | Heterogeneous, **quality-aware routing** (ours) | **routing** (vs B4/B5) |

The router is **initialised to exactly uniform weights**, so untrained B6 *is*
B4. Any gap is attributable to what training learned from quality features, not
to a change of fusion family. (`test_untrained_router_is_exactly_uniform_fusion`
enforces this.)

---

## Metrics

ECE alone does not evaluate a gate — it measures whether stated confidence
matches accuracy on average, not whether thresholding on it removes errors.

- **Calibration:** ECE, adaptive ECE, Brier, NLL, reliability curve
- **Gate quality:** risk–coverage curve, **AURC**, **E-AURC**
- **Operational:** risk @ 80% coverage, coverage @ 5%/10% risk
- **Significance:** paired bootstrap CI on ΔAURC between systems
- **Figures:** `--figures DIR` writes risk–coverage curves (overall and per
  cohort) and a reliability diagram

E-AURC matters because raw AURC is bounded below by the model's error rate: an
accurate model looks good on AURC even with worthless confidence. Subtracting
the optimal curve isolates the confidence *ranking*.

---

## The skull-stripping trap

Most published IQMs (SNR, CNR, FBER, QI1) are defined against the **air** region
outside the head. BraTS is skull-stripped and background-zeroed, so that region
is identically zero and every air-based metric collapses to a constant or a
divide-by-zero.

`compute_iqms` detects a degenerate background and returns **NaN** for air-based
metrics rather than a fake number. `ROUTER_FEATURES` lists only the seven metrics
that survive skull-stripping. The air metrics remain implemented because they are
valid on *raw* pre-skull-strip data, such as unprocessed BraTS-Africa scans.

---

## Split discipline

```
train      → experts
calib_fit  → per-expert temperatures, router
calib_gate → final fused temperature, gate threshold
test       → touched once
```

Calibration is **halved** because the router is optimised on `calib_fit`;
choosing the gate threshold on the same cases lets the gate exploit the router's
in-sample fit and overstates coverage at a given risk.

Splits are **patient-level**. `assert_no_leakage` runs automatically on every
split — with 2D-slice or 3D-patch training, splitting by sample puts
neighbouring slices of the same tumour on both sides, which is the most common
silent bug in this literature.

### Two protocols — use `external` for the real experiment

```bash
cgate --protocol stratified     # random split, both cohorts everywhere (default)
cgate --protocol external       # hold out an ENTIRE cohort as test
```

`external` trains and calibrates on the high-quality cohort, then tests on
**every** case of the degraded one. Use it for BraTS 2021 → BraTS-Africa, for
two reasons:

1. **Statistical power.** BraTS-Africa has ~146 cases. A 20% test split leaves
   ~29 — measured on phantoms, that is not enough to resolve the ~0.03 AURC
   effect this project is looking for; the confidence interval swamps it.
   Holding out the whole cohort multiplies the test set roughly fivefold at no
   cost.
2. **It is the real deployment question.** "We trained where the data is good;
   what happens when the scanner is worse?" A random split mixes degraded scans
   into training, which is exactly the confound that hides whether routing
   transfers.

Trade-off to state when reporting: under `external`, train and test differ in
acquisition as well as quality, so a drop cannot be attributed to quality alone.
Report both protocols.

---

## Status

| Component | State |
|---|---|
| IQMs, findings, fusion, router, calibration, gate, metrics | ✅ implemented, numpy-only, tested |
| Radiomics expert (gradient-boosted trees) | ✅ real |
| CNN / ViT experts | ⚠️ **surrogates** — MLPs on hand-built views |
| MC dropout | ⚠️ **surrogate** — bootstrap + feature dropout |
| Real BraTS loader (2021 + Africa naming) | ✅ implemented, untested against real files |

`torch` has no wheel for the Python 3.14 interpreter on this machine, so the CNN
and ViT experts are stood in by MLPs consuming deliberately different feature
views (texture / local / global) chosen to degrade under different acquisition
defects. **Nothing in the code claims these are CNNs.** Replacing them means
implementing `fit`/`predict_proba` in `experts/torch_experts.py`; no other module
changes.

The core contributions — routing, calibration, gating, selective-prediction
metrics — are real and complete, and are what the surrogates exist to let us
develop and validate end-to-end before GPU work lands.

---

## Synthetic data is not a result

`cgate` defaults to phantoms so the pipeline is runnable and CI-testable without
the multi-gigabyte download. The runner prints a banner saying so.

Phantom numbers verify that the plumbing is correct and the metrics move in the
expected direction. They are **not** evidence that quality-aware routing helps on
real gliomas. That requires:

```bash
cgate --data brats --root /path/to/datasets   # BraTS2021/ alongside BraTS-Africa/
```

Train on BraTS 2021 (predominantly 3 T), evaluate the low-quality stratum on
**BraTS-Africa** (1.5 T, resource-constrained settings). The cohort is inferred
from the directory name.

---

## Scope

Project-I delivers routing + calibration + gating on structured findings.
Report generation is **excluded**, because BraTS ships co-registered
skull-stripped volumes and expert segmentations — and **no free-text reports**.
There is nothing to train a report generator on or evaluate it against.

Project-II picks up grounded generation, conditioned on the structured findings
and gate decision this project produces.

---

## Layout

```
src/cgate/
  schema.py       the fused prediction target + validation
  findings.py     mask → finding vector (train-fitted terciles)
  quality.py      IQMs, Otsu, standardiser
  calibration.py  ECE / adaptive ECE / Brier / NLL, temperature scaling
  selective.py    risk–coverage, AURC, E-AURC, gate, paired bootstrap
  fusion.py       linear / log opinion pools, uniform + confidence weights
  routing.py      quality-conditioned softmax-linear router (numpy Adam)
  pipeline.py     ingest → fit → calibrate → route → gate
  data/           synthetic phantoms, BraTS loader, feature views, splits
  experts/        expert contract + scikit-learn experts (see experts/README.md)
  experiments/    ablation grid, metrics, figures, CLI
docs/
  quality-metrics.md   IQM definitions and the skull-stripping trap
  scope.md             why report generation is not in Project-I
  findings.md          results observed so far, and what each one changed
```

---

## Read this before running the real experiment

`docs/findings.md` records two results that will affect how you set the
experiment up:

- **A 20% test split on BraTS-Africa (~29 cases) cannot resolve the effect
  size this project is looking for.** Use `--protocol external`.
- **A router fitted only on high-quality calibration data made routing *worse*
  than uniform fusion under shift** (AURC +0.06, gate coverage collapsing from
  84% to 56%). It has to extrapolate into a quality range it never saw. Run the
  `--calib-target-fraction` control before concluding anything about the
  hypothesis.

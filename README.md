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
pytest -q                                       # 193 tests, ~170s
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
| `et_present` | absent / present | no | ET volume ≥ 500 voxels |
| `ncr_present` | absent / present | **yes** | NCR volume ≥ train-fitted cut |
| `laterality` | left / right / bilateral | **yes** | WT mass about the mid-sagittal plane |
| `burden` | small / medium / large | no | WT volume tercile |
| `et_fraction` | low / mid / high | no | ET/WT ratio tercile |

Every finding is derivable from the BraTS mask alone — no extra annotation.
"Edema present" is deliberately excluded: it is true in nearly every case, so it
would inflate accuracy while telling us nothing about calibration.

`et_present` turned out to be that same case and is no longer critical. Measured
over 146 real BraTS masks it is present in 97.3% of them, and in 98.6% at a
threshold of a single voxel — no threshold makes it a question. It is still
reported; it just no longer gates a case. `ncr_present` failed for the opposite
reason, a fixed threshold an order of magnitude below the real distribution, so
its cut is now fitted on the training split like the tercile edges. See F13 in
[docs/findings.md](docs/findings.md), reproducible via `tools/prevalence.py`.

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
| CNN expert (ResNet on 2-D slices) | ✅ real, `--experts torch` |
| ViT expert (ViT-S/16 on 2-D slices) | ✅ real, `--experts torch` |
| MC dropout (B2) | ✅ real — dropout active at inference, N passes averaged |
| Slice cache (label-free selection) | ✅ implemented, tested |
| scikit-learn surrogates | ✅ retained as the no-torch fallback |
| Real BraTS loader (2021 + Africa naming) | ✅ verified against the real BraTS-Africa tree (146 cases, cohorts correct) |
| Case-table cache (ingest once, train elsewhere) | ✅ implemented, tested |

```bash
cgate --experts torch --arch resnet50 --vit-arch vit_small_patch16_224 \
      --pretrained --epochs 12                                   # real backbones
cgate --experts surrogate                                        # no torch needed
```

The CNN and the ViT take **separate** architecture flags. They shared one until
F5: `--arch resnet50` silently rebuilt the ViT as a second ResNet, and B4–B6
went on reporting themselves as CNN+ViT while fusing two copies of one
architecture. Every run now prints both backbones.

### Fitting it on the GPU you have

The device is chosen automatically (`--device` overrides). Memory, not speed, is
what usually decides whether a run is possible: the stock
ResNet-50 / 224px / batch-32 configuration needs roughly 5 GB of activations in
fp32, which a 4 GB card cannot hold.

```bash
cgate --experts torch --batch-size 8            # the cheapest lever
cgate --experts torch --amp                     # halves activation memory
cgate --experts torch --img-size 128            # discards detail, saves a lot
```

`--amp` defaults to on for CUDA and off for CPU. It roughly halves activation
memory, and it **changes the numerics** — so it is printed in the run banner and
recorded in the JSON. A row trained with AMP and a row trained without are two
different experiments, not two samples of one.

Losses stay in fp32 even under AMP: summing five cross-entropies in fp16 loses
the small gradients that the rarer finding classes depend on.

When training does run out of memory, the error names the flags in order of what
they cost the result — batch size first, architecture last — rather than
reporting only the allocation that failed.

Both families are kept. `surrogate` runs anywhere and is useful for fast
iteration on the routing/calibration/gate logic; `torch` is the one that
constitutes evidence about CNN/ViT behaviour. Every run records which was used
under `"experts"` in its JSON, because the two are **not** interchangeable as
evidence and the difference is invisible in the numbers.

> **Correction.** Earlier revisions of this file claimed torch had no wheel for
> Python 3.14. That was wrong — it was inferred from an `ImportError` (meaning
> "not installed") without checking availability. PyPI has `cp314` wheels from
> torch 2.9.0 onward. Results in `docs/findings.md` predating this were produced
> with surrogates and are flagged there as provisional.

---

## Synthetic data is not a result

`cgate` defaults to phantoms so the pipeline is runnable and CI-testable without
the multi-gigabyte download. The runner prints a banner saying so.

Phantom numbers verify that the plumbing is correct and the metrics move in the
expected direction. They are **not** evidence that quality-aware routing helps on
real gliomas. That requires:

```bash
cgate --data brats --root /path/to/BraTS2021 /path/to/BraTS-Africa
```

Train on BraTS 2021 (predominantly 3 T), evaluate the low-quality stratum on
**BraTS-Africa** (1.5 T, resource-constrained settings).

`--root` takes several paths because the two are separate downloads and often
sit on different drives. Case directories are found at any depth, so the
tumour-type level in the distributed Africa package
(`BraTS-Africa/95_Glioma/BraTS-SSA-00002-000/`) needs no rearranging.

The cohort is read from the case directory's own name and its ancestors up to
the root you passed — `BraTS-SSA-*` identifies itself. Every run prints the
counts before fitting anything:

```
Cohorts: {'high': 1251, 'low': 146}
```

**Check that line.** A mislabelled cohort does not raise; it merges the target
into the source and the headline comparison silently measures nothing. This
happened — see F5 in `docs/findings.md`. Override with one label per root when
the layout does not carry it:

```bash
cgate --data brats --root /data/BraTS2021 /data/africa --cohort high low
```

### Ingest once, train many times

Ingest reads every volume in the study and costs far more than fitting the grid,
yet it is repeated identically for every seed and every router setting:

```bash
cgate --data brats --root ... --table-cache runs/table --experts torch   # builds it
cgate --table-cache runs/table --experts torch --seed 1                  # reuses it
```

This is also how the study moves between machines. The raw archive is ~15 GB;
the table and its slice cache are a couple of GB, so ingest where the download
is and train where the GPU is:

```bash
cgate --table-cache runs/table --slice-cache /content/cache --experts torch
```

`--slice-cache` overrides the path recorded at ingest, since the location that
was right on one machine rarely exists on the other. Each cache is stamped with
a build id that the table pins, and row indices are re-derived on load, so a
table and cache that disagree raise instead of training the network on one
patient's images with another's labels — which no downstream metric can reveal.
Matching patient ids are not enough to establish a match: a second ingest into
the same directory keeps every id and replaces every pixel.

Surrogate runs need no pixels, so the few-MB table travels on its own:

```bash
cgate --table-cache runs/table --experts surrogate      # slice cache not required
```

Two things the reuse path will refuse rather than guess. `--n-slices` must agree
with the cache, which is what actually decides the slice count — otherwise a
sweep produces identical runs filed under different labels. And
`--rebuild-table` will not overwrite a table ingested from BraTS unless
`--data brats` is given, since `--data` defaults to `synthetic` and a reuse
command line carries neither it nor `--root`. Flags that ingest consumed
(`--n-cases`, `--seed`, `--root`) cannot be honoured on reuse and are listed as
ignored when they disagree with what the table was built with.

---

## Grounded report generation (Project-II)

`cgate-report` turns a run's exported test-split predictions into one
radiology-style report per case. Two properties are non-negotiable:

- **Every sentence is grounded.** Each carries evidence — a calibrated finding
  probability, an image-quality metric, or the gate decision — recorded in a
  per-sentence *grounding ledger* rendered alongside the report. A sentence
  without evidence is never emitted (`assert_grounded`).
- **The gate still decides what is released.** Deferred cases render a
  deferral notice naming the weakest critical finding, not a released report.

BraTS has no free-text reports, so this is evaluated structurally (grounding,
deferral correctness, VLM-contradiction vetoes) rather than against reference
text — see [Scope](#scope).

### Workflow

```bash
# 1. Run the grid and export per-case fused predictions for one system (B6):
cgate run --data brats --root <data> --experts torch --pretrained \
    --out runs/results.json --predictions runs/predictions.json --report-system B6

# 2. Render grounded reports (deterministic template; offline, no VLM):
pip install -e ".[vlm]"                             # adds transformers etc.
cgate-report --preds runs/predictions.json --out runs/reports

# 3. Optionally add VLM appearance phrases over a 4-sequence mosaic:
cgate-report --preds runs/predictions.json --out runs/reports \
    --vlm --slice-cache runs/cache --limit 25
```

Output per case: `<patient>.md` (report + ledger) and `<patient>.json`
(machine-readable sidecar), plus `index.md`.

### The VLM is advisory, never authoritative

The optional VLM (`--vlm`, default `google/medgemma-4b-it`, falling back to
`Qwen/Qwen2-VL-2B-Instruct`) contributes only *qualitative appearance* —
signal, margins, enhancement pattern, mass effect. It never states
presence/absence of a finding, never feeds the confidence or the gate, and
every phrase passes a **veto filter** (`grounding.VETO_RULES`) that drops any
phrase contradicting a firm calibrated finding — e.g. "no enhancement" is
vetoed when `et_present` is predicted present at p ≥ 0.7. Vetoed phrases are
listed in the report with the reason, never silently discarded. If no VLM can
be loaded, reports render deterministic-template-only.

### Layout of the module

```
src/cgate/reporting/
  grounding.py   the contract: Sentence + Evidence, ReportCase, VLM veto rules
  template.py    deterministic findings → prose (hedged by confidence band)
  vlm.py         4-sequence mosaic, VLM inference, phrase parsing
  render.py      assembles report + ledger, JSON sidecar
  cli.py         `cgate-report`
```

---

## Scope

Project-I delivers routing + calibration + gating on structured findings.
Report generation was **excluded from Project-I**, because BraTS ships
co-registered skull-stripped volumes and expert segmentations — and **no
free-text reports**. There is nothing to train a report generator on or
evaluate it against.

Project-II picks up grounded generation, conditioned on the structured
findings and gate decision this project produces — and is now implemented
(see [Grounded report generation](#grounded-report-generation-project-ii)).
Because there is no reference text, generated reports are audited by a
*grounding ledger* (every sentence traced to a probability, quality metric or
the gate decision), not by text-similarity metrics against a gold report that
does not exist.

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
  reporting/      grounded report generation (Project-II, see cgate-report)
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
- **Routing came out *worse* than uniform fusion under shift** (AURC +0.05 to
  +0.06). Two causes were isolated, and only one is fixed:
  - the router never saw the target domain while being fitted →
    `--calib-target-fraction 0.15` (fixes the gate collapse, 56% → 89% coverage)
  - **the router is overparameterised**: 120 parameters fitted on ~40
    calibration cases. Its confidence *ranking* is fine (E-AURC 0.164 vs 0.168);
    its *accuracy* is not (0.437 vs 0.513) → `--router-shared` cuts it to 24
    parameters.

**Run the low-capacity router before reporting a negative result.** A model with
three times more parameters than training cases tells you about capacity, not
about the hypothesis. The run log prints `params_per_sample` and warns.

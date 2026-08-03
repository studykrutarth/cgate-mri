# Findings log

Results observed while building the system. **All numbers below are from
synthetic phantoms**, so they are statements about the *machinery*, not about
gliomas. They are recorded because each one changed a design decision or
predicts a problem the real experiment will hit.

---

## F1 — The ablation grid separates heterogeneity from ensembling

Stratified protocol, 400 phantoms, 80 test cases:

| Group | acc | AURC |
|---|---|---|
| B0–B3 (single expert / homogeneous ensemble) | 0.50–0.53 | 0.32–0.37 |
| B4–B6 (heterogeneous experts) | 0.60–0.61 | 0.24–0.26 |

**Heterogeneity, not ensembling, is what moves the numbers.** B3 (three copies
of one architecture, different seeds) sits with the single-expert group;
B4 (three *different* experts, uniform weights) jumps.

This is exactly the separation the grid was built to expose, and it is the
answer to "isn't this just a deep ensemble?" — no, and B3 is the row that
proves it.

## F2 — 28 test cases cannot resolve the effect, and real data is no better

Same run, low-quality stratum:

```
B6 E-AURC 0.094  vs  B4 E-AURC 0.127        (direction favours routing)
dAURC = -0.0334, 95% CI [-0.1342, +0.0567], p = 0.471
```

Directionally right, statistically nothing. The stratified split left only 28
low-quality test cases.

**This is not a synthetic-data problem.** BraTS-Africa has ~146 cases in total,
so a 20% test split leaves ~29 — the identical wall, on real data, after the
full training effort.

**Design consequence:** added `--protocol external`, which holds out the entire
target cohort as test (140 phantom cases; ~146 real BraTS-Africa cases). Roughly
5× the test set at no cost, and it matches the deployment question.

## F3 — Routing fitted only on the source cohort *hurts* under shift

External protocol, 400 phantoms, train+calibrate on high-quality only, test on
140 low-quality:

```
B4 (uniform)        AURC 0.3304   gate coverage 84.3%
B6 (quality-routed) AURC 0.3925   gate coverage 56.4%
dAURC = +0.0621, 95% CI [-0.0117, +0.1339], p = 0.088
```

Positive ΔAURC means **routing was worse than uniform fusion**, and the effect
is near-significant in the wrong direction. Note also the gate collapse: B6
deferred 61 of 140 cases against B4's 22.

**Diagnosis.** Everything post-hoc — per-expert temperatures, the router, the
gate threshold — was fitted on calibration data drawn entirely from the
high-quality cohort. `QualityStandardizer` deliberately keeps *training*
statistics (so real shift stays visible rather than being normalised away), so
degraded scans arrive as large z-scores. A softmax-linear router evaluated far
outside its fitted range saturates and collapses weight onto one expert — which
is precisely the failure `RouterConfig.entropy_reg` exists to damp, and it
defaults to 0.

**Why this matters more than the phantom numbers:** the real experiment has this
structure by construction. Train on BraTS 2021, test on BraTS-Africa, and the
router never sees a 1.5 T scan while being fitted. Left unaddressed, the headline
comparison would likely come out *against* the hypothesis for a reason that has
nothing to do with whether quality-aware routing is a good idea.

**Design consequence:** added `--calib-target-fraction`. It lends a share of the
target cohort's patients to calibration (they leave the test set, so the two stay
disjoint). This mirrors what a deployment can realistically obtain — a small
labelled sample from the new site — and turns F3 into a controlled comparison:

```bash
cgate --protocol external --calib-target-fraction 0.0    # router extrapolates
cgate --protocol external --calib-target-fraction 0.15   # router has seen the domain
```

If routing only helps in the second condition, that is a publishable, honest
claim with a clearly stated requirement, not a failure.

Other levers worth testing before concluding, in rough order of cost:
`RouterConfig.entropy_reg > 0` to prevent weight collapse; clipping standardised
quality features to the observed training range to bound extrapolation; and
per-finding vs shared router weights (`per_finding=False`) to cut parameters
when calibration data is scarce.

## F4 — `ncr_present` is close to degenerate on phantoms

Prevalence 0.88 / 0.12. A finding that lopsided contributes little and inflates
apparent accuracy. `label_prevalence` is printed on every run for this reason.

**Check this on real BraTS before trusting any per-finding number**, and consider
dropping the finding from the critical set if it stays this skewed. The critical
set is what case-level confidence takes its minimum over, so a near-constant
finding there mostly adds noise to the gate.

---

## Standing caveats

- Every number here is from phantoms with **surrogate** CNN/ViT/MC-dropout
  experts. Treat them as tests of the machinery.
- Single seed. Re-run across several `--seed` values before believing any gap;
  the bootstrap CI covers sampling noise in the test set, not seed variance in
  training.
- Case-level accuracy is the **conjunction** of three critical findings being
  simultaneously correct, so it reads much lower than per-finding accuracy.
  Per-finding numbers are in the JSON under `per_finding`.

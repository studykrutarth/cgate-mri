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

**First diagnosis (partly wrong, kept for the record).** That everything
post-hoc was fitted on high-quality cases only, so the router had to extrapolate
its quality-to-weight mapping into an unobserved range and saturated. The gate
collapse looked like evidence for it.

**The control that falsified it.** Lending 15% of the target cohort to
calibration (`--calib-target-fraction 0.15`, F3b below) fixed the gate collapse
but *not* the AURC regression. So extrapolation explains at most part of it.

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

## F3b — The control fixed the gate but not the regression

`--protocol external --calib-target-fraction 0.15`, 119 test cases:

| | acc | AURC | **E-AURC** | gate coverage |
|---|---|---|---|---|
| B4 uniform | 0.513 | **0.3149** | 0.1680 | 80.7% |
| B6 routed | 0.437 | 0.3673 | **0.1637** | 89.1% |

ΔAURC = +0.0524, 95% CI [-0.022, +0.122], p = 0.188.

The gate collapse is gone (56% → 89% coverage), so giving the router
target-domain calibration data did help *that*. But routing is still behind on
AURC, which falsifies extrapolation as the whole story.

**The decomposition is the clue.** E-AURC subtracts the AURC of a perfect
confidence ranking, isolating ranking quality from accuracy:

- B6 **E-AURC 0.1637 < B4 0.1680** → B6's confidence *ranking* is marginally
  **better**
- B6 **accuracy 0.437 < B4 0.513** → B6's *predictions* are worse

So the router is not mis-ranking confidence. It is choosing worse expert
weights, and the accuracy loss drags total AURC down with it.

## F3c — The router is overparameterised for the calibration split

Counting parameters: `n_groups × n_experts × (n_features + 1)`
= 5 findings × 3 experts × (7 IQMs + bias) = **120 parameters**.

Fitted on `calib_fit`, which is *half* the calibration split by design (the
other half sets the gate threshold). Under the external protocol that is
**roughly 32–43 cases**.

**120 parameters fitted on ~40 cases.** The router can fit calibration noise
while selecting worse experts on unseen data — which is exactly the signature
observed: training loss falls, confidence ranking stays fine, accuracy drops.

**Design consequence:** `QualityRouter.fit` now counts its own parameters and
records `params_per_sample` and an `overparameterised` flag in the run log; the
CLI prints a warning and suggests the fix. Two levers exposed:

```bash
cgate --router-shared     # share weights across findings: 120 -> 24 params
cgate --router-l2 0.1     # stronger penalty
```

Before reporting that quality-aware routing does not work, the low-capacity
router must be run. A negative result from a model with three times more
parameters than training cases is a statement about capacity, not about the
hypothesis.

## F3d — Cutting capacity fixes accuracy but trades away ranking

`--protocol external --calib-target-fraction 0.15 --router-shared`, 119 test cases:

| Config | params | acc | AURC | E-AURC |
|---|---|---|---|---|
| B4 uniform | — | 0.513 | **0.3149** | 0.1680 |
| B6 per-finding router | 120 | 0.437 | 0.3673 | **0.1637** |
| B6 shared router | 24 | **0.504** | 0.3677 | 0.2151 |

ΔAURC (shared) = +0.0528, 95% CI [-0.006, +0.101], p = 0.071.

F3c is **confirmed as far as accuracy goes**: cutting parameters 5× restored
B6's accuracy from 0.437 to 0.504, essentially matching uniform fusion. So the
accuracy loss really was overfitting.

But the confidence ranking degraded in exchange (E-AURC 0.164 → 0.215), and
total AURC did not move. There is a capacity trade-off here:

- **high capacity** — ranks confidence well, predicts badly (overfits weights)
- **low capacity** — predicts well, ranks badly (too coarse to exploit quality)

Neither configuration beats uniform fusion on this testbed.

## Where this leaves the hypothesis

**On synthetic phantoms, quality-aware routing does not beat uniform
heterogeneous fusion under distribution shift, in any of the three
configurations tested.** ΔAURC stayed positive (0.052–0.062) with p between 0.07
and 0.19 — consistently the wrong direction, never significant.

That is a genuine negative result *for this testbed*, and it should be stated
plainly rather than tuned away. Two honest readings, and the real experiment
distinguishes them:

1. **The hypothesis is weak.** Heterogeneity captures most of the available gain
   (F1) and conditioning the weights on quality adds nothing on top.
2. **The testbed cannot express it.** The three feature views may not be
   *differentially* informative enough for routing to have anything to exploit,
   and the surrogate experts are far weaker than real CNN/ViT backbones. The
   phantom corruption model is also mine, so the very structure routing needs
   is something I designed rather than observed.

Reading 2 is not an excuse to dismiss reading 1. It is a reason the conclusion
must come from BraTS 2021 → BraTS-Africa with real backbones, where the quality
gap is measured rather than simulated.

**What the deck should claim, given this:** the falsifiable-comparison design is
the contribution, and B4 vs B3 (heterogeneity beats ensembling, F1) is currently
the only effect with any support. Do not promise that routing will win.

Still untested, in rough order of expected value: `RouterConfig.entropy_reg > 0`
to damp weight collapse; clipping standardised quality features to the training
range; a per-finding router with much stronger `--router-l2` (an intermediate
capacity between the two rows above); and multiple seeds, since every number
here is a single seed.

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

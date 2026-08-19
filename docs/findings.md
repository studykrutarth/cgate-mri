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

## F5 — Three bugs in the torch path, found before the first real run

Preparing the BraTS 2021 → BraTS-Africa run surfaced three defects that all
shared a property: each produced plausible numbers rather than an error, and
each would have invalidated a different part of the grid.

**1. Every BraTS-Africa case was labelled `cohort="high"`.** The distributed
package nests a tumour-type level the loader did not expect:

```
PKG - BraTS-Africa/BraTS-Africa/95_Glioma/BraTS-SSA-00002-000/
```

The loader walked exactly two levels and read the cohort from the case's
immediate parent — `95_Glioma`, which carries no Africa marker. Measured on the
real download: 146 cases found, 146 tagged `high`. Pointed one level higher it
found **zero** cases instead. Under `--protocol external` this does not fail; it
silently merges the target cohort into the source cohort, and the headline
comparison then measures nothing at all.

Cases are now discovered at any depth, and the cohort is read from the case
directory's own name and every ancestor up to the supplied root.
`BraTS-SSA-00002-000` identifies itself, so the answer no longer depends on how
the archive was unpacked.

**2. `--arch resnet50` silently rebuilt the ViT as a second ResNet-50.**
`ViTExpert` took the shared config whole, including its `arch` field, so B4–B6
fused radiomics with two copies of one architecture while still being reported
as "Radiomics + CNN + ViT". That removes exactly the heterogeneity those rows
exist to isolate — and F1, the one result with any support, is the heterogeneity
result. The ViT now carries its own `--vit-arch` / `--vit-img-size`.

**3. B3's three-seed deep ensemble was three identical models.** Every expert in
a row is handed the same `TorchExpertConfig` instance, and each constructor did
`cfg.seed = seed`, writing through to all of them. The last expert constructed
set the seed for the whole row, so B3's members trained from one initialisation
on one batch order. B3 is the control that separates ensembling from
heterogeneity; collapsed to a single model it makes B4-vs-B3 unreadable.
Configs are now copied per expert, never mutated.

**Consequence for the record:** `runs/torch_run{1,2,3}.json` were produced with
all three defects present and are void. They were smoke runs (64 cases, 12 test
cases, 1 epoch) and were never cited as results, but B3 scoring `acc=0.000,
AURC=1.000` in them is now explained rather than mysterious.

**What this says about the surrogate findings above:** nothing directly — F1–F3d
were produced with `--experts surrogate`, which touches none of this code. It
does mean the torch path had never actually been exercised as evidence, only as
a smoke test, which is why three defects survived in it.

---

## F6 — Six more defects, found by reviewing the F5 fixes

An adversarial review of the F5 changes raised twelve candidates across four
lenses. Its verification stage never ran — every verify agent died on a session
limit — so all twelve were adjudicated by hand against the code. Six were real
and are fixed; the rest were duplicates or restatements of accepted behaviour.

**1. B2 was not MC dropout.** `MCDropoutExpert` did
`config or TorchExpertConfig(dropout=0.5)`, which reaches 0.5 only when no
config is passed at all — and `run.py` always passes one, with `dropout` at the
shared 0.3. So the row whose entire purpose is sampling-based uncertainty
trained at the same rate as B0/B1/B3, too small for the passes to move apart,
and came out indistinguishable from B0 (ECE 0.1442 vs 0.1440 in the smoke run)
with the rate appearing nowhere in the log. This is bug 2 of F5 — a shared
config silently overriding a row's defining parameter — left in place for
`dropout` while being fixed for `arch`. The rate now has its own `mc_dropout`
field. After the fix B2 separates from B0 on its own gate operating point.

**2. A reused table could reconnect to the wrong slice cache.** `load`
re-derived row indices via `rows_for`, which only checks that every patient id
*exists* in the cache. A second ingest into the same directory overwrites the
`.npy` files while keeping every id, so the table reconnected happily and the
network would train on one study's pixels under another's labels — undetectable
downstream. The default cache path is `<--out parent>/cache`, shared by every
run writing to one output directory, and synthetic ids are positional
(`SYN-0000`), so this needed nothing unusual. Caches now stamp a `build_id` at
write time and tables pin it; a mismatch raises. Table format bumped to v2.

**3. `--n-slices` was silently ignored on reuse.** The cache fixes the slice
count and torch reads `cache.n_slices`, but the run log recorded the flag. A
sweep over `--n-slices` against a cached table therefore produced byte-identical
results filed under different labels. A mismatch now aborts.

**4. `--rebuild-table` could destroy a real ingest.** The flag re-enters the
ingest branch keyed on `--data`, which defaults to `synthetic`, and a reuse
command line deliberately carries neither `--data` nor `--root`. Adding
`--rebuild-table` to it replaced a multi-hour BraTS table with 30 phantoms and
reported success. Refused when the saved provenance says `brats`.

**5. A cache-free machine could not use the table at all.** `load` opened the
recorded slice cache unconditionally, so the workflow the table exists for —
ingest beside the ~15 GB archive, copy the few-MB table elsewhere, train there —
died with `FileNotFoundError` for surrogate runs, which never read a pixel.
`require_slices=False` now tolerates a missing *recorded* cache; an explicitly
requested one still must exist.

**6. Directory links broke discovery both ways.** `is_dir()` follows junctions
and symlinks, so a mirrored link made one case reachable by several paths and
yielded duplicate patient ids — which land in different splits and survive
`assert_no_leakage`, because by then they are simply two rows. And
`_names_up_to` resolved both paths before `relative_to`, so a link anywhere
between root and case put them on different branches, the ancestor scan raised,
the exception was swallowed, and every case fell back to `"high"` — F5 bug 1
reintroduced for linked trees. Discovery now dedupes on resolved paths; the
ancestor scan tries the unresolved form first.

**Accepted, not fixed:** `cohort_for_case` uses `any` over the case name *and*
its ancestors up to and including `root`, so pointing `--root` at a mixed
directory named for one cohort marks everything beneath it. Removing `root` from
the scan would break the ordinary case (`--root .../BraTS-Africa`), which is
what `infer_cohort` is for. The docstring now states the trade-off, and the run
prints cohort counts before fitting anything.

**Consequence for the record:** no cited result changes. F1–F3d are surrogate
runs, which touch none of this. The F5 smoke runs were already void.

---

## F7 — BraTS 2023 renumbered enhancing tumour, and it aligned with the cohort split

The first real-data run. 95 BraTS-Africa gliomas, GPU, full grid. It reported:

```
et_present    absent=1.00  present=0.00
et_fraction   low=1.00     mid=0.00   high=0.00
```

No enhancing tumour in any of 95 gliomas, which is not a plausible cohort.

**Cause.** BraTS 2021 labels enhancing tumour `4` and leaves `3` unused;
`schema.py` encoded that, with a comment saying so. The 2023 challenge
renumbered ET to `3`, and BraTS-Africa ships in the 2023 format. Confirmed
directly on the files:

| cohort | NCR | ED | ET |
|---|---|---|---|
| BraTS 2021 | 1 | 2 | **4** |
| BraTS-Africa | 1 | 2 | **3** |

So `region_mask(mask, "ET")` matched nothing on every Africa case, and `TC`
silently lost its enhancing component too.

**Why this one was dangerous rather than merely wrong.** The artefact does not
fall randomly across the study — it falls *exactly* on the cohort boundary.
BraTS 2021 is the high-quality training cohort and parses correctly; BraTS-Africa
is the low-quality test cohort and would have had two of its five findings
constant. Under `--protocol external` that reads as a model which handles
enhancing tumour well at 3 T and fails completely at 1.5 T — a large, clean,
entirely fictitious quality effect, pointing the same way as the hypothesis, and
agreed on by every metric in the grid. Accuracy, AURC, ECE and the gate would
all have corroborated it. Nothing in the pipeline could have caught it
downstream, because the labels and the images would have been perfectly
consistent with each other.

**Fix.** `canonicalise_labels` in the loader renumbers 3 to 4 on read, so
everything downstream keeps one convention. A mask containing both raises
rather than guessing. Verified on the real files: ET now measures 21k-82k voxels
per case where it previously measured zero.

**Consequence for the record:** the pilot run is void and is being re-run. No
earlier result is affected — F1-F3d are synthetic, where the generator emits
label 4 directly.

**What it says about the remaining ingest:** nothing else in the study is
verified against real data yet. This one surfaced within minutes of the first
real cohort touching the code; the honest expectation is that more will.

---

## F8 — The cohorts differ in more than image quality

Measured on the ingested 495-case study (400 BraTS 2021, 95 BraTS-Africa):

| cohort | n | median WT | median ET | median ET/WT |
|---|---|---|---|---|
| high (BraTS 2021) | 400 | 87,008 | 16,672 | 0.215 |
| low (BraTS-Africa) | 95 | **162,872** | 31,100 | 0.253 |

**Africa gliomas present at roughly twice the whole-tumour volume.** The
enhancing fraction is similar, so this is a difference in size at presentation,
not in tumour composition — consistent with later presentation in
resource-constrained settings.

`burden` is the WT tercile, and its edges are fitted on `train` only, which
under the external protocol is the source cohort alone. That is the correct
split discipline, and it produces:

| | small | mid | large |
|---|---|---|---|
| high (train) | 0.33 | 0.33 | 0.34 |
| **low (test)** | 0.17 | 0.16 | **0.67** |

Two thirds of the test cohort lands in one class. A model that always answers
"large" scores 0.67 on that finding, and any accuracy difference between cohorts
is then partly a class-prior shift that has nothing to do with acquisition
quality. `et_fraction` shifts far less (0.31 / 0.27 / 0.42).

**What is and is not affected.** Case confidence — the gated quantity, and what
AURC ranks — is the **minimum over the critical findings**, which are
`et_present`, `ncr_present` and `laterality`. `burden` and `et_fraction` are not
critical, so the headline B6-vs-B4 comparison is insulated from this. Aggregate
**accuracy** is not: it averages all five findings, so per-finding accuracy has
to be reported for the low stratum or the burden prior will be read as a
quality effect.

**The wider point, which is not a bug and cannot be fixed in code.** The project
states its cohorts differ in *quality*. They also differ in *presentation*, and
the two are confounded by construction, because both follow from where the scans
were acquired. "Routing helps in the low-quality stratum" is therefore not
separable from "routing helps on larger tumours from Sub-Saharan sites" on this
data alone. Whatever the headline comes out as, that limitation belongs in the
report next to it. Distinguishing them would need either a size-matched
subsample — which the case counts do not support — or a cohort that varies
quality while holding presentation fixed, which is what a phantom or a
degradation study provides and BraTS does not.

Also worth noting: `et_present` is 0.95 present across the whole study, and
0.98 in the Africa cohort. It is a **critical** finding, so it enters the
confidence minimum, while carrying almost no discriminative signal. That is F4's
`ncr_present` observation reappearing on real data on a different finding, and
whether it earns critical status should be revisited.

---

## F9 — The two cohorts store their voxels in opposite order

First real external-protocol run: train on 300 BraTS 2021, test on 81
BraTS-Africa. Per-finding accuracy on the test cohort:

| row | et_present | ncr_present | laterality | burden | et_fraction |
|---|---|---|---|---|---|
| B0 | 0.975 | 0.840 | 0.333 | 0.728 | 0.346 |
| B3 | 0.975 | 0.840 | 0.333 | 0.753 | 0.420 |
| B4 | 0.975 | 0.840 | 0.309 | 0.765 | 0.469 |
| B5 | 0.975 | 0.840 | 0.346 | 0.778 | 0.543 |
| B6 | 0.975 | 0.840 | 0.284 | 0.728 | 0.506 |

`laterality` sits at 0.28-0.36 against a **0.45 majority-class baseline**. Below
the constant predictor is not "hard" — it is anti-correlated, which means a sign
error somewhere.

**Cause.** Orientation codes, sampled 40 cases per cohort:

```
BraTS 2021    (L,P,S) x40          left_is_low_index = True
BraTS-Africa  (R,A,S) x38          left_is_low_index = False
              (L,P,S) x2
```

The cohorts run in opposite left-right *and* anterior-posterior order, so their
axial slices are 180 degrees apart — and BraTS-Africa is not even internally
consistent. The loader reads laterality from each affine, so **both cohorts'
labels are correct**. Nothing canonicalises the pixels, and the slice cache
stores them as they come.

**Why the correct labels are what make it dangerous.** A label bug would show up
as disagreement somewhere. Here every label matches its own image, every
consistency check passes, and the damage is entirely in the train/test relation:
the network learns "tumour at low index means left" from BraTS 2021 and is
tested on a cohort where that position is the other side. Under the external
protocol that inverts laterality on 95% of the test set — and mirrors every
other spatial feature the CNN learned along with it, which is why `burden` and
`et_fraction` are also weak.

This is the failure the module docstring already warned about, applied to the
wrong half of the problem: orientation was read for the *labels* and never
applied to the *images*.

**Fix.** Every volume and mask is reoriented with `nib.as_closest_canonical` on
load, so the cache holds one frame. Confirmed: both cohorts now report
`left_is_low_index = False`. The old test pinning the flag to the affine's sign
encoded the pre-fix contract and was replaced — what must still follow the file
is the anatomical label, which is now pinned separately.

**Consequence for the record:** the study400 run is void. Its slice cache holds
mirrored pixels, so this needs a re-ingest, not just re-training.

**Two other things this run established, which the re-run should confirm:**

1. `et_present` (0.975) and `ncr_present` (0.840) are **constant predictions** —
   identical across all seven systems, matching the class priors exactly. Both
   are *critical* findings, and case confidence is the minimum over the critical
   set. With the third critical finding (`laterality`) broken by the above, the
   gated quantity was built from three uninformative findings. That is why the
   gate does nothing: B0 defers 51 of 81 cases to move error 74.1% -> 73.3%.
   The two findings that *do* carry signal, `burden` and `et_fraction`, are
   marked non-critical and excluded from confidence. Whether that assignment is
   right is now an open design question, not a bug.

2. Heterogeneity shows up on real data anyway. On `et_fraction`, the rows with
   radiomics + CNN + ViT (B4 0.469, B5 0.543, B6 0.506) clearly beat the
   CNN-only rows (B0-B2 0.346). That is F1 reproducing outside the phantoms,
   with the mirroring still in place.

**The headline from this run is not reportable**, but for the record it was
dAURC = +0.0144, 95% CI [-0.0581, +0.0733], p = 0.770 — no significant
difference, with B6 holding the best E-AURC of any row (0.2992 vs B4's 0.3195).
That is the F3b split again — routing ranks better, predicts worse — this time
with a correctly sized router (24 parameters on 57 cases, 0.42 per case), so
F3c's capacity explanation does not apply.

---

## F10 — Left and right were swapped from the beginning

Found while asking why `laterality` still only reached its majority baseline
after F9. Derived from the affine rather than argued:

```
identity affine, axcodes (R,A,S):
  world x at voxel index 0  = +0
  world x at voxel index 10 = +10
  world space is RAS+, so +x is anatomical RIGHT
  => the LEFT hemisphere sits at the LOW indices  -> True
  left_is_low_index() returned                    -> False
```

`return code == "L"` should be `code == "R"`. An axcode names the direction an
axis increases *toward*, so an axis running toward the right puts the left
hemisphere at the low indices. **Every laterality label the project has ever
produced was mirrored.**

**Why nothing caught it.** A left/right swap is a permutation of two classes.
The classes stay near-balanced (0.44 / 0.41 / 0.15), so accuracy, ECE, AURC and
the gate are all exactly unchanged — the network simply learns the mirrored
mapping and scores identically. This is the failure mode the module docstring
names in its opening paragraph, and it was present in the function that
paragraph describes.

**It does not invalidate the measurements.** A consistent relabelling of two
classes leaves every metric where it was, so the F9 run's numbers stand as
measurements. What was wrong is the *meaning*: every "tumour on the left" the
system produced described a tumour on the right. For a system whose entire
output is a radiology finding, that is the most dangerous class of error
available — clinically inverted, metrically invisible.

**My part in it.** Earlier in the same session a test asserted `False` for the
LAS case and failed. I judged the test wrong and rewrote it to match the code,
reasoning that "increasing index moves toward anatomical left" implied left was
the low index. It implies the opposite. The test that existed to catch this bug
was changed to encode it.

The replacement test computes the expected answer from the affine instead of
restating the convention in prose, so the same mistake cannot be made twice by
reasoning about it.

### F10b — and the value was never plumbed through anyway

Checking what a re-ingest would actually change, the fix altered nothing: 17
real cases across both cohorts gave laterality 0 -> 0, 1 -> 1, 2 -> 2.

`pipeline.py` called `measure(case.mask)` with no orientation argument, so it
used the default (`spatial_axis=0`, `left_is_low_index=True`) for every case in
the study. The loader computed `left_is_low_index(affine)` and stored it in
`case.truth`; nothing read it. `laterality_split`'s own docstring asserts that
"`cgate.data.brats` does that and passes the correct value through" — it
computes it and does not pass it.

So the real history is not the one F10 tells:

| | orientation | flag used | correct? |
|---|---|---|---|
| pre-F9 BraTS 2021 | LPS | True (default) | **no** — mirrored |
| pre-F9 BraTS-Africa | RAS | True (default) | yes |
| post-F9 both | RAS | True (default) | yes |

The two cohorts were labelled *against each other* before F9, on top of the
pixel mirroring. After F9 everything is canonical RAS, where the default
happens to be right — so the study400_ras labels are correct, but by
coincidence rather than by construction.

**Consequences.** No re-ingest is needed: the existing table's laterality is
already correct, and the F9 run's numbers stand. The inverted
`left_is_low_index` was still a real bug and is still fixed; it was simply dead
code with respect to the pipeline. Ingest now passes each case's `lr_axis` and
`left_is_low_index` through, so correctness no longer depends on every dataset
arriving in one convention.

The lesson is narrower than F10's and worth more: a value computed carefully,
stored, documented as being used, and never read. The test that would have
caught it is not a test of the convention but a test that two identical
anatomies stored in opposite orders come out of *ingest* with the same label —
which is what now exists.

---

## F11 — Why the gate cannot work: the critical findings are the useless ones

Each finding against the best **constant** predictor (its class marginal in the
test cohort), which is the honest baseline — not uniform:

| finding | critical | acc | const acc | Brier | const Brier | signal |
|---|---|---|---|---|---|---|
| `et_present` | yes | 0.975 | **0.975** | 0.043 | 0.048 | +0.005 |
| `ncr_present` | yes | 0.840 | **0.840** | 0.252 | 0.269 | +0.017 |
| `laterality` | yes | 0.432 | 0.440 | 0.679 | 0.616 | **-0.063** |
| `burden` | no | 0.765 | 0.670 | 0.345 | 0.497 | **+0.151** |
| `et_fraction` | no | 0.556 | 0.420 | 0.551 | 0.655 | **+0.104** |

The two binary criticals match their constant baseline to three decimals.
`laterality` is *worse* than one. The two findings carrying real signal are both
marked non-critical.

Case confidence is the **minimum over the critical findings**, so it is computed
from two constants and one noise source — and the minimum is almost always the
noise. That is the mechanism behind the gate: B0 defers 38 of 81 cases to move
error 75.3% -> 65.1%, and B6 makes it *worse*. No amount of routing can fix a
confidence signal assembled from findings that carry none.

### The laterality cause: the augmentation was mirroring the hemispheres

A hypothesis worth recording as refuted first: that global pooling in the
backbone discards absolute position, making laterality unlearnable. Tested
directly — feeding an image and its mirror through both encoders gives
max|diff| 0.41 (resnet18) and 2.09 (vit_tiny). Learned kernels are not
reflection-equivariant, so the features distinguish the two fine. Not the cause.

The actual cause:

```python
# Anterior-posterior flips only -- see TorchExpertConfig.
imgs[flip] = imgs[flip][..., ::-1, :]     # flips axis -2 = H
```

A cached slice is `(C, H, W) = (C, X, Y)` because `_extract` takes along
`slice_axis=2` (z), and canonical RAS makes **X the left-right axis**. So the
anterior-posterior axis is the last one, and `[..., ::-1, :]` was a *left-right*
flip — exactly what `TorchExpertConfig.augment` documents as forbidden:

> A left-right flip would mirror the hemisphere and silently invert the
> `laterality` label while the target stayed unchanged -- corrupting the one
> finding that depends on spatial orientation, invisibly.

Half of every training batch had its hemispheres swapped with the label left
alone, so laterality was trained on coin flips. It was never hard; it was
supervised with noise, which is why it lands below its own constant predictor
while every other finding looks plausible.

**This is the fourth bug in a row whose signature is "correct labels, wrong
pixels, no metric moves"** — F9 (cohorts mirrored against each other), F10 (the
convention inverted), F10b (the value never plumbed through), and now the
augmentation undoing all of it. The common thread is that laterality is the only
finding with a spatial precondition, and the class balance hides every violation
of it. The tests that now exist assert the *property* — same anatomy, same
label; augmentation preserves hemispheric mass — rather than restating the
convention.

**Fixed**: the flip now targets the last axis. Needs re-training only; the slice
cache is unaffected because augmentation happens after the cache is read.

**Still open, and a design question rather than a bug:** `et_present` at 0.95
prevalence and `ncr_present` at 0.86 cannot carry much signal on 81 test cases
whatever the model does, yet they gate every case. Either the critical set
should follow the evidence (`burden` and `et_fraction` carry it), or the gate
should not be a hard minimum over findings that are near-constant by nature.
That decision should be made explicitly and recorded, not inherited.

---

## F12 — The first result the system is entitled to produce

Re-trained on the same table after the F11 augmentation fix. Nothing else
changed: same cases, same splits, same slice cache, training only.

| | before F11 | after F11 |
|---|---|---|
| `laterality` (B0) | 0.321 | **0.642** |
| case accuracy (B0) | 0.247 | **0.506** |
| AURC (B6) | 0.773 | **0.224** |
| E-AURC (B4) | 0.343 | **0.085** |
| B4 gate | 72.8% -> 69.7% (+4.3%) | 50.6% -> **21.4%** (+57.7%) |

`laterality` went from below its constant predictor (0.44) to 0.64, confirming
F11's diagnosis: it was never hard, it was supervised with coin flips.

**The gate now does what it exists to do.** At ~52% coverage every row roughly
halves the error, B4 from 50.6% to 21.4% and B6 from 45.7% to 20.9%. Before the
fix the best row moved error by 4%.

### The headline

```
B6 (quality-routed) vs B4 (uniform fusion), paired bootstrap n=2000
dAURC = -0.0214   95% CI [-0.0635, +0.0087]   p = 0.185
```

Negative favours B6, so **for the first time in this project routing is
directionally better than uniform fusion.** Every previous measurement — F3
(+0.062), F3b (+0.052), F3d (+0.053) on phantoms, and both pre-F11 real runs —
put it the wrong way. The CI has also tightened to about half its previous
width.

**It is still not significant.** The interval crosses zero and p = 0.185, so the
honest statement remains "routing did not measurably beat uniform fusion". What
changed is that the prior negative results are now known to have been measured
through a broken laterality channel, so they should not be cited as evidence
against the hypothesis either.

### What the rows say

| | acc | AURC | E-AURC | gate |
|---|---|---|---|---|
| B0 single CNN | 0.506 | 0.290 | 0.138 | 49.4 -> 31.7 |
| B4 uniform fusion | 0.494 | 0.246 | **0.085** | 50.6 -> 21.4 |
| B6 quality-routed | **0.543** | **0.224** | 0.096 | 45.7 -> 20.9 |

**Heterogeneity is doing the heavy lifting**, as F1 predicted on phantoms:
B4 cuts E-AURC from B0's 0.138 to 0.085 with no accuracy gain, i.e. the
improvement is entirely in confidence *ranking*. Routing then adds accuracy
(0.494 -> 0.543) and total AURC (0.246 -> 0.224) while giving back a little
ranking (E-AURC 0.085 -> 0.096) — the same accuracy/ranking trade F3b found,
now with routing on the winning side of the total.

### What still blocks a claim

1. `et_present` (0.975) and `ncr_present` (0.840) are **unchanged and still
   exactly their constant baselines**. Two of the three critical findings
   carry no signal; only `laterality` now does. The gate works despite them,
   not because of them, and the design question from F11 stands.
2. F8's confound is untouched: the low cohort also has twice the tumour volume,
   so "helps under degraded quality" is not separable from "helps on larger
   tumours" on this data.
3. 81 test cases and 400 of 1251 available training cases. p = 0.185 with a CI
   this wide is an underpowered study, not a null one — the full ingest is the
   obvious next lever, and F2 already established that test-set size is what
   binds.

---

## Standing caveats

> **F1-F3d are provisional and must be re-run.** Every number above was produced
> with `--experts surrogate`: scikit-learn MLPs on three hand-built feature views
> (texture / local / global) that *I designed* to degrade under different
> acquisition defects. Routing was therefore tested on a bench where the
> differential-degradation premise it depends on was authored rather than
> observed — so "routing does not help" is at least as much a statement about
> the bench as about the hypothesis.
>
> Real ResNet/ViT backbones (`--experts torch`) have genuinely different
> inductive biases and give that premise its first fair test. Re-run the grid
> before drawing any conclusion:
>
> ```bash
> cgate --experts torch --arch resnet50 --pretrained --n-cases 400
> ```

- Every number here is from phantoms with **surrogate** CNN/ViT/MC-dropout
  experts. Treat them as tests of the machinery.
- Single seed. Re-run across several `--seed` values before believing any gap;
  the bootstrap CI covers sampling noise in the test set, not seed variance in
  training.
- Case-level accuracy is the **conjunction** of three critical findings being
  simultaneously correct, so it reads much lower than per-finding accuracy.
  Per-finding numbers are in the JSON under `per_finding`.

# Scope: why report generation is not in Project-I

## The blocking fact

**BraTS contains no radiology reports.**

BraTS ships co-registered, skull-stripped multi-parametric MRI volumes (T1,
T1ce, T2, FLAIR) plus expert voxel-level segmentation labels for three tumour
subregions. That is all. There is no free-text findings section, no impression,
no report of any kind.

So a report generator trained on BraTS has nothing to learn from and nothing to
be evaluated against. Reporting BLEU, ROUGE or RadFact against a target that
does not exist is not possible.

## Options considered

**1. Template-generated pseudo-reports.** Emit sentences from the mask
("enhancing tumour in the left temporal lobe, whole-tumour volume 34 mL").
Workable, but the task collapses to structured description with a text
renderer on top — the language model is not learning to draft, it is filling a
template. Presenting that as "report generation" would overclaim.

**2. Swap datasets.** Move the generation task to a corpus that has reports
(MIMIC-CXR) and keep BraTS for routing and uncertainty. This means two
unrelated datasets, two modalities, and no shared evaluation — the confidence
gate would be measured on brain MRI and the generation on chest X-ray, so the
central claim (gating *before* generation) would never be tested end-to-end.

**3. Drop generation from Project-I.** ← chosen

## What Project-I delivers instead

The routing and gating half, done properly:

- Quality-aware expert routing conditioned on measured IQMs
- Calibrated per-finding confidence
- A gate with a threshold chosen on held-out data
- Selective-prediction evaluation (risk–coverage, AURC, E-AURC)
- An ablation grid that isolates routing from ensembling

The prediction target is a **case-level structured finding vector** derived from
the segmentation mask — see `cgate/schema.py`. Every finding is derivable from
BraTS labels alone, with no extra annotation.

## What this preserves for Project-II

The novelty claim — *gate before generation rather than verify after* — is
unaffected. Project-I builds and validates the gate. Project-II adds the
generator behind it, conditioned on the structured findings and the gate
decision this project produces.

That ordering is also the defensible one: a verification-free generator is not
publishable, but a gate validated independently of any generator is a complete
result on its own.

## Consequence for related work

MAIRA-2 and RadFact remain **context, not baselines**. They are chest X-ray
work benchmarked on MIMIC-CXR. No number in this project can be compared
against them directly, and the write-up must say so rather than implying a
comparison.

## Scope guard

Two things are deliberately excluded from Project-I and should be pushed back on
if they reappear during the term:

- **Any text generation**, including template rendering. It adds surface area
  without testing the hypothesis.
- **A full 3-D ViT.** On 240×240×155×4 volumes it will not fit the available
  compute. Training is 2-D slice-based or patch-based 3-D by design — decided up
  front rather than discovered in week 10.

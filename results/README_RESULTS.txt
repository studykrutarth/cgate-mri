cgate-mri — Full BraTS2021 run + grounded VLM report generation
================================================================

Run completed: 17663.8 s (~4.9 h: ~2.9 h ingest of 1251 cases + grid training)
Hardware: NVIDIA GeForce RTX 4070 (12 GB), torch 2.14.0+cu130
Seed 0, protocol stratified, 750/250/250 train/calib/test splits

RUN CONFIGURATION
-----------------
  data            brats, 1251 cases (BraTS2021_Training_Data.tar)
  experts         torch (ResNet-50 CNN + ViT-S/16 + radiomics), --pretrained
  epochs          12   batch-size 32   arch resnet50
  grid            B0-B6 (single expert -> quality-routed fusion)
  label cuts      et_present >= 500 voxels (fixed)
                  ncr_present >= 7530 voxels (fitted on train split, F13)
  outputs         runs/brats_full_seed0.json   runs/figs/*.png
                  runs/predictions.json  (NEW: per-case export for cgate-report)

HEADLINE RESULTS (test n=250)
------------------------------
  B6  quality-routed fusion     acc 0.660  AURC 0.2167  E-AURC 0.1503
                                gate: coverage 85.2%, error 34.0% -> 30.0%
  B4  uniform fusion            acc 0.620  AURC 0.2289  E-AURC 0.1445
                                gate: coverage 80.4%, error 38.0% -> 32.8%

  Paired bootstrap B6 vs B4:  dAURC = -0.0122  95% CI [-0.0464, +0.0230]
                              p = 0.493  -> favours B6, NOT significant
  The CI crossing zero is reported as-is: on this data routing did not
  measurably beat uniform fusion. That is a result, not a failure.

  Reproducibility: seed 0 re-run reproduced the previous run's numbers
  exactly (acc 0.660, AURC 0.2167, dAURC -0.0122, p 0.493).

GROUNDED REPORT GENERATION (Project-II)
---------------------------------------
  New package: src/cgate/reporting/  (grounding, template, vlm, render, cli)
  New CLI:     cgate-report

  Design (per project scope: BraTS ships NO free-text reports, so nothing
  can be trained or scored against reference text):
    * every sentence carries evidence - a calibrated finding probability,
      an image-quality metric, or the gate decision - and appears in a
      per-sentence GROUNDING LEDGER beside the report
    * the gate decides release: deferred cases get a deferral notice naming
      the weakest critical finding, not a released report
    * VLM phrases (Qwen2-VL-2B-Instruct over a 4-sequence T1/T1ce/T2/FLAIR
      mosaic) are ADVISORY ONLY: they never enter confidence or the gate,
      and a veto filter drops any phrase contradicting a firm finding
      (e.g. "no enhancement" is vetoed when et_present=present, p>=0.7;
      vetoed phrases are listed with the reason, never silently dropped)

  Artifacts:
    runs/reports/        250 test cases, deterministic template
                         (213 released / 37 deferred), each with .md +
                         .json ledger + index.md
    runs/reports_vlm/    sample with VLM appearance phrases added
    Vetted-out phrases are shown per report under "Vetted-out VLM phrases"

  Validation: 185/185 tests pass (23 new for reporting). During live
  validation the veto filter caught a real model contradiction
  ("no apparent enhancement" vs et_present p=0.97) and dropped it.

  Model note: google/medgemma-4b-it is a GATED HuggingFace repo (401
  without login), so the active VLM is Qwen/Qwen2-VL-2B-Instruct
  (open). MedGemma remains the code default when access is available.

  Evaluation caveat (state this in any write-up): with no reference
  reports in BraTS, report quality is evaluated structurally -
  grounding completeness, deferral correctness, contradiction vetoes -
  NOT by text-similarity metrics.

REPRODUCE
---------
  # training run (writes results + predictions):
  .venv\Scripts\python.exe -m cgate.experiments.run --data brats --root "data" ^
      --experts torch --pretrained --arch resnet50 --epochs 12 --batch-size 32 ^
      --table-cache "runs\table" --slice-cache "runs\cache" ^
      --out "runs\brats_full_seed0.json" --figures "runs\figs" ^
      --predictions "runs\predictions.json" --report-system B6 --seed 0

  # template reports (offline, no VLM):
  .venv\Scripts\python.exe -m cgate.reporting.cli --preds "runs\predictions.json" ^
      --out "runs\reports"

  # with VLM phrases (needs: pip install -e ".[vlm]"):
  .venv\Scripts\python.exe -m cgate.reporting.cli --preds "runs\predictions.json" ^
      --out "runs\reports_vlm" --vlm --slice-cache "runs\cache"

  # live monitor during a run:
  powershell -File runs\monitor.ps1

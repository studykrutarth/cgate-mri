"""Command-line entry point: `cgate --help`.

Runs the full ablation grid on either synthetic phantoms (default, no download
required) or a real BraTS-style directory tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..data.splits import (
    external_cohort_split,
    split_summary,
    stratified_patient_split,
)
from ..data.synthetic import SyntheticConfig, build_synthetic_dataset
from ..findings import FindingExtractor, label_prevalence
from ..pipeline import CaseTable, ingest_cases
from ..routing import RouterConfig
from ..schema import FINDING_NAMES
from ..selective import paired_bootstrap_aurc_delta
from .ablations import ABLATIONS, run_ablations
from .evaluate import format_strata, format_table

SYNTHETIC_BANNER = """
+---------------------------------------------------------------------------+
|  SYNTHETIC PHANTOM DATA -- NOT A SCIENTIFIC RESULT                        |
|                                                                           |
|  These numbers verify that the pipeline runs and that the metrics respond  |
|  in the expected direction. They are NOT evidence that quality-aware       |
|  routing helps on real gliomas. For that, train on BraTS 2021 and evaluate |
|  the low-quality stratum on BraTS-Africa:  cgate --data brats --root ...   |
+---------------------------------------------------------------------------+
"""


def _to_jsonable(obj):
    """Drop the private arrays and make values JSON-serialisable.

    Path is handled here rather than by stringifying individual arguments at
    the call site: every new `type=Path` CLI flag would otherwise crash the
    JSON dump at the very end of a run, after all the expensive work is done.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cgate",
        description="Quality-aware expert routing with calibrated confidence gating.",
    )
    p.add_argument(
        "--data",
        choices=("synthetic", "brats"),
        default="synthetic",
        help="synthetic phantoms (default) or a real BraTS directory tree",
    )
    p.add_argument(
        "--root",
        type=Path,
        nargs="+",
        help="one or more dataset roots, required for --data brats. Several are "
        "allowed because BraTS 2021 and BraTS-Africa are separate downloads and "
        "often sit on different drives. Case directories are found at any depth.",
    )
    p.add_argument(
        "--cohort",
        nargs="+",
        help="explicit cohort per --root (e.g. high low). Omit to infer from "
        "directory and case names. Pass it when a root is named in a way the "
        "inference cannot read -- a mislabelled cohort silently destroys the "
        "external protocol rather than failing.",
    )
    p.add_argument("--n-cases", type=int, default=240, help="synthetic cohort size")
    p.add_argument(
        "--low-quality-fraction",
        type=float,
        default=0.35,
        help="share of the synthetic cohort drawn from the degraded cohort",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--protocol",
        choices=("stratified", "external"),
        default="stratified",
        help="stratified: random patient-level split, both cohorts in every "
        "split. external: hold out an ENTIRE cohort as test (train+calibrate on "
        "the rest). Use 'external' for the real BraTS 2021 -> BraTS-Africa "
        "experiment; it gives ~5x the test cases in the degraded cohort, which "
        "a 20%% split cannot supply.",
    )
    p.add_argument(
        "--test-cohort",
        default="low",
        help="cohort held out as test when --protocol external",
    )
    p.add_argument(
        "--calib-target-fraction",
        type=float,
        default=0.0,
        help="with --protocol external, share of the target cohort lent to "
        "calibration instead of test (e.g. 0.15). With the default of 0, the "
        "router is fitted only on the source cohort and must extrapolate into "
        "quality it never saw -- measured to make routing worse than uniform "
        "fusion under shift. Mirrors obtaining a small labelled sample from a "
        "new site.",
    )
    p.add_argument(
        "--target-coverage",
        type=float,
        default=0.8,
        help="gate operating point: fraction of cases released without review",
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(48, 48, 32),
        metavar=("H", "W", "D"),
        help="synthetic volume shape",
    )
    p.add_argument(
        "--min-voxels",
        type=int,
        default=None,
        help="voxels before a finding counts as present. Defaults to 500 "
        "(0.5 mL at 1 mm iso) for real BraTS, and 50 for synthetic phantoms, "
        "which are generated at a much smaller volume size.",
    )
    p.add_argument("--out", type=Path, help="write full results to this JSON file")
    p.add_argument(
        "--figures",
        type=Path,
        help="directory for risk-coverage and reliability figures (needs matplotlib)",
    )
    p.add_argument(
        "--experts",
        choices=("surrogate", "torch"),
        default="surrogate",
        help="surrogate: scikit-learn MLPs on hand-built feature views (no "
        "torch needed). torch: real ResNet-50 / ViT-S/16 on 2-D slices, with "
        "genuine MC dropout for B2. Only 'torch' constitutes evidence about "
        "CNN/ViT behaviour.",
    )
    p.add_argument(
        "--slice-cache",
        type=Path,
        default=None,
        help="directory for the slice cache (default: <out dir>/cache). Built "
        "during ingest when --experts torch. Also overrides the cache location "
        "recorded in a --table-cache, which is how a table ingested on one "
        "machine is trained on another.",
    )
    p.add_argument(
        "--table-cache",
        type=Path,
        default=None,
        help="directory holding the ingested case table. Reused if it exists, "
        "written after ingest if it does not. Ingest re-reads every volume in "
        "the study and costs far more than fitting the grid, so without this "
        "every seed and every router setting pays for it again.",
    )
    p.add_argument(
        "--rebuild-table",
        action="store_true",
        help="re-ingest from the raw data even when --table-cache exists",
    )
    p.add_argument("--epochs", type=int, default=8, help="epochs per torch expert")
    p.add_argument(
        "--n-slices",
        type=int,
        default=16,
        help="slices cached per case for the torch experts. Selection is "
        "label-free (never reads the mask); see cgate.data.slices.",
    )
    p.add_argument(
        "--arch", default="resnet18", help="timm backbone for the CNN expert"
    )
    p.add_argument(
        "--img-size", type=int, default=128, help="input size fed to the CNN backbone"
    )
    p.add_argument(
        "--vit-arch",
        default="vit_small_patch16_224",
        help="timm backbone for the ViT expert. Separate from --arch on purpose: "
        "one shared flag made --arch resnet50 rebuild the ViT as a second ResNet, "
        "collapsing the heterogeneity that B4-B6 exist to measure.",
    )
    p.add_argument(
        "--vit-img-size",
        type=int,
        default=224,
        help="input size fed to the ViT backbone (its position embeddings are "
        "interpolated to match)",
    )
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="use ImageNet-pretrained weights (downloads on first use). Strongly "
        "recommended for real data: ~1200 cases is far too few to train a ViT "
        "from scratch.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="ingest at most this many cases per --root (default: all). A "
        "seeded random subset, not the first N -- BraTS ids are grouped by "
        "contributing institution. Use it to validate the pipeline on a few "
        "hundred cases before committing to the full multi-hour ingest.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="seed for --max-cases, so a subset is reproducible from the config",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="slices per training batch. The main lever when a GPU runs out of "
        "memory: the stock resnet50/224px/32 configuration needs roughly 5 GB "
        "of activations, which a 4 GB card cannot hold.",
    )
    p.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="mixed precision. Default: on for CUDA, off for CPU. Roughly halves "
        "activation memory. It changes the numerics, so it is recorded in the "
        "run log -- an AMP row and a non-AMP row are different experiments.",
    )
    p.add_argument(
        "--no-amp", dest="amp", action="store_false", help="force full precision"
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda when available, else cpu)",
    )
    p.add_argument(
        "--router-shared",
        action="store_true",
        help="share router weights across findings instead of fitting one set "
        "per finding. Cuts parameters 5x (120 -> 24). The calibration split is "
        "small by construction, so per-finding weights are easily "
        "overparameterised; the run log reports the ratio.",
    )
    p.add_argument(
        "--router-l2",
        type=float,
        default=None,
        help="L2 penalty on router weights (default 0.01). Raise it when the "
        "run log reports params_per_sample > 1.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="paired bootstrap resamples for the headline comparison (0 to skip)",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _slice_cache_root(args) -> Path:
    return args.slice_cache or ((args.out.parent if args.out else Path("runs")) / "cache")


# Flags that ingest consumed and a reuse cannot honour. The saved provenance
# records what was actually built, so a mismatch is detectable -- and worth
# saying out loud, because otherwise `--n-cases 500` on a cached 40-case table
# looks exactly like a 500-case run in everything but the results.
_REUSE_IGNORED = {"n_cases": "--n-cases", "seed": "--seed", "roots": "--root"}


def _warn_stale_reuse(args, table, verbose: bool) -> None:
    if not verbose or not table.source:
        return
    current = {
        "n_cases": args.n_cases,
        "seed": args.seed,
        "roots": [str(r) for r in args.root] if args.root else None,
    }
    stale = [
        f"{flag} (table was built with {table.source[key]!r}, "
        f"this run asked for {current[key]!r})"
        for key, flag in _REUSE_IGNORED.items()
        if key in table.source and current[key] is not None and table.source[key] != current[key]
    ]
    if stale:
        print("  ignored, the table is reused as built:")
        for line in stale:
            print(f"    {line}")


def _guard_rebuild(args) -> None:
    """Refuse to overwrite a real-data table with phantoms.

    --rebuild-table re-enters the ingest branch keyed on --data, which defaults
    to synthetic. Adding the flag to a reuse-style command line -- which
    deliberately needs neither --data nor --root -- would otherwise replace a
    multi-hour BraTS ingest with 30 synthetic cases and report success.
    """
    if not (args.rebuild_table and args.table_cache):
        return
    meta_path = Path(args.table_cache) / "meta.json"
    if not meta_path.exists():
        return
    try:
        saved = json.loads(meta_path.read_text(encoding="utf8")).get("source", {})
    except (OSError, ValueError):
        return
    if saved.get("data") == "brats" and args.data != "brats":
        raise SystemExit(
            f"--rebuild-table would overwrite the table at {args.table_cache}, which "
            f"was ingested from real BraTS data ({saved.get('roots')}), with "
            f"--data {args.data}. Re-ingest with --data brats --root <paths>, or "
            f"point --table-cache somewhere else."
        )


def load_table(args, verbose: bool = True) -> CaseTable:
    cache_root = None
    slice_config = None
    if args.experts == "torch":
        from ..data.slices import SliceCacheConfig

        cache_root = _slice_cache_root(args)
        slice_config = SliceCacheConfig(n_slices=args.n_slices)

    # A saved table skips the entire ingest pass. Reuse comes first so the
    # expensive branch below is never entered by accident.
    if args.table_cache is not None and not args.rebuild_table:
        if (Path(args.table_cache) / "meta.json").exists():
            table = CaseTable.load(
                args.table_cache,
                slice_cache_root=args.slice_cache,
                require_slices=args.experts == "torch",
            )
            if verbose:
                print(f"Reusing case table from {args.table_cache} ({table.n_cases} cases)")
            if args.experts == "torch" and table.slice_cache is None:
                raise SystemExit(
                    f"the table at {args.table_cache} has no slice cache, so the "
                    f"CNN/ViT experts have no pixels to read. Point --slice-cache at "
                    f"one, or re-ingest with --rebuild-table --experts torch."
                )
            # The cache fixes the slice count; --n-slices cannot change it here.
            # Left unchecked, a sweep over --n-slices trains every arm on the
            # cached count and files byte-identical results under different
            # labels, because the run log records the flag and not the cache.
            if table.slice_cache is not None and table.slice_cache.n_slices != args.n_slices:
                raise SystemExit(
                    f"--n-slices {args.n_slices} does not match the reused slice "
                    f"cache, which holds {table.slice_cache.n_slices} slices per case. "
                    f"The cache decides, so the run would be recorded under a slice "
                    f"count it did not use. Pass --n-slices "
                    f"{table.slice_cache.n_slices}, or re-ingest with --rebuild-table."
                )
            _warn_stale_reuse(args, table, verbose)
            return table

    _guard_rebuild(args)

    if args.data == "synthetic":
        cfg = SyntheticConfig(
            shape=tuple(args.shape),
            n_cases=args.n_cases,
            low_quality_fraction=args.low_quality_fraction,
            seed=args.seed,
        )
        table = ingest_cases(
            build_synthetic_dataset(cfg),
            slice_cache_root=cache_root,
            slice_config=slice_config,
        )
        source = {"data": "synthetic", "n_cases": args.n_cases, "seed": args.seed}
        table.source = source
    else:
        if not args.root:
            raise SystemExit("--root is required when --data brats is used")
        if args.cohort and len(args.cohort) != len(args.root):
            raise SystemExit(
                f"--cohort takes one label per --root: got {len(args.cohort)} "
                f"labels for {len(args.root)} roots"
            )
        from ..data.brats import load_brats_roots

        table = ingest_cases(
            load_brats_roots(
                args.root,
                cohorts=args.cohort,
                max_cases=args.max_cases,
                sample_seed=args.sample_seed,
            ),
            slice_cache_root=cache_root,
            slice_config=slice_config,
        )
        source = {
            "data": "brats",
            "roots": [str(r) for r in args.root],
            "cohort": list(args.cohort) if args.cohort else None,
            # Recorded so a reused table cannot be mistaken for a full ingest.
            "max_cases": args.max_cases,
            "sample_seed": args.sample_seed,
        }
        table.source = source

    if args.table_cache is not None:
        table.save(args.table_cache, source=source)
        if verbose:
            print(f"Saved case table to {args.table_cache}")
    return table


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    t0 = time.time()

    if verbose:
        # Not args.data: a reused table needs no --data, so the flag sits at its
        # synthetic default while real BraTS is being loaded. F6 keyed the
        # banner, min_voxels and the payload off the table's provenance and
        # missed this line, which then announced a real run as synthetic.
        print("Loading cases ...")
    table = load_table(args, verbose=verbose)

    # Read from the table, not from --data: a reused BraTS table would otherwise
    # be announced and recorded as synthetic whenever --data was left at its
    # default, which is precisely the sort of invisible mislabelling the run log
    # exists to prevent.
    is_synthetic = table.source.get("data", args.data) == "synthetic"
    if is_synthetic and verbose:
        print(SYNTHETIC_BANNER)
    if verbose:
        print(f"  {table.n_cases} cases, "
              f"IQM features: {list(table.iqm_names)}")
        for name, mat in table.views.items():
            print(f"  view {name:<8} dim={mat.shape[1]}")

    # Printed before anything is fitted because a mislabelled cohort is not
    # recoverable downstream: --protocol external picks its test set by this
    # column, so getting it wrong quietly swaps source and target rather than
    # producing an error.
    values, counts = np.unique(table.cohorts, return_counts=True)
    cohort_counts = {str(v): int(c) for v, c in zip(values, counts)}
    print(f"\nCohorts: {cohort_counts}")
    if args.protocol == "external" and cohort_counts.get(args.test_cohort, 0) == 0:
        raise SystemExit(
            f"--protocol external needs cases with cohort {args.test_cohort!r}, "
            f"found {cohort_counts}. Cohort is read from the case and directory "
            f"names; pass --cohort explicitly (one label per --root) if the "
            f"layout does not carry it."
        )

    if args.protocol == "external":
        splits = external_cohort_split(
            table.patient_ids,
            table.cohorts,
            test_cohort=args.test_cohort,
            seed=args.seed,
            calib_target_fraction=args.calib_target_fraction,
        )
    else:
        # Patient-level, cohort-stratified. Stratifying on cohort keeps the
        # low-quality stratum populated in every split.
        splits = stratified_patient_split(
            table.patient_ids,
            table.cohorts,
            {"train": 0.6, "calib": 0.2, "test": 0.2},
            seed=args.seed,
        )
    if verbose:
        print(f"\nSplits (patient-level, protocol={args.protocol}):")
        for name, info in split_summary(splits, table.cohorts).items():
            print(f"  {name:<6} {info}")

    # Tercile edges and the ncr_present cut are fitted on train only.
    min_voxels = args.min_voxels
    if min_voxels is None:
        # Also keyed off the table, not --data: phantoms are generated at a much
        # smaller volume size, so getting this from the wrong source would move
        # every present/absent label in the study.
        min_voxels = 50 if is_synthetic else 500
    extractor = FindingExtractor(min_voxels=min_voxels).fit_from_measurements(
        [table.measurements[i] for i in splits["train"]]
    )
    table.assign_labels(extractor)

    if verbose:
        print()
        print(
            f"Thresholds: et_present >= {min_voxels:,} voxels (fixed), "
            f"ncr_present >= {extractor.ncr_cut:,.0f} voxels (fitted on train)"
        )
        print("\nLabel prevalence (all cases):")
        for finding, dist in label_prevalence(table.labels).items():
            pretty = "  ".join(f"{k}={v:.2f}" for k, v in dist.items())
            print(f"  {finding:<13} {pretty}")

    torch_config = None
    if args.experts == "torch":
        from ..experts.torch_experts import TorchExpertConfig

        torch_config = TorchExpertConfig(
            arch=args.arch,
            img_size=args.img_size,
            vit_arch=args.vit_arch,
            vit_img_size=args.vit_img_size,
            pretrained=args.pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            amp=args.amp,
            device=args.device,
            seed=args.seed,
        )
        if verbose:
            import torch as _t

            dev = args.device or ("cuda" if _t.cuda.is_available() else "cpu")
            amp_on = args.amp if args.amp is not None else dev.startswith("cuda")
            print(
                f"\nExperts: REAL torch backbones on {dev} "
                f"(pretrained={args.pretrained}, {args.epochs} epochs)"
            )
            # Both architectures are printed because they must differ for B4-B6
            # to mean anything, and a silent collapse to one backbone is exactly
            # the failure this flag pair was introduced to prevent.
            print(f"  CNN: {args.arch} @ {args.img_size}px")
            print(f"  ViT: {args.vit_arch} @ {args.vit_img_size}px")
            # AMP changes the numerics, so it belongs beside the architectures
            # rather than buried in the JSON: two rows that differ on it are
            # not comparable.
            print(f"  batch={args.batch_size}  amp={amp_on}")
            if dev.startswith("cuda") and not _t.cuda.is_available():
                raise SystemExit(
                    f"--device {dev} was asked for, but this torch build "
                    f"({_t.__version__}) has no CUDA support. Install a CUDA "
                    f"build, or drop --device to run on the CPU."
                )
            if dev.startswith("cuda"):
                vram = _t.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {_t.cuda.get_device_name(0)} ({vram:.1f} GB)")
                if vram < 6:
                    print(
                        "  Note: under ~6 GB. If training dies out of memory, "
                        "lower --batch-size first; it costs the result least."
                    )
            elif not _t.cuda.is_available():
                print(
                    f"  Note: no CUDA in this torch build ({_t.__version__}). "
                    "Fine for a smoke run; use a CUDA build or Colab to train."
                )
    elif verbose:
        print(
            "\nExperts: scikit-learn SURROGATES (not CNN/ViT). "
            "Pass --experts torch for real backbones."
        )

    if verbose:
        print("\nRunning ablations:")
    results = run_ablations(
        table,
        splits,
        specs=ABLATIONS,
        seed=args.seed,
        target_coverage=args.target_coverage,
        router=RouterConfig(
            seed=args.seed,
            per_finding=not args.router_shared,
            l2=args.router_l2 if args.router_l2 is not None else RouterConfig.l2,
        ),
        verbose=verbose,
        use_torch=args.experts == "torch",
        torch_config=torch_config,
    )

    cap = results.get("B6", {}).get("fit_log", {}).get("router_capacity")
    if cap:
        note = "  <-- more parameters than cases" if cap["overparameterised"] else ""
        print(
            f"\nRouter capacity: {cap['n_params']} parameters fitted on "
            f"{cap['n_samples']} cases "
            f"({cap['params_per_sample']} per case){note}"
        )
        if cap["overparameterised"]:
            print(
                "  Try --router-shared (5x fewer parameters) or a larger "
                "--router-l2 before concluding routing does not help."
            )

    print("\n" + format_table(results, args.target_coverage))
    print(format_strata(results, args.target_coverage))

    print("\nGate operating points (threshold chosen on calibration, applied to test):")
    for key, res in results.items():
        print(f"  {key}: {res['gate']['summary']}")

    # ---- headline comparison -------------------------------------------
    comparison = {}
    if args.bootstrap > 0 and "B6" in results and "B4" in results:
        print(
            f"\nHeadline comparison  B6 (quality-routed) vs B4 (uniform fusion), "
            f"paired bootstrap n={args.bootstrap}:"
        )
        for scope, mask in _comparison_scopes(table, results["B6"]):
            a, b = results["B6"], results["B4"]
            if int(mask.sum()) < 20:
                print(f"  {scope:<12} skipped (only {int(mask.sum())} cases)")
                continue
            stats = paired_bootstrap_aurc_delta(
                a["_confidence"][mask],
                b["_confidence"][mask],
                a["_correct"][mask],
                b["_correct"][mask],
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            comparison[scope] = stats
            verdict = (
                "B6 better" if stats["ci_high"] < 0
                else "B4 better" if stats["ci_low"] > 0
                else "no significant difference"
            )
            print(
                f"  {scope:<12} dAURC={stats['delta']:+.4f} "
                f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                f"p={stats['p_two_sided']:.3f}  -> {verdict}"
            )
        print(
            "\n  Negative dAURC favours B6. A CI crossing zero means routing did not\n"
            "  measurably beat uniform fusion on this data -- which is a real result,\n"
            "  not a failure to report."
        )

    if args.figures:
        try:
            from .figures import save_all_figures

            written = save_all_figures(results, args.figures, table)
            print(f"\nFigures written to {args.figures}:")
            for path in written:
                print(f"  {path.name}")
        except ImportError as exc:
            # Figures are a convenience; losing them must not fail a run that
            # already produced valid numbers.
            print(f"\nSkipped figures: {exc}")

    if args.out:
        payload = {
            "config": _to_jsonable(vars(args)),
            "n_cases": table.n_cases,
            "splits": split_summary(splits, table.cohorts),
            "label_prevalence": label_prevalence(table.labels),
            "results": _to_jsonable(results),
            "comparison": _to_jsonable(comparison),
            "synthetic": is_synthetic,
            # Where the cases actually came from, which survives a reused table.
            "source": _to_jsonable(table.source),
            "min_voxels": min_voxels,
            # The cut point for `ncr_present` is fitted on the training split
            # (F13), so unlike min_voxels it is a property of the cohort rather
            # than of the command line, and two runs over different splits are
            # not labelled the same way.
            "ncr_min_voxels": extractor.ncr_cut,
            # Recorded because surrogate and torch runs are not interchangeable
            # as evidence, and the distinction is invisible in the numbers.
            "experts": args.experts,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf8")
        print(f"\nWrote {args.out}")

    if verbose:
        print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


def _comparison_scopes(table: CaseTable, res: dict):
    """Yield (name, boolean mask over the test set) for the comparison."""
    test_idx = res["_test_idx"]
    yield "overall", np.ones(len(test_idx), dtype=bool)
    cohorts = table.cohorts[test_idx]
    for cohort in np.unique(cohorts):
        yield f"cohort={cohort}", cohorts == cohort


if __name__ == "__main__":
    sys.exit(main())


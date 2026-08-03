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
    """Drop the private arrays and make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
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
    p.add_argument("--root", type=Path, help="dataset root, required for --data brats")
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
        "--bootstrap",
        type=int,
        default=2000,
        help="paired bootstrap resamples for the headline comparison (0 to skip)",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def load_table(args) -> CaseTable:
    if args.data == "synthetic":
        cfg = SyntheticConfig(
            shape=tuple(args.shape),
            n_cases=args.n_cases,
            low_quality_fraction=args.low_quality_fraction,
            seed=args.seed,
        )
        return ingest_cases(build_synthetic_dataset(cfg))

    if args.root is None:
        raise SystemExit("--root is required when --data brats is used")
    from ..data.brats import load_brats_cases

    return ingest_cases(load_brats_cases(args.root))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    t0 = time.time()

    if args.data == "synthetic" and verbose:
        print(SYNTHETIC_BANNER)

    if verbose:
        print(f"Ingesting cases ({args.data}) ...")
    table = load_table(args)
    if verbose:
        print(f"  {table.n_cases} cases, "
              f"IQM features: {list(table.iqm_names)}")
        for name, mat in table.views.items():
            print(f"  view {name:<8} dim={mat.shape[1]}")

    if args.protocol == "external":
        splits = external_cohort_split(
            table.patient_ids,
            table.cohorts,
            test_cohort=args.test_cohort,
            seed=args.seed,
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

    # Tercile edges are fitted on train only.
    min_voxels = args.min_voxels
    if min_voxels is None:
        min_voxels = 50 if args.data == "synthetic" else 500
    extractor = FindingExtractor(min_voxels=min_voxels).fit_from_measurements(
        [table.measurements[i] for i in splits["train"]]
    )
    table.assign_labels(extractor)

    if verbose:
        print("\nLabel prevalence (all cases):")
        for finding, dist in label_prevalence(table.labels).items():
            pretty = "  ".join(f"{k}={v:.2f}" for k, v in dist.items())
            print(f"  {finding:<13} {pretty}")

    if verbose:
        print("\nRunning ablations:")
    results = run_ablations(
        table,
        splits,
        specs=ABLATIONS,
        seed=args.seed,
        target_coverage=args.target_coverage,
        router=RouterConfig(seed=args.seed),
        verbose=verbose,
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
            "config": vars(args) | {"root": str(args.root), "out": str(args.out)},
            "n_cases": table.n_cases,
            "splits": split_summary(splits, table.cohorts),
            "label_prevalence": label_prevalence(table.labels),
            "results": _to_jsonable(results),
            "comparison": _to_jsonable(comparison),
            "synthetic": args.data == "synthetic",
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

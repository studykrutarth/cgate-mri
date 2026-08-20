"""Measure the class prevalence of each mask-derived finding, from masks alone.

F13 asked whether `et_present` and `ncr_present` are degenerate by construction
rather than merely hard. Answering that needs no modalities, no slice cache and
no ingested table -- only the segmentations -- which is why this is a separate
script rather than part of a run: it stays available when nothing else does.

    python tools/prevalence.py "D:/path/to/BraTS-Africa"

Prints the ET / NCR / WT volume distributions and the prevalence of `present`
as a function of `FindingExtractor.min_voxels`, so a threshold can be chosen
against the data instead of inherited from the phantoms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cgate.data.brats import canonicalise_labels
from cgate.findings import DEFAULT_MIN_VOXELS, region_volumes

THRESHOLDS = (1, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000)
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)


def measure_masks(root: Path, pattern: str) -> list[dict[str, object]]:
    segs = sorted(root.glob(pattern))
    if not segs:
        raise SystemExit(f"no segmentations matching {pattern!r} under {root}")
    print(f"{len(segs)} segmentations", flush=True)

    rows: list[dict[str, object]] = []
    for i, path in enumerate(segs, start=1):
        img = nib.as_closest_canonical(nib.load(str(path)))
        mask = canonicalise_labels(np.asarray(img.dataobj), context=path.parent.name)
        rows.append({"id": path.parent.name, **region_volumes(mask)})
        if i % 25 == 0 or i == len(segs):
            print(f"  {i}/{len(segs)}", flush=True)
    return rows


def report(rows: list[dict[str, object]]) -> None:
    cols = {name: np.array([r[name] for r in rows], dtype=float) for name in ("ET", "NCR", "WT")}

    for name, v in cols.items():
        q = np.percentile(v, PERCENTILES)
        edges = "/".join(str(p) for p in PERCENTILES)
        print(f"\n{name} voxels  {edges}:")
        print("  " + "  ".join(f"{x:,.0f}" for x in q))
        print(f"  zero in {int((v == 0).sum())}/{len(v)} cases  median {np.median(v):,.0f}")

    et, ncr = cols["ET"], cols["NCR"]
    print(f"\nprevalence of `present` by min_voxels (current default {DEFAULT_MIN_VOXELS})")
    print(f"{'min_voxels':>10}  {'et_present':>10}  {'ncr_present':>11}")
    for thr in THRESHOLDS:
        mark = "  <-- current" if thr == DEFAULT_MIN_VOXELS else ""
        print(f"{thr:>10}  {(et >= thr).mean():>10.3f}  {(ncr >= thr).mean():>11.3f}{mark}")

    print(
        "\nA finding whose prevalence stays near 1.0 across every threshold is "
        "degenerate by\nconstruction: no choice of min_voxels turns it into a "
        "question worth asking."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory holding the case folders")
    ap.add_argument(
        "--pattern",
        default="*/*/*-seg.nii.gz",
        help="glob for the segmentations, relative to root "
        "(BraTS-Africa nests cases under a cohort folder; 2021 does not, so "
        "use '*/*-seg.nii.gz' or '*/*_seg.nii.gz' there)",
    )
    ap.add_argument("--out", type=Path, help="also write the per-case volumes as JSON")
    args = ap.parse_args()

    rows = measure_masks(args.root, args.pattern)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf8")
        print(f"wrote {args.out}")
    report(rows)


if __name__ == "__main__":
    main()

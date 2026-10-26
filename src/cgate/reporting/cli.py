"""`cgate-report`: render grounded reports from an exported prediction file.

Consumes the JSON written by `cgate run --predictions`, i.e. what the fitted
system actually predicted on the test split, then renders one markdown report
per case with its grounding-ledger sidecar. With `--vlm`, each case's
4-sequence mosaic also goes through the VLM, and the veto filter runs before
any phrase is allowed into the report.

Kept separate from `cgate run` on purpose: VLM inference is slow relative to
training and needs a GPU the training process may still hold, so re-rendering
reports must not require re-running the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .grounding import ReportCase, filter_phrases
from .render import render_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cgate-report",
        description="Render grounded radiology-style reports from exported predictions.",
    )
    p.add_argument("--preds", type=Path, required=True,
                   help="predictions JSON written by `cgate run --predictions`")
    p.add_argument("--out", type=Path, default=Path("runs/reports"),
                   help="output directory for .md reports, .json ledgers, index.md")
    p.add_argument("--limit", type=int, default=0,
                   help="render at most N cases (0 = all)")
    p.add_argument("--released-only", action="store_true",
                   help="skip deferred cases (default renders both; deferral notices are reports too)")
    p.add_argument("--vlm", action="store_true",
                   help="add VLM appearance phrases (needs --slice-cache and the vlm extra installed)")
    p.add_argument("--model", default=None, help="override the VLM model id")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--slice-cache", type=Path, required=False,
                   help="slice cache root, required with --vlm")
    p.add_argument("--slice-index", type=int, default=None,
                   help="slice index for the mosaic (default: middle)")
    p.add_argument("--no-ledger", action="store_true",
                   help="omit the per-sentence ledger table from the markdown")
    return p


def _load_cases(payload: dict) -> list[ReportCase]:
    q_names = tuple(payload.get("q_names", ()))
    cases: list[ReportCase] = []
    for c in payload["cases"]:
        cases.append(
            ReportCase(
                patient_id=str(c["patient_id"]),
                cohort=str(c["cohort"]),
                probs=[np.asarray(p, dtype=float) for p in c["probs"]],
                confidence=float(c["confidence"]),
                gate_threshold=float(payload["gate_threshold"]),
                accepted=bool(c["accepted"]),
                iqms={k: float(v) for k, v in (c.get("iqms") or {}).items()},
                q_std=tuple(float(z) for z in (c.get("q_std") or ())),
                q_names=q_names,
                labels=np.asarray(c["labels"]) if c.get("labels") is not None else None,
                correct=c.get("correct"),
            )
        )
    return cases


def _findings_context(case: ReportCase) -> str:
    lines = []
    for name, (cls, p) in case.predicted.items():
        lines.append(f"- {name}: {cls} (p={p:.2f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    payload = json.loads(args.preds.read_text(encoding="utf8"))
    cases = _load_cases(payload)
    if args.released_only:
        cases = [c for c in cases if c.accepted]
    if args.limit > 0:
        cases = cases[: args.limit]

    vlm = None
    cache = None
    if args.vlm:
        if args.slice_cache is None:
            print("--vlm requires --slice-cache", file=sys.stderr)
            return 2
        from ..data.slices import SliceCache

        from .vlm import VLMConfig, VLMReporter

        cache = SliceCache.open(args.slice_cache)
        cfg = VLMConfig(device=args.device)
        if args.model:
            cfg.model_id, cfg.fallback_model_id = args.model, None
        cfg.slice_index = args.slice_index
        vlm = VLMReporter(config=cfg)
        if not vlm.ensure_loaded():
            print("Proceeding template-only.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    n_phrases = n_rejected = 0

    for i, case in enumerate(cases, start=1):
        phrases: list[str] = []
        rejected: list[tuple[str, str]] = []
        if vlm is not None:
            from .vlm import mosaic_for_patient

            try:
                image = mosaic_for_patient(cache, case.patient_id, args.slice_index)
            except (KeyError, FileNotFoundError):
                image = None  # not every case has cached slices; phrase-free is fine
            raw = vlm.describe(image, _findings_context(case)) if image is not None else []
            kept, vetoed = filter_phrases(raw, case)
            phrases = kept
            rejected = [(p, r.why or r.name) for p, r in vetoed]

        report = render_report(
            case,
            vlm_phrases=phrases,
            vlm_rejected=rejected,
            vlm_model=vlm.loaded_model_id if vlm is not None else None,
        )
        stem = args.out / case.patient_id
        stem.with_suffix(".md").write_text(
            report.markdown(include_ledger=not args.no_ledger), encoding="utf8"
        )
        stem.with_suffix(".json").write_text(report.to_json(), encoding="utf8")
        n_phrases += len(phrases)
        n_rejected += len(rejected)
        rows.append(
            {
                "patient_id": case.patient_id,
                "cohort": case.cohort,
                "confidence": f"{case.confidence:.3f}",
                "status": "released" if case.accepted else "deferred",
                "phrases": len(phrases),
                "vetoes": len(rejected),
            }
        )
        if i % 25 == 0 or i == len(cases):
            print(f"  rendered {i}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)

    # Index: one table an auditor can scan without opening 250 files.
    lines = [
        "# Reports index",
        "",
        f"- source predictions: `{args.preds}`",
        f"- system: {payload.get('system', '?')}",
        f"- gate threshold: {payload.get('gate_threshold', float('nan')):.3f}",
        f"- VLM: {vlm.loaded_model_id if vlm and vlm.loaded_model_id else 'not used'}",
        f"- cases: {len(rows)} "
        f"({sum(1 for r in rows if r['status'] == 'released')} released, "
        f"{sum(1 for r in rows if r['status'] == 'deferred')} deferred)",
        f"- VLM phrases kept: {n_phrases}; vetoed: {n_rejected}",
        "",
        "| patient | cohort | confidence | status | phrases | vetoes |",
        "|---------|--------|------------|--------|---------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| [{r['patient_id']}]({r['patient_id']}.md) | {r['cohort']} | "
            f"{r['confidence']} | {r['status']} | {r['phrases']} | {r['vetoes']} |"
        )
    (args.out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    released = sum(1 for r in rows if r["status"] == "released")
    print(
        f"\nWrote {len(rows)} reports to {args.out} "
        f"({released} released, {len(rows) - released} deferred; "
        f"{n_phrases} VLM phrases kept, {n_rejected} vetoed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

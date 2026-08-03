"""Figures for the report: risk-coverage curves and reliability diagrams.

These are the two plots that carry the argument:

  risk-coverage  -- shows the gate working. A curve that falls as coverage
                    drops means confidence is ranking errors correctly. A flat
                    curve means the gate is useless no matter how good ECE is.

  reliability    -- shows calibration. Points below the diagonal are
                    overconfident, which is the dangerous direction in a
                    clinical setting.

matplotlib is an optional dependency (`pip install -e '.[sci]'`), so it is
imported lazily and the runner degrades to text-only output if it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..selective import risk_coverage_curve

# Fixed per-system colours so a system keeps its identity across every figure.
SERIES_COLOURS: dict[str, str] = {
    "B0": "#9E9E9E",
    "B1": "#7E57C2",
    "B2": "#26A69A",
    "B3": "#EF6C00",
    "B4": "#1E88E5",
    "B5": "#8E24AA",
    "B6": "#C62828",
}
OURS = "B6"


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: never try to open a window
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for figures. Install with: "
            "pip install -e '.[sci]'"
        ) from exc


def _mask_for_cohort(table, res: dict, cohort: str | None) -> np.ndarray:
    idx = res["_test_idx"]
    if cohort is None:
        return np.ones(len(idx), dtype=bool)
    return np.asarray(table.cohorts)[idx] == cohort


def risk_coverage_figure(
    results: dict[str, dict],
    out_path: Path,
    table=None,
    cohort: str | None = None,
    title: str | None = None,
    min_cases: int = 15,
) -> Path | None:
    """One risk-coverage curve per system."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    plotted = 0

    for key, res in results.items():
        conf, correct = res.get("_confidence"), res.get("_correct")
        if conf is None or correct is None:
            continue
        if cohort is not None:
            if table is None:
                raise ValueError("a CaseTable is required to filter by cohort")
            m = _mask_for_cohort(table, res, cohort)
            if int(m.sum()) < min_cases:
                continue
            conf, correct = conf[m], correct[m]
        if conf.size == 0:
            continue

        curve = risk_coverage_curve(conf, correct)
        is_ours = key == OURS
        ax.plot(
            curve["coverage"],
            curve["risk"],
            label=f"{key} {res.get('spec', {}).get('label', '')}"[:44],
            color=SERIES_COLOURS.get(key, "#555555"),
            linewidth=2.4 if is_ours else 1.3,
            alpha=1.0 if is_ours else 0.75,
            zorder=3 if is_ours else 2,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("Coverage  (fraction of cases released)")
    ax.set_ylabel("Selective risk  (error rate among released)")
    ax.set_title(title or f"Risk-coverage{'' if cohort is None else f'  -  cohort={cohort}'}")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    # Lower-left is better: less error at the same coverage.
    ax.annotate(
        "better",
        xy=(0.06, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color="#666666",
        style="italic",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def reliability_figure(
    results: dict[str, dict],
    out_path: Path,
    n_bins: int = 10,
    keys: tuple[str, ...] | None = None,
) -> Path | None:
    """Confidence vs observed accuracy, with the ideal diagonal."""
    plt = _require_matplotlib()

    chosen = [k for k in (keys or results.keys()) if k in results]
    if not chosen:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=160)
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.0, label="perfect")

    plotted = 0
    for key in chosen:
        res = results[key]
        conf, correct = res.get("_confidence"), res.get("_correct")
        if conf is None or conf.size == 0:
            continue
        edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = (conf >= edges[b]) & (conf < edges[b + 1])
            if m.sum() < 3:  # too few cases to estimate an accuracy
                continue
            xs.append(float(conf[m].mean()))
            ys.append(float(correct[m].mean()))
        if not xs:
            continue
        is_ours = key == OURS
        ax.plot(
            xs, ys, marker="o", markersize=4,
            color=SERIES_COLOURS.get(key, "#555555"),
            linewidth=2.2 if is_ours else 1.2,
            alpha=1.0 if is_ours else 0.7,
            label=key,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability  (below the diagonal = overconfident)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_all_figures(
    results: dict[str, dict], outdir: Path, table=None
) -> list[Path]:
    """Write every figure; returns the paths actually produced."""
    outdir = Path(outdir)
    written: list[Path] = []

    p = risk_coverage_figure(results, outdir / "risk_coverage_overall.png", table)
    if p:
        written.append(p)

    if table is not None:
        for cohort in sorted({str(c) for c in np.unique(table.cohorts)}):
            p = risk_coverage_figure(
                results, outdir / f"risk_coverage_{cohort}.png", table, cohort=cohort
            )
            if p:
                written.append(p)

    p = reliability_figure(results, outdir / "reliability.png")
    if p:
        written.append(p)
    return written

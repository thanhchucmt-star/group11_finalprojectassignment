"""Distribution figure for the descriptive profile: histograms and percentile boxplots by outcome."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmda_profile_defs import NUMERIC

BLUE, ORANGE, GRAY = "#006DAA", "#B45116", "#505A65"
INK, GREEN = "#1A1A1A", "#009E73"  # high-contrast third series
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42,
})
PANELS = [
    ("loan_amount", "Loan amount\n(USD thousands)", 1e-3),
    ("income", "Applicant income\n(USD thousands)", 1),
    ("interest_rate", "Interest rate\n(percent)", 1),
    ("rate_spread", "Rate spread\n(percentage points)", 1),
]
OUTCOMES = [("1", "Originated", BLUE), ("2", "Approved, not accepted", ORANGE)]
SHORT = {"1": "Originated", "2": "Not accepted"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tables", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(args.tables / "profile_hist_spark.csv", dtype={"outcome": str})
    stats = pd.read_csv(args.tables / "profile_numeric_spark.csv", dtype={"outcome": str}).set_index(
        ["variable", "outcome"])

    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.4), layout="constrained",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for col, (var, label, scale) in enumerate(PANELS):
        ax, bx = axes[0, col], axes[1, col]
        part = hist[hist.variable.eq(var)]
        lo, hi, width = NUMERIC[var]
        nbins = int(round((hi - lo) / width))
        for code, name, color in OUTCOMES:
            h = part[part.outcome.eq(code)].set_index("bin")["count"]
            share = h / h.sum()
            inside = share[(share.index >= 0) & (share.index < nbins)]
            lo, _, width = NUMERIC[var]
            x = (lo + inside.index.to_numpy() * width) * scale
            ax.step(x, inside.to_numpy() / (width * scale), where="post", color=color, lw=1, label=name)
        ax.set_title(label, fontsize=8)
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Density")
        boxes = []
        for code, name, color in OUTCOMES:
            s = stats.loc[(var, code)]
            boxes.append({"med": s.p50 * scale, "q1": s.p25 * scale, "q3": s.p75 * scale,
                          "whislo": s.p05 * scale, "whishi": s.p95 * scale, "label": SHORT[code]})
        try:
            art = bx.bxp(boxes, orientation="horizontal", showfliers=False, widths=0.6, patch_artist=True)
        except TypeError:  # Matplotlib before 3.10
            art = bx.bxp(boxes, vert=False, showfliers=False, widths=0.6, patch_artist=True)
        for patch, (_, _, color) in zip(art["boxes"], OUTCOMES):
            patch.set(facecolor="white", edgecolor=color)
        for median, (_, _, color) in zip(art["medians"], OUTCOMES):
            median.set(color=color, lw=1.4)
        bx.set_xlim(ax.get_xlim())
        bx.tick_params(axis="y", labelsize=7)
        if col:
            bx.set_yticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=7)
    fig.savefig(args.out / "variable_distributions.pdf", dpi=220)
    fig.savefig(args.out / "variable_distributions.png", dpi=160)
    plt.close(fig)
    print("Saved variable_distributions", flush=True)
    quality_figure(args.tables, args.out)
    benchmark_figure(args.tables, args.out)


def quality_figure(tables, out):
    """(a) How concentrated missing prices are across lenders; (b) non-origination by case-mix band."""
    lend = pd.read_csv(tables / "lender_reporting_lenders.csv")
    rates = pd.read_csv(tables / "rates_by_band.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.5), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.35]})
    series = [((lend.n_2 - lend.n_price_valid_2), "Missing spreads, not accepted", ORANGE, "-"),
              (lend.n_2, "Approvals not accepted", INK, "--"),
              (lend.n_1, "Originations", BLUE, ":")]
    for values, label, color, style in series:
        v = np.sort(values.to_numpy(dtype=float))[::-1]
        share = np.cumsum(v) / v.sum()
        ax.plot(np.arange(1, len(v) + 1), share * 100, color=color, ls=style, lw=1.3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Lenders ranked by count (log scale)")
    ax.set_ylabel("Cumulative share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Concentration across lenders", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    order = {"dti_band": ["<36", "36-<43", "43-<50", "50+", "Unknown"],
             "cltv_band": ["<70", "70-<80", "80-<90", "90-<97", "97+", "Unknown"]}
    labels, values, colors = [], [], []
    for band, name in (("dti_band", "DTI"), ("cltv_band", "CLTV")):
        part = rates[rates.band.eq(band)].set_index("value").reindex(order[band])
        for value, row in part.iterrows():
            labels.append(f"{name} {value}")
            values.append(100 * row.non_origination_rate)
            colors.append(ORANGE if value == "Unknown" else BLUE)
        labels.append("")
        values.append(np.nan)
        colors.append("white")
    x = np.arange(len(values))
    bx.bar(x, values, color=colors, width=0.75)
    for xi, v in zip(x, values):
        if not np.isnan(v):
            bx.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.5)
    bx.set_xticks(x, labels, rotation=55, ha="right", fontsize=6)
    bx.set_ylabel("Non-origination rate (%)")
    bx.set_title("(b) Non-origination by case-mix band", fontsize=8)
    fig.savefig(out / "data_quality.pdf", dpi=220)
    fig.savefig(out / "data_quality.png", dpi=160)
    plt.close(fig)
    print("Saved data_quality", flush=True)


def benchmark_figure(tables, out):
    """Median time per condition with min-max range, against input rows."""
    path = tables / "benchmark_summary.csv"
    if not path.exists():
        return
    b = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(3.42, 2.0), layout="constrained")
    for (engine, mode), label, color, marker in [
            (("Pandas", "uncached"), "pandas", BLUE, "o"),
            (("Spark", "uncached"), "Spark, uncached", ORANGE, "s"),
            (("Spark", "cached"), "Spark, cached", GREEN, "^")]:
        part = b[(b.engine == engine) & (b["mode"] == mode)].sort_values("input_rows")
        x = part.input_rows / 1e6
        ax.errorbar(x, part.median_seconds, yerr=[part.median_seconds - part.min_seconds,
                                                  part.max_seconds - part.median_seconds],
                    color=color, marker=marker, ms=4, lw=1.1, capsize=2, label=label)
    ax.set_xlabel("Input records (millions)")
    ax.set_ylabel("Seconds (median, min-max)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(out / "benchmark_scaling.pdf", dpi=220)
    fig.savefig(out / "benchmark_scaling.png", dpi=160)
    plt.close(fig)
    print("Saved benchmark_scaling", flush=True)


if __name__ == "__main__":
    main()

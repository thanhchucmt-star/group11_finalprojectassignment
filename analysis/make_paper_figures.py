"""Generate the four manuscript figures from the archived HMDA tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BLUE, ORANGE, GRAY = "#006DAA", "#B45116", "#505A65"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42,
})

def save(fig, folder, name):
    fig.savefig(folder / f"{name}.pdf", dpi=220)
    plt.close(fig)

def cohort_figure(tables, out):
    flow = pd.read_csv(tables / "waterfall.csv")
    flow = flow[flow.step.ne("year")]
    labels = ["Public LAR", "Actions 1 or 2", "Home purchase", "First lien",
              "Site-built, 1–4 units", "Owner occupied", "Closed end", "Non-reverse"]
    cov = pd.read_csv(tables / "coverage.csv")
    cov = cov[cov.dimension.isin(["overall", "product"])].copy()
    cov["share"] = cov.n_price_valid / cov.n
    prices = cov.pivot(index="group", columns="action_taken", values="share").loc[["All", "1", "2", "3", "4"]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.45), layout="constrained")
    bars = ax.barh(labels, flow.n / 1e6, color=BLUE)
    ax.bar_label(bars, labels=[f"{n / 1e6:.2f}M" for n in flow.n], padding=3, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlim=(0, 15), xlabel="Records (millions)", title="(a) Cohort selection")
    pos = np.arange(len(prices))
    for action, delta, color, hatch, label in [
        (1, -0.18, BLUE, "", "Originated"), (2, 0.18, ORANGE, "//", "Approved, not accepted")
    ]:
        bars = bx.bar(pos + delta, prices[action], .36, color=color, hatch=hatch, label=label)
        bx.bar_label(bars, labels=[f"{p:.0%}" for p in prices[action]], padding=2, fontsize=6.5)
    bx.set_xticks(pos, ["All", "Conv.", "FHA", "VA", "USDA"])
    bx.set(ylim=(0, 1.25), yticks=[0, .25, .50, .75, 1], ylabel="Valid rate spread",
           title="(b) Price coverage by outcome")
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    bx.legend(loc="upper center", ncol=2, fontsize=6.3, frameon=False, handlelength=1.2)
    save(fig, out, "cohort_coverage")

def matched_figure(tables, out):
    segments = pd.read_csv(tables / "segments.csv", dtype={"market": str})
    segments = segments.loc[segments.geo.eq("county") & segments.market.notna()]
    wide = segments.pivot(index=["lei", "market", "loan_type"], columns="action_taken",
                          values=["n_price_valid", "median"])
    weight = wide["n_price_valid"].min(axis=1, skipna=False)
    use = weight.ge(10)
    gaps = (wide["median"][2] - wide["median"][1]).loc[use] * 100
    mean = np.average(gaps, weights=weight[use])
    summary = pd.read_csv(tables / "matched_price_summary.csv").query(
        "geo == 'county' and loan_type == 'All' and scenario == 'minimum_10_prices_each'"
    ).iloc[0]
    assert len(gaps) == summary.n_strata and np.isclose(mean / 100, summary.weighted_mean_of_median_gaps_pp)
    fig, ax = plt.subplots(figsize=(3.42, 2.1), layout="constrained")
    edges = np.arange(np.floor(gaps.min() / 10) * 10, np.ceil(gaps.max() / 10) * 10 + 1, 10)
    ax.hist(gaps, bins=edges, color=BLUE, alpha=.82)
    ax.axvline(0, color=GRAY, lw=1, ls=":")
    ax.axvline(mean, color=ORANGE, lw=1.2, ls="--", label=f"Weighted mean: {mean:+.1f} bp")
    ax.legend(frameon=False, fontsize=7)
    ax.set(xlabel="Difference in median rate spread (bp)", ylabel="County segments")
    save(fig, out, "matched_price_gaps")

def diagnostic_figure(tables, out):
    data = pd.read_csv(tables / "lender_market_diagnostics.csv")
    data = data[data.primary_eligible]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.1, 2.5), layout="constrained")
    for ax, geo, label in zip(axes, ["county", "msa"], ["County", "MSA or metro division"]):
        part = data[data.geo.eq(geo)]
        ax.axhline(0, color=GRAY, lw=.7)
        ax.axvline(0, color=GRAY, lw=.7)
        for flag, marker, color, legend in [(False, "o", BLUE, "Other segments"),
                                           (True, "^", ORANGE, "Both gaps positive")]:
            points = part[part.primary_candidate.eq(flag)]
            ax.scatter(points.relative_spread_pp * 100, points.adjusted_gap_pp, marker=marker,
                       s=8, alpha=.45, color=color, linewidths=0, label=legend)
        ax.set_title(f"{label} (n = {len(part):,})")
        ax.set_xlabel("Mean originated rate spread gap (bp)")
    axes[0].set_ylabel("Adjusted non-origination gap (pp)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)
    save(fig, out, "peer_diagnostic")

def model_figure(tables, out):
    model = pd.read_csv(tables / "price_model.csv")
    labels = {
        "lender_county_product": "Segment effects",
        "lender_county_product_on_case_mix_sample": "Segment effects,\ncase-mix sample",
        "lender_county_product_case_mix": "Add case mix",
        "balanced_price_coverage_case_mix": "Balanced coverage",
        "central_98_percent_price_case_mix": "Central 98% of spreads",
    }
    names = model.specification.map(labels).fillna(model.specification)
    counts = model.n_identifying_records
    beta = model.fallout_pp_per_spread_pp
    y = np.arange(len(model))
    fig, ax = plt.subplots(figsize=(3.42, 0.45 * len(model) + 0.45), layout="constrained")
    ax.axvline(0, color=GRAY, lw=.8, ls=":")
    ax.errorbar(beta, y, xerr=[beta-model.ci95_lower_pp, model.ci95_upper_pp-beta],
                fmt="o", color=BLUE, capsize=3, ms=4)
    ax.set_yticks(y, [f"{name}\n(n = {n:,})" for name, n in zip(names, counts)])
    ax.invert_yaxis()
    ax.set_xlabel("Change per 1 pp higher spread (pp)")
    save(fig, out, "model_sensitivity")

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tables", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for draw in (cohort_figure, matched_figure, diagnostic_figure, model_figure):
        draw(args.tables, args.out)

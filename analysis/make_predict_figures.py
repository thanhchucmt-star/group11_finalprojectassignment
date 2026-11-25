"""Figures for the predictive extension: price gradients from the within-segment model and
the decomposition of the within-segment gap with bootstrap intervals."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hmda_predict_defs as P

BLUE, ORANGE, INK, GREEN, PURPLE = "#006DAA", "#B45116", "#1A1A1A", "#009E73", "#6A3D9A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})
PANELS = [("cltv_bin", "Combined loan-to-value (%)"), ("dti_bin", "Debt-to-income (%)"),
          ("amount_bin", "Loan amount (USD 000)"), ("income_bin", "Income (USD 000)")]


def short(label, name):
    if name == "amount_bin":
        k = lambda v: f"{int(v) // 1000:,}"
        if label.startswith("<"):
            return f"<{k(label[1:])}"
        if label.endswith("+"):
            return f"{k(label[:-1])}+"
        lo, hi = label.split("-<")
        return f"{k(lo)}–{k(hi)}"
    return label.replace("-<", "–")


def gradients(coef, out):
    fig, grid = plt.subplots(2, 2, figsize=(3.42, 3.7), layout="constrained", sharey=True)
    axes = grid.ravel()
    coef = coef.set_index("column")
    for ax, (name, title) in zip(axes, PANELS):
        _, edges, ref = P.CORE[name]
        cats = P.labels(edges)
        y, lo, hi = [], [], []
        for c in cats:
            key = f"{name}={c}"
            if c == ref:
                y.append(0.0); lo.append(0.0); hi.append(0.0)
            else:
                r = coef.loc[key]
                y.append(100 * r["mean"]); lo.append(100 * r["min"]); hi.append(100 * r["max"])
        x = np.arange(len(cats))
        y, lo, hi = map(np.asarray, (y, lo, hi))
        ax.axhline(0, color=INK, lw=0.6)
        ax.plot(x, y, color=BLUE, lw=1.4, marker="o", ms=3)
        ax.vlines(x, lo, hi, color=BLUE, lw=2.5, alpha=0.5)
        ax.plot(x[cats.index(ref)], 0, marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.4)
        ax.set_xticks(x, [short(c, name) for c in cats], rotation=90, fontsize=6)
        ax.set_title(title, fontsize=7.5)
    for ax in grid[:, 0]:
        ax.set_ylabel("Spread vs reference (bp)", fontsize=7)
    fig.savefig(out / "predict_gradients.pdf"); fig.savefig(out / "predict_gradients.png", dpi=220)
    plt.close(fig)


def decomposition(matched, decision, out):
    rows = [("Main analysis\n(all records)", decision["original_gap_bp"], None, None, INK)]
    m = matched.set_index(["variant", "amount_band"])
    main = m.loc[("main", "All")]
    rows.append(("Raw gap,\ncomplete cases", main.raw_gap_bp, main.raw_lo_bp, main.raw_hi_bp, ORANGE))
    rows.append(("Beyond observables\n(linear model)", main.resid_gap_bp, main.resid_lo_bp, main.resid_hi_bp, BLUE))
    if ("gbt", "All") in m.index:
        g = m.loc[("gbt", "All")]
        rows.append(("Beyond observables\n(boosted trees)", g.resid_gap_bp, g.resid_lo_bp, g.resid_hi_bp, GREEN))
    sens = [("unknown_core", "Unknown kept\nas category"), ("central_98", "Central 98%\nof spreads"),
            ("support", "Thin cells\ndropped")]
    for v, label in sens:
        if (v, "All") in m.index:
            s = m.loc[(v, "All")]
            rows.append((label, s.resid_gap_bp, s.resid_lo_bp, s.resid_hi_bp, PURPLE))
    bands = matched[(matched.variant == "main_by_amount_band")].copy()
    # Below USD 150,000 fixed fees dominate APR-based spreads (15 segments); not interpreted, not drawn
    order = ["150000-<250000", "250000-<400000", "400000-<650000", "650000+"]
    bands = bands.set_index("amount_band").reindex([b for b in order if b in set(bands.amount_band)])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.25), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, lo, hi, color) in zip(y, rows):
        ax.barh(yi, v, color=color, height=0.62)
        if lo is not None:
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1, capsize=2.5)
        ax.text(max(v, hi or v) + 0.4, yi, f"{v:.1f}", va="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.8)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Within-segment gap, not accepted minus originated (bp)")
    ax.set_title("Gap before and after observable characteristics", fontsize=8)
    xb = np.arange(len(bands))
    w = 0.38
    bx.bar(xb - w / 2, bands.raw_gap_bp, w, color=ORANGE, label="Raw gap")
    bx.bar(xb + w / 2, bands.resid_gap_bp, w, color=BLUE, label="Beyond observables")
    bx.errorbar(xb + w / 2, bands.resid_gap_bp, yerr=[bands.resid_gap_bp - bands.resid_lo_bp,
                                                      bands.resid_hi_bp - bands.resid_gap_bp],
                fmt="none", ecolor=INK, lw=0.9, capsize=2)
    labels = ["150–250", "250–400", "400–650", "650+"]
    bx.set_xticks(xb, [labels[order.index(b)] for b in bands.index], fontsize=7)
    bx.axhline(0, color=INK, lw=0.6)
    bx.set_xlabel("Loan amount (USD 000)")
    bx.set_ylabel("Gap (bp)")
    bx.set_title("By loan size", fontsize=8)
    bx.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.savefig(out / "predict_gap.pdf"); fig.savefig(out / "predict_gap.png", dpi=220)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--summary", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import json
    gradients(pd.read_csv(args.summary / "predict_coef_main.csv"), args.out)
    decomposition(pd.read_csv(args.summary / "predict_matched.csv"),
                  json.loads((args.summary / "predict_decision.json").read_text()), args.out)


if __name__ == "__main__":
    main()

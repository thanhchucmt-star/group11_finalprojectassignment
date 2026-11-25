"""HMDA benchmark estimates from the Spark cohort, coverage, and case-mix tables."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = ["geo", "lei", "market", "loan_type"]
MARKET = ["geo", "market", "loan_type"]
BANDS = ["amount_band", "income_band", "dti_band", "cltv_band"]
PRODUCTS = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA / RHS"}
PRIMARY_VOLUME = 100
MIN_ORIGINATED_PRICES = 30
MIN_MATCHED_PRICES = 10
MIN_SUPPORT = 0.60
MIN_PRICE_COVERAGE = 0.50
MAX_PRICE_COVERAGE_GAP = 0.20


def grouped(frame, keys):
    return frame.groupby(keys, sort=False, observed=True, dropna=False)


def add_wilson(frame):
    """Descriptive binomial intervals; lender and market dependence is unmodeled."""
    n = frame["n"].where(frame["n"] > 0)
    p = frame["n_fallout"] / n
    z = 1.95996398454
    midpoint = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    frame["fallout_rate"] = p
    frame["wilson_low"] = (midpoint - half_width).clip(lower=0)
    frame["wilson_high"] = (midpoint + half_width).clip(upper=1)
    frame["below_min_volume"] = frame["n"] < PRIMARY_VOLUME
    return frame


def crude_rates(coverage, segments):
    columns = ["n", "n_fallout", "n_price_valid", "n_fallout_price_valid"]
    coverage = coverage.assign(
        n_fallout=coverage["n"].where(coverage["action_taken"].eq(2), 0),
        n_fallout_price_valid=coverage["n_price_valid"].where(coverage["action_taken"].eq(2), 0),
    )
    overall = grouped(coverage, ["dimension", "group"])[columns].sum().reset_index()
    overall["geo"] = ""
    overall["market"] = ""
    overall["loan_type"] = ""
    overall["lei"] = ""
    sections = [overall]
    local = segments.assign(
        n_fallout=segments["n"].where(segments["action_taken"].eq(2), 0),
        n_fallout_price_valid=segments["n_price_valid"].where(segments["action_taken"].eq(2), 0),
    )
    for keys, dimension in [
        (["geo", "market"], "market"),
        (MARKET, "market_product"),
        (GROUP, "lender_market_product"),
    ]:
        part = grouped(local, keys)[columns].sum().reset_index()
        part["dimension"] = dimension
        part["group"] = part["market"]
        sections.append(part)
    rates = pd.concat(sections, ignore_index=True)
    rates["priced_fallout_rate"] = rates["n_fallout_price_valid"] / rates["n_price_valid"].replace(0, np.nan)
    return add_wilson(rates)[[
        "dimension", "group", "geo", "lei", "market", "loan_type", *columns,
        "fallout_rate", "wilson_low", "wilson_high", "below_min_volume", "priced_fallout_rate",
    ]]


def standardize(cells, n_column, outcome_column, suffix=""):
    """Compare a lender with all other lenders in its geography and product."""
    keys = [*MARKET, *BANDS]
    same_stratum = grouped(cells, keys)
    n = cells[n_column].to_numpy()
    outcomes = cells[outcome_column].to_numpy()
    ref_n = same_stratum[n_column].transform("sum").to_numpy()
    peer_n = ref_n - n
    peer_outcomes = same_stratum[outcome_column].transform("sum").to_numpy() - outcomes
    support = (n > 0) & (peer_n > 0)
    weight = np.where(support, ref_n, 0)
    contributions = cells[GROUP].copy()
    contributions["n"] = n
    contributions["n_fallout"] = outcomes
    contributions["n_own_supported"] = np.where(support, n, 0)
    contributions["n_peer_supported"] = np.where(support, peer_n, 0)
    contributions["n_reference_supported"] = weight
    contributions["n_strata_supported"] = support.astype(np.int32)
    contributions["own_weighted"] = weight * np.divide(
        outcomes, n, out=np.zeros(len(n), dtype=float), where=support,
    )
    contributions["peer_weighted"] = weight * np.divide(
        peer_outcomes, peer_n, out=np.zeros(len(n), dtype=float), where=support,
    )
    result = grouped(contributions, GROUP).sum().reset_index()
    reference_total = grouped(result, MARKET)["n"].transform("sum")
    result["n_reference"] = reference_total
    result["n_peer"] = reference_total - result["n"]
    result["own_support_share"] = result["n_own_supported"] / result["n"].replace(0, np.nan)
    result["peer_support_share"] = result["n_peer_supported"] / result["n_peer"].replace(0, np.nan)
    result["reference_support_share"] = result["n_reference_supported"] / reference_total.replace(0, np.nan)
    denom = result["n_reference_supported"].replace(0, np.nan)
    result["adjusted_rate"] = result.pop("own_weighted") / denom
    result["peer_adjusted_rate"] = result.pop("peer_weighted") / denom
    result["adjusted_gap_pp"] = 100 * (result["adjusted_rate"] - result["peer_adjusted_rate"])
    own_total = grouped(result, MARKET)["n_fallout"].transform("sum")
    result["peer_crude_rate"] = (own_total - result["n_fallout"]) / result["n_peer"].replace(0, np.nan)
    result["crude_gap_pp"] = 100 * (result["n_fallout"] / result["n"].replace(0, np.nan) - result["peer_crude_rate"])
    return result.rename(columns={col: f"{col}{suffix}" for col in result if col not in GROUP})


def price_diagnostics(segments):
    price_metrics = ["n", "n_price_valid", "spread_mean", "q25", "median", "q75"]
    splits = {}
    for action in (1, 2):
        rows = segments.loc[segments["action_taken"].eq(action), [*GROUP, *price_metrics]].copy()
        if action == 1:
            valid = rows["n_price_valid"]
            rows["spread_sum"] = rows["spread_mean"].fillna(0) * valid
            market = grouped(rows, MARKET)
            rows["peer_n_originated"] = market["n"].transform("sum") - rows["n"]
            rows["peer_n_price_valid"] = market["n_price_valid"].transform("sum") - valid
            rows["peer_originated_coverage"] = rows["peer_n_price_valid"] / rows["peer_n_originated"].replace(0, np.nan)
            rows["peer_spread_mean"] = (
                market["spread_sum"].transform("sum") - rows.pop("spread_sum")
            ) / rows["peer_n_price_valid"].replace(0, np.nan)
            rows["relative_spread_pp"] = rows["spread_mean"] - rows["peer_spread_mean"]
        rows["coverage"] = rows["n_price_valid"] / rows["n"].replace(0, np.nan)
        label = "originated" if action == 1 else "fallout"
        splits[action] = rows.rename(columns={col: f"{col}_{label}" for col in [*price_metrics, "coverage"]})
    prices = splits[1].merge(splits[2], on=GROUP, how="outer", validate="one_to_one")
    for col in ("n", "n_price_valid"):
        for outcome in ("originated", "fallout"):
            key = f"{col}_{outcome}"
            prices[key] = prices[key].fillna(0).astype("int64")
    prices = prices.rename(columns={"n_fallout": "n_approved_not_accepted"})
    prices["matched_median_gap_pp"] = prices["median_fallout"] - prices["median_originated"]
    prices["matched_price_weight"] = prices[["n_price_valid_originated", "n_price_valid_fallout"]].min(axis=1)
    prices["matched_min_count"] = prices["matched_price_weight"] >= MIN_MATCHED_PRICES
    prices["matched_coverage_ok"] = (
        prices["coverage_originated"].ge(MIN_PRICE_COVERAGE)
        & prices["coverage_fallout"].ge(MIN_PRICE_COVERAGE)
        & prices["coverage_originated"].sub(prices["coverage_fallout"]).abs().le(MAX_PRICE_COVERAGE_GAP)
    )
    return prices


def eligible(segments, volume=PRIMARY_VOLUME, support=MIN_SUPPORT, coverage=MIN_PRICE_COVERAGE,
             both_outcomes=False, priced_only=False):
    suffix = "_priced" if priced_only else ""
    mask = (
        segments["n" + suffix].ge(volume)
        & segments["n_peer" + suffix].ge(volume)
        & segments["reference_support_share" + suffix].ge(support)
        & segments["n_price_valid_originated"].ge(MIN_ORIGINATED_PRICES)
        & segments["peer_n_price_valid"].ge(MIN_ORIGINATED_PRICES)
        & segments["coverage_originated"].ge(coverage)
        & segments["peer_originated_coverage"].ge(coverage)
    )
    if both_outcomes:
        mask &= (
            segments["matched_min_count"]
            & segments["coverage_fallout"].ge(coverage)
            & segments["coverage_originated"].sub(segments["coverage_fallout"]).abs().le(MAX_PRICE_COVERAGE_GAP)
        )
    return mask


def robust_results(segments):
    rules = [
        ("primary", {}),
        ("minimum_volume_50", {"volume": 50}),
        ("minimum_volume_200", {"volume": 200}),
        ("reference_support_80pct", {"support": 0.80}),
        ("originated_price_coverage_80pct", {"coverage": 0.80}),
        ("balanced_price_coverage_both_actions", {"both_outcomes": True}),
        ("material_gaps_10bp_and_1pp", {"min_price_gap": 0.10, "min_fallout_gap": 1.00}),
    ]
    if "adjusted_gap_pp_priced" in segments:
        rules.append(("price_available_outcome_cohort", {"priced_only": True}))
    primary = segments["primary_candidate"]
    sensitivity = []
    associations = []
    for rule, options in rules:
        eligibility = {key: value for key, value in options.items() if not key.startswith("min_")}
        use = eligible(segments, **eligibility)
        gap = "adjusted_gap_pp_priced" if options.get("priced_only") else "adjusted_gap_pp"
        price_gap = segments["relative_spread_pp"].round(10)
        fallout_gap = segments[gap].round(10)
        if "min_price_gap" in options:
            high_high = price_gap.ge(options["min_price_gap"]) & fallout_gap.ge(options["min_fallout_gap"])
        else:
            high_high = price_gap.gt(0) & fallout_gap.gt(0)
        sample = segments[["geo", "loan_type"]].copy()
        sample["eligible_segments"] = use.astype(int)
        sample["high_price_high_fallout"] = (use & high_high).astype(int)
        sample["primary_candidates"] = primary.astype(int)
        sample["primary_still_eligible"] = (primary & use).astype(int)
        sample["primary_retained"] = (primary & use & high_high).astype(int)
        measure_columns = [col for col in sample if col not in ("geo", "loan_type")]
        for keys in (["geo"], ["geo", "loan_type"]):
            counts = grouped(sample, keys)[measure_columns].sum().reset_index()
            if "loan_type" not in keys:
                counts["loan_type"] = "All"
            counts["scenario"] = rule
            sensitivity.append(counts)
        crude_gap = "crude_gap_pp_priced" if options.get("priced_only") else "crude_gap_pp"
        selected = segments.loc[use, ["geo", "loan_type", "relative_spread_pp", gap, crude_gap]]
        for keys in ("geo", ["geo", "loan_type"]):
            for key, part in grouped(selected, keys):
                geo, loan_type = key if isinstance(key, tuple) else (key, "All")
                price_rank = part["relative_spread_pp"].rank()
                adjusted_rank = part[gap].rank()
                crude_rank = part[crude_gap].rank()
                associations.append({
                    "scenario": rule,
                    "geo": geo,
                    "loan_type": loan_type,
                    "eligible_segments": len(part),
                    "spearman_price_vs_crude_gap": (
                        price_rank.corr(crude_rank) if price_rank.nunique() > 1 and crude_rank.nunique() > 1 else np.nan
                    ),
                    "spearman_price_vs_adjusted_gap": (
                        price_rank.corr(adjusted_rank)
                        if price_rank.nunique() > 1 and adjusted_rank.nunique() > 1 else np.nan
                    ),
                    "median_abs_change_after_adjustment_pp": float(part[gap].sub(part[crude_gap]).abs().median()),
                    "share_reversing_direction_after_adjustment": float((part[gap] * part[crude_gap] < 0).mean()),
                })
        segments[f"eligible_{rule}"] = use
        segments[f"candidate_{rule}"] = use & high_high
    counts = pd.concat(sensitivity, ignore_index=True)[[
        "scenario", "geo", "loan_type", "eligible_segments", "high_price_high_fallout",
        "primary_candidates", "primary_still_eligible", "primary_retained",
    ]]
    return counts, pd.DataFrame(associations)


def matched_summary(prices):
    records = []
    rules = [
        ("minimum_10_prices_each", prices["matched_min_count"]),
        ("balanced_coverage", prices["matched_min_count"] & prices["matched_coverage_ok"]),
        ("both_coverage_80pct", prices["matched_min_count"]
         & prices["coverage_originated"].ge(0.80) & prices["coverage_fallout"].ge(0.80)),
    ]
    for rule, use in rules:
        matched = prices.loc[use & prices["matched_median_gap_pp"].notna()]
        for keys in (["geo"], ["geo", "loan_type"]):
            grouping = keys[0] if len(keys) == 1 else keys
            for key, part in grouped(matched, grouping):
                geo, loan_type = key if isinstance(key, tuple) else (key, "All")
                weight = part["matched_price_weight"]
                gap = part["matched_median_gap_pp"]
                records.append({
                    "scenario": rule,
                    "geo": geo,
                    "loan_type": loan_type,
                    "n_strata": len(part),
                    "n_originated_prices": int(part["n_price_valid_originated"].sum()),
                    "n_fallout_prices": int(part["n_price_valid_fallout"].sum()),
                    "matched_price_weight_total": int(weight.sum()),
                    "weighted_mean_of_median_gaps_pp": float(np.average(gap, weights=weight)),
                    "median_of_stratum_gaps_pp": float(gap.median()),
                    "share_strata_positive": float(gap.gt(0).mean()),
                })
    summary = pd.DataFrame(records, columns=[
        "scenario", "geo", "loan_type", "n_strata", "n_originated_prices", "n_fallout_prices",
        "matched_price_weight_total", "weighted_mean_of_median_gaps_pp",
        "median_of_stratum_gaps_pp", "share_strata_positive",
    ])
    counts = ["n_price_valid_originated", "n_price_valid_fallout"]
    overall = grouped(prices, ["geo"])[counts].sum().reset_index().assign(loan_type="All")
    by_product = grouped(prices, ["geo", "loan_type"])[counts].sum().reset_index()
    summary = summary.merge(pd.concat([overall, by_product], ignore_index=True),
                            on=["geo", "loan_type"], how="left", validate="many_to_one")
    summary["share_valid_originated_prices_matched"] = (
        summary["n_originated_prices"] / summary["n_price_valid_originated"]
    )
    summary["share_valid_fallout_prices_matched"] = (
        summary["n_fallout_prices"] / summary["n_price_valid_fallout"]
    )
    return summary


def make_plots(path, waterfall, coverage, rates, segments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
    })
    blue, orange, gray = "#0072B2", "#B84E13", "#57616C"

    def save(fig, name):
        fig.savefig(path / f"{name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(waterfall) + 1.5)), layout="constrained")
    step_labels = {
        "input": "Public LAR records",
        "year": "Reporting year 2024",
        "action": "Originated or approved, not accepted",
        "purpose": "Home purchase",
        "lien": "First lien",
        "dwelling": "Site-built, 1–4 units",
        "occupancy": "Principal residence",
        "closed_end": "Closed-end credit",
        "non_reverse": "Exclude reverse mortgages",
    }
    labels = waterfall["step"].map(step_labels).fillna(waterfall["step"])
    counts = waterfall["n"]
    bars = ax.barh(labels, counts, color=blue)
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.19)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set(xlabel="Public LAR records", title="HMDA 2024 cohort selection")
    save(fig, "hmda_cohort_flow")

    fig, ax = plt.subplots(figsize=(8.3, 4.6), layout="constrained")
    cov = coverage.loc[coverage["dimension"].isin(["overall", "product"])].copy()
    cov["share"] = cov["n_price_valid"] / cov["n"].replace(0, np.nan)
    cov["label"] = cov["group"].map(PRODUCTS).fillna(cov["group"])
    cov = cov.pivot(index="label", columns="action_taken", values="share")
    order = [name for name in ["All", *PRODUCTS.values()] if name in cov.index]
    cov = cov.reindex(order or cov.index)
    x = np.arange(len(cov))
    for action, shift, color, hatch, label in [
        (1, -0.18, blue, "", "Originated"),
        (2, 0.18, orange, "//", "Approved, not accepted"),
    ]:
        if action in cov:
            bars = ax.bar(x + shift, cov[action], 0.34, label=label, color=color, hatch=hatch)
            ax.bar_label(bars, labels=[f"{v:.1%}" if pd.notna(v) else "" for v in cov[action]],
                         padding=3, fontsize=8)
    ax.set_xticks(x, cov.index)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(ylim=(0, 1.20), ylabel="Records with valid rate spread", title="Price reporting by application outcome")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    save(fig, "hmda_price_coverage")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    product_rates = rates.loc[rates["dimension"].eq("product")].sort_values("group")
    y = np.arange(len(product_rates))
    rate = product_rates["fallout_rate"]
    ax.errorbar(rate, y, xerr=np.vstack([
        (rate - product_rates["wilson_low"]).clip(lower=0),
        (product_rates["wilson_high"] - rate).clip(lower=0),
    ]), fmt="o", capsize=3, color=blue, label="Rate and 95% Wilson interval")
    names = product_rates["group"].map(PRODUCTS).fillna(product_rates["group"])
    ax.set_yticks(y, [f"{name}  (n={n:,})" for name, n in zip(names, product_rates["n"])])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlim=(0, None), xlabel="Approved but not accepted / approved",
           title="Post-approval non-origination by loan product")
    ax.legend(loc="center left", frameon=False)
    save(fig, "hmda_fallout_by_product")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), sharex=True, sharey=True, layout="constrained")
    for geo, title, ax in zip(("county", "msa"), ("County", "MSA / metro division"), axes):
        plot = segments.loc[segments["geo"].eq(geo) & segments["primary_eligible"]]
        ax.axhline(0, color=gray, lw=0.8)
        ax.axvline(0, color=gray, lw=0.8)
        for flag, color, marker, label in [
            (False, blue, "o", "Other eligible segments"),
            (True, orange, "^", "Higher price and fallout"),
        ]:
            points = plot.loc[plot["primary_candidate"].eq(flag)]
            ax.scatter(points["relative_spread_pp"], points["adjusted_gap_pp"], s=19,
                       color=color, marker=marker, alpha=0.57, linewidths=0, label=label)
        ax.set_title(f"{title}  (n={len(plot):,})")
        ax.set_xlabel("Mean originated rate spread gap (pp)")
    axes[0].set_ylabel("Adjusted fallout rate gap (pp)")
    axes[1].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("Lender segments versus peers in the same market and product", fontweight="bold")
    save(fig, "hmda_price_fallout_diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path(__file__).parent / "hmda_tables")
    parser.add_argument("--figures", type=Path, default=Path(__file__).parent / "hmda_figures")
    args = parser.parse_args()
    tables = args.tables
    args.figures.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "hmda-matplotlib"))

    waterfall = pd.read_csv(tables / "waterfall.csv")
    coverage = pd.read_csv(tables / "coverage.csv", dtype={"group": str}, keep_default_na=False)
    segments = pd.read_csv(tables / "segments.csv", dtype={col: str for col in GROUP})
    segments = segments.dropna(subset=GROUP)
    cells_path = tables / "cells.csv"
    if not cells_path.exists():
        cells_path = cells_path.with_suffix(".csv.gz")
    cells = pd.read_csv(
        cells_path, dtype={col: "category" for col in [*GROUP, *BANDS]},
    ).dropna(subset=GROUP)

    rates = crude_rates(coverage, segments)
    rates.to_csv(tables / "crude_rates.csv", index=False)

    adjusted = standardize(cells, "n", "n_fallout")
    if {"n_price_valid", "n_fallout_price_valid"}.issubset(cells):
        priced_adjusted = standardize(cells, "n_price_valid", "n_fallout_price_valid", "_priced")
        adjusted = adjusted.merge(priced_adjusted, on=GROUP, validate="one_to_one")
    del cells
    prices = price_diagnostics(segments)
    adjusted = adjusted.merge(prices, on=GROUP, how="left", validate="one_to_one")
    assert (adjusted["n"] == adjusted["n_originated"] + adjusted["n_approved_not_accepted"]).all(), "Segment counts disagree"
    assert (adjusted["n_fallout"] == adjusted["n_approved_not_accepted"]).all(), "Fallout counts disagree"
    if "n_priced" in adjusted:
        assert (adjusted["n_priced"] == adjusted["n_price_valid_originated"] + adjusted["n_price_valid_fallout"]).all(), "Valid-price counts disagree"
    adjusted["primary_eligible"] = eligible(adjusted)
    adjusted["primary_candidate"] = (
        adjusted["primary_eligible"]
        & adjusted["relative_spread_pp"].round(10).gt(0)
        & adjusted["adjusted_gap_pp"].round(10).gt(0)
    )
    robustness, associations = robust_results(adjusted)
    output_segments = adjusted.loc[adjusted["n"].ge(50) & adjusted["n_peer"].ge(50)]
    output_segments.sort_values(GROUP).to_csv(tables / "lender_market_diagnostics.csv", index=False)
    robustness.to_csv(tables / "robustness.csv", index=False)
    associations.to_csv(tables / "price_fallout_association.csv", index=False)
    matched = matched_summary(prices)
    matched.to_csv(tables / "matched_price_summary.csv", index=False)
    primary = adjusted.loc[adjusted["primary_candidate"]].copy()
    lender_products = grouped(primary, ["lei", "loan_type"])["geo"].nunique().rename("geographies")
    primary = primary.join(lender_products, on=["lei", "loan_type"])
    primary.sort_values(["geo", "adjusted_gap_pp"], ascending=[True, False]).to_csv(
        tables / "candidate_segments.csv", index=False,
    )

    overall = coverage.loc[coverage["dimension"].eq("overall")].set_index("action_taken")
    n_orig, n_fall = [int(overall.at[action, "n"]) for action in (1, 2)]
    priced_orig, priced_fall = [int(overall.at[action, "n_price_valid"]) for action in (1, 2)]
    coverage_orig = priced_orig / n_orig if n_orig else 0
    coverage_fall = priced_fall / n_fall if n_fall else 0
    strong_coverage = (
        min(coverage_orig, coverage_fall) >= MIN_PRICE_COVERAGE
        and abs(coverage_orig - coverage_fall) <= MAX_PRICE_COVERAGE_GAP
    )
    summary = {
        "n_approved": n_orig + n_fall,
        "n_originated": n_orig,
        "n_approved_not_accepted": n_fall,
        "fallout_rate": n_fall / (n_orig + n_fall),
        "n_priced_originated": priced_orig,
        "n_priced_approved_not_accepted": priced_fall,
        "price_coverage_originated": coverage_orig,
        "price_coverage_approved_not_accepted": coverage_fall,
        "coverage_gate_pass": strong_coverage,
        "primary_diagnostics": robustness.loc[robustness["scenario"].eq("primary") & robustness["loan_type"].eq("All")]
            .to_dict("records"),
        "matched_price": matched.loc[matched["loan_type"].eq("All")].to_dict("records"),
        "methods": {
            "outcome": "Action 2 divided by actions 1 and 2 in the filtered cohort. All rates are proportions unless labeled pp.",
            "small_cells": "Rates with n below 100 are flagged. The adjusted diagnostics export omits segments with fewer than 50 own or peer records. Wilson intervals are descriptive binomial intervals and omit lender and market dependence.",
            "standardization": "For each lender within geography × market × product, weight its own and other lenders' observed stratum rates by the same pooled market/product case-mix weights. Restrict and renormalize those weights to strata with at least one record on each side. Support shares refer to the unrenormalized counts. Adjusted gaps are descriptive, and support sets can differ between lenders.",
            "price_metric": "Difference in valid originated-loan mean rate spread between the lender and all other lenders in its market and product. Means are record-weighted. Price reflects originated records, while the primary fallout outcome uses all approvals.",
            "price_coverage_gate": "Treat matched price evidence as a limited extension when either action has under 50% valid pricing or their nationwide coverage gap exceeds 20 percentage points. This threshold is an analysis rule, not evidence that reporting is random when passed.",
            "matched_price": "Subtract the originated median spread from the approved-not-accepted median within each lender × market × product with at least 10 valid prices per action. Summary averages weight each contrast by the smaller of its two priced counts. Gaps remain conditional on reported prices.",
            "primary_eligibility": "Own and peer n at least 100 each, reference case-mix support at least 60%, and own and peer originated price coverage at least 50% with at least 30 valid originated prices each. A candidate has a strictly positive price gap and a strictly positive adjusted fallout gap. This is an exploratory screen, not a significance test.",
            "association": "Unweighted Spearman rank correlation across eligible lender–market–product segments. This segment-level association is descriptive and differs from a borrower-level price effect.",
            "geography": "County and MSA/metro division estimates use their respective observed geographic codes. In the candidate table, geographies=2 means the lender and product were flagged in at least one market at both geographic levels.",
            "material_gap_sensitivity": "Uses the primary eligibility rules and retains segments at least 10 basis points above peer originated rate spread and at least 1 percentage point above adjusted peer fallout. These are exploratory size thresholds.",
            "price_available_sensitivity": "Uses only valid price records for the outcome, and recalculates support and reference weights within that subset. Originated-loan relative price is unchanged.",
        },
    }
    (tables / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    make_plots(args.figures, waterfall, coverage, rates, adjusted)
    print(f"Approved: {n_orig + n_fall:,}; Fallout: {n_fall:,}; price coverage: {coverage_orig:.2%} / {coverage_fall:.2%}")
    print(f"Eligible diagnostic segments: {int(adjusted['primary_eligible'].sum()):,}; candidates: {len(primary):,}")


if __name__ == "__main__":
    main()

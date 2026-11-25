"""Follow-up checks on the diagnostic layer, computed from the Spark-built tables:
lender reporting concentration, sensitivity to anomalous reporters and to Unknown
case-mix strata, a permutation baseline for the flag count, and the case-mix
composition of the matched price segments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hmda_results as hr
from hmda_results import BANDS, GROUP, MARKET, grouped


def standardize_without_unknown(cells, n_column, outcome_column, suffix=""):
    """hmda_results.standardize with Unknown band strata treated as unsupported."""
    unknown = np.zeros(len(cells), dtype=bool)
    for band in BANDS:
        unknown |= (cells[band].astype(str) == "Unknown").to_numpy()
    keys = [*MARKET, *BANDS]
    same = grouped(cells, keys)
    n = cells[n_column].to_numpy()
    outcomes = cells[outcome_column].to_numpy()
    ref_n = same[n_column].transform("sum").to_numpy()
    peer_n = ref_n - n
    peer_out = same[outcome_column].transform("sum").to_numpy() - outcomes
    support = (n > 0) & (peer_n > 0) & ~unknown
    weight = np.where(support, ref_n, 0)
    parts = cells[GROUP].copy()
    parts["n"] = n
    parts["n_fallout"] = outcomes
    parts["n_reference_supported"] = weight
    parts["own_weighted"] = weight * np.divide(outcomes, n, out=np.zeros(len(n)), where=support)
    parts["peer_weighted"] = weight * np.divide(peer_out, peer_n, out=np.zeros(len(n)), where=support)
    result = grouped(parts, GROUP).sum().reset_index()
    total = grouped(result, MARKET)["n"].transform("sum")
    result["n_peer"] = total - result["n"]
    result["reference_support_share"] = result["n_reference_supported"] / total.replace(0, np.nan)
    denom = result["n_reference_supported"].replace(0, np.nan)
    result["adjusted_gap_pp"] = 100 * (result.pop("own_weighted") - result.pop("peer_weighted")) / denom
    own_total = grouped(result, MARKET)["n_fallout"].transform("sum")
    result["crude_gap_pp"] = 100 * (result["n_fallout"] / result["n"].replace(0, np.nan)
                                    - (own_total - result["n_fallout"]) / result["n_peer"].replace(0, np.nan))
    keep = ["n", "n_fallout", "n_peer", "reference_support_share", "adjusted_gap_pp", "crude_gap_pp"]
    return result[[*GROUP, *keep]].rename(columns={c: c + suffix for c in keep})


def diagnose(segments, cells, label, standardize=hr.standardize):
    adjusted = standardize(cells, "n", "n_fallout")
    adjusted = adjusted.merge(standardize(cells, "n_price_valid", "n_fallout_price_valid", "_priced"),
                              on=GROUP, validate="one_to_one")
    prices = hr.price_diagnostics(segments)
    adjusted = adjusted.merge(prices, on=GROUP, how="left", validate="one_to_one")
    adjusted["primary_eligible"] = hr.eligible(adjusted)
    adjusted["primary_candidate"] = (adjusted["primary_eligible"] & adjusted["relative_spread_pp"].round(10).gt(0)
                                     & adjusted["adjusted_gap_pp"].round(10).gt(0))
    robustness, association = hr.robust_results(adjusted)
    matched = hr.matched_summary(prices)
    rows = []
    for geo in ("county", "msa"):
        r = robustness.query("scenario == 'primary' and geo == @geo and loan_type == 'All'").iloc[0]
        a = association.query("scenario == 'primary' and geo == @geo and loan_type == 'All'").iloc[0]
        m = matched.query("scenario == 'minimum_10_prices_each' and geo == @geo and loan_type == 'All'").iloc[0]
        rows.append({"scenario": label, "geo": geo, "eligible_segments": int(r.eligible_segments),
                     "flags": int(r.high_price_high_fallout), "spearman": a.spearman_price_vs_adjusted_gap,
                     "matched_segments": int(m.n_strata), "matched_gap_bp": 100 * m.weighted_mean_of_median_gaps_pp,
                     "share_positive": m.share_strata_positive})
    return rows, adjusted


def flag_null(adjusted, reps, seed=2024):
    """Flags expected if price gaps were unrelated to adjusted gaps: price gaps are permuted
    across eligible segments within each geography and product."""
    rng = np.random.default_rng(seed)
    rows = []
    for geo, part in adjusted.loc[adjusted.primary_eligible].groupby("geo"):
        price = part.relative_spread_pp.round(10).gt(0).to_numpy()
        fallout = part.adjusted_gap_pp.round(10).gt(0).to_numpy()
        products = part.loan_type.to_numpy()
        observed = int((price & fallout).sum())
        draws = np.empty(reps)
        for i in range(reps):
            shuffled = price.copy()
            for p in np.unique(products):
                idx = np.flatnonzero(products == p)
                shuffled[idx] = rng.permutation(price[idx])
            draws[i] = (shuffled & fallout).sum()
        peer = part.peer_adjusted_rate.clip(1e-6, 1 - 1e-6)
        se = 100 * np.sqrt(peer * (1 - peer) / part.n_own_supported)
        flagged = part.primary_candidate.to_numpy()
        z = (part.adjusted_gap_pp / se).to_numpy()
        rows.append({"geo": geo, "eligible_segments": len(part), "observed_flags": observed,
                     "share_price_gap_positive": price.mean(), "share_adjusted_gap_positive": fallout.mean(),
                     "expected_flags_if_independent": len(part) * price.mean() * fallout.mean(),
                     "permutation_mean": draws.mean(), "permutation_sd": draws.std(ddof=1),
                     "permutation_p_value": (np.sum(draws >= observed) + 1) / (reps + 1),
                     "flags_with_fallout_z_above_1_96": int((z[flagged] > 1.96).sum()),
                     "flags_with_price_gap_below_5bp": int((part.relative_spread_pp[flagged] < 0.05).sum())})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tables", type=Path, required=True)
    cli.add_argument("--reps", type=int, default=2000)
    args = cli.parse_args()
    T = args.tables
    segments = pd.read_csv(T / "segments.csv", dtype={c: str for c in GROUP}).dropna(subset=GROUP)
    cells_path = T / "cells.csv" if (T / "cells.csv").exists() else T / "cells.csv.gz"
    cells = pd.read_csv(cells_path, dtype={c: "category" for c in [*GROUP, *BANDS]}).dropna(subset=GROUP)
    coverage = pd.read_csv(T / "coverage.csv", dtype={"group": str}, keep_default_na=False)

    # Lender reporting profile
    lend = coverage[coverage.dimension.eq("lender")].pivot_table(
        index="group", columns="action_taken", values=["n", "n_price_valid"], aggfunc="sum").fillna(0)
    lend.columns = [f"{a}_{b}" for a, b in lend.columns]
    lend["non_origination_rate"] = lend.n_2 / (lend.n_1 + lend.n_2)
    lend["action2_price_coverage"] = lend.n_price_valid_2 / lend.n_2.replace(0, np.nan)
    county = cells[cells.geo.astype(str).eq("county")]
    unknown = county[county.dti_band.astype(str).eq("Unknown")].groupby("lei", observed=True).n_fallout.sum()
    lend["action2_unknown_dti_share"] = (unknown / county.groupby("lei", observed=True).n_fallout.sum()
                                         .replace(0, np.nan)).reindex(lend.index).fillna(0)
    rules = {
        "non_origination_above_50pct": lend.non_origination_rate > 0.5,
        "action2_price_coverage_below_50pct": (lend.action2_price_coverage < 0.5) & (lend.n_2 >= 20),
        "action2_unknown_dti_above_50pct": (lend.action2_unknown_dti_share > 0.5) & (lend.n_2 >= 20),
    }
    rules["any_rule"] = np.logical_or.reduce(list(rules.values()))
    missing_2 = lend.n_2 - lend.n_price_valid_2
    unknown_total = unknown.sum()
    audit = []
    for name, rule in rules.items():
        keep = lend[~rule]
        audit.append({"rule": name, "lenders": int(rule.sum()),
                      "share_of_action2_records": lend.n_2[rule].sum() / lend.n_2.sum(),
                      "share_of_missing_action2_spreads": missing_2[rule].sum() / missing_2.sum(),
                      "share_of_unknown_dti_action2_records": unknown.reindex(lend.index[rule]).fillna(0).sum() / unknown_total,
                      "price_coverage_action1_without": keep.n_price_valid_1.sum() / keep.n_1.sum(),
                      "price_coverage_action2_without": keep.n_price_valid_2.sum() / keep.n_2.sum(),
                      "non_origination_rate_without": keep.n_2.sum() / (keep.n_1 + keep.n_2).sum()})
    ranked = missing_2.sort_values(ascending=False).cumsum() / missing_2.sum()
    concentration = {f"top_{k}_lenders_share_of_missing_action2_spreads": float(ranked.iloc[k - 1]) for k in (1, 5, 10, 20)}
    ranked_u = unknown.sort_values(ascending=False).cumsum() / unknown_total
    concentration.update({f"top_{k}_lenders_share_of_unknown_dti_action2": float(ranked_u.iloc[k - 1]) for k in (1, 5, 10)})
    pd.DataFrame(audit).to_csv(T / "lender_reporting_audit.csv", index=False)
    lend.assign(**{k: v for k, v in rules.items()}).to_csv(T / "lender_reporting_lenders.csv")

    # Sensitivity of the main results
    rows, base = diagnose(segments, cells, "baseline")
    flagged = lend.index[rules["any_rule"]]
    extra, _ = diagnose(segments[~segments.lei.isin(flagged)], cells[~cells.lei.astype(str).isin(flagged)],
                        "excluding_anomalous_reporters")
    rows += extra
    extra, _ = diagnose(segments, cells, "unknown_strata_unsupported", standardize_without_unknown)
    rows += extra
    pd.DataFrame(rows).to_csv(T / "sensitivity_summary.csv", index=False)

    # Permutation baseline for the flag count
    pd.DataFrame(flag_null(base, args.reps)).to_csv(T / "flag_null.csv", index=False)

    # Case-mix composition of matched county segments by outcome
    prices = hr.price_diagnostics(segments)
    matched = prices.loc[prices.matched_min_count & prices.matched_median_gap_pp.notna() & prices.geo.eq("county"), GROUP]
    c = county.copy()
    for col in GROUP:
        c[col] = c[col].astype(str)
    c = c.merge(matched, on=GROUP)
    c["n_originated"] = c.n - c.n_fallout
    comp = []
    for band in BANDS:
        t = c.groupby(band, observed=True)[["n_originated", "n_fallout"]].sum()
        t = t / t.sum()
        comp += [{"band": band, "value": str(k), "share_originated": r.n_originated, "share_not_accepted": r.n_fallout}
                 for k, r in t.iterrows()]
    pd.DataFrame(comp).to_csv(T / "matched_composition.csv", index=False)
    rates = []
    for band in BANDS:
        t = county.groupby(band, observed=True)[["n", "n_fallout", "n_price_valid", "n_fallout_price_valid"]].sum()
        for k, r in t.iterrows():
            rates.append({"band": band, "value": str(k), "approvals": int(r.n),
                          "non_origination_rate": r.n_fallout / r.n,
                          "price_coverage_action2": r.n_fallout_price_valid / r.n_fallout if r.n_fallout else np.nan})
    pd.DataFrame(rates).to_csv(T / "rates_by_band.csv", index=False)
    (T / "sensitivity_concentration.json").write_text(json.dumps(concentration, indent=2))
    print(pd.DataFrame(rows).to_string(), flush=True)
    print(pd.read_csv(T / "flag_null.csv").T.to_string(), flush=True)
    print(json.dumps(concentration, indent=2))


if __name__ == "__main__":
    main()

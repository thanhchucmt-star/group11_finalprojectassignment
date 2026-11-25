"""Pandas build of the diagnostic inputs from staged records, followed by the original
Pandas diagnostics in hmda_results.py. Nothing here reads a Spark output table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hmda_results as hr
from hmda_defs import BANDS, GROUP, case_mix_bands, market_code, number, read_cohort

COLUMNS = ["lei", "county_code", "derived_msa_md", "loan_type", "action_taken", "rate_spread",
           "interest_rate", "loan_amount", "income", "debt_to_income_ratio", "combined_loan_to_value_ratio"]


def geo_frame(cohort, bands, geo, column):
    market = market_code(cohort[column])
    keep = market.notna().to_numpy() & cohort["lei"].notna().to_numpy() & cohort["loan_type"].notna().to_numpy()
    frame = pd.DataFrame({
        "geo": geo,
        "lei": cohort["lei"].astype(str).to_numpy()[keep],
        "market": market.to_numpy()[keep].astype(str),
        "loan_type": cohort["loan_type"].astype(str).to_numpy()[keep],
        "action_taken": cohort["action_taken"].astype(str).astype(int).to_numpy()[keep],
        "spread": cohort["spread"].to_numpy()[keep],
        "interest": cohort["interest"].to_numpy()[keep],
    })
    for name, values in bands.items():
        frame[name] = values[keep]
    for col in ["geo", "lei", "market", "loan_type", *BANDS]:
        frame[col] = frame[col].astype("category")
    return frame


def segment_table(frame, extra=()):
    keys = [*GROUP, *extra, "action_taken"]
    g = frame.groupby(keys, observed=True, sort=False)
    out = g.agg(n=("spread", "size"), n_price_valid=("spread", "count"), spread_mean=("spread", "mean"),
                n_rate_valid=("interest", "count"))
    priced = frame.loc[frame["spread"].notna()].groupby(keys, observed=True, sort=False)["spread"]
    q = priced.quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["q25", "median", "q75"]
    rate = frame.loc[frame["interest"].notna()].groupby(keys, observed=True, sort=False)["interest"].median()
    out = out.join(q).join(rate.rename("rate_median")).reset_index()
    for col in [*GROUP, *extra]:
        out[col] = out[col].astype(str)
    return out


def cell_table(frame):
    priced = frame["spread"].notna()
    fallout = frame["action_taken"].eq(2)
    work = frame[[*GROUP, *BANDS]].assign(
        n=1, n_fallout=fallout.astype(np.int64), n_price_valid=priced.astype(np.int64),
        n_fallout_price_valid=(fallout & priced).astype(np.int64))
    return work.groupby([*GROUP, *BANDS], observed=True, sort=False).sum().reset_index()


def extra_matched(seg, extra=()):
    """Interest-rate and amount-band matched comparisons, mirroring hmda_diag_spark.matched_summary."""
    keys = [*GROUP, *extra]
    one = seg.loc[seg.action_taken.eq(1)].set_index(keys)
    two = seg.loc[seg.action_taken.eq(2)].set_index(keys)
    both = one.join(two, how="outer", lsuffix="_1", rsuffix="_2")
    for c in ("n", "n_price_valid", "n_rate_valid"):
        both[[f"{c}_1", f"{c}_2"]] = both[[f"{c}_1", f"{c}_2"]].fillna(0)
    both["weight"] = both[["n_price_valid_1", "n_price_valid_2"]].min(axis=1)
    both["gap"] = both["median_2"] - both["median_1"]
    both["rate_weight"] = both[["n_rate_valid_1", "n_rate_valid_2"]].min(axis=1)
    both["rate_gap"] = both["rate_median_2"] - both["rate_median_1"]
    both = both.reset_index()
    by = ["geo", *extra]
    totals = both.groupby(by, observed=True)[["n_price_valid_1", "n_price_valid_2"]].sum()
    rows = []
    spread = both.loc[both.weight.ge(hr.MIN_MATCHED_PRICES) & both.gap.notna()]
    for key, part in spread.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        has_rate = part.rate_gap.notna()
        rows.append({**dict(zip(by, key)), "scenario": "minimum_10_prices_each", "n_strata": len(part),
                     "weighted_mean_of_median_gaps_pp": np.average(part.gap, weights=part.weight),
                     "share_strata_positive": part.gap.gt(0).mean(),
                     "share_valid_fallout_prices_matched": part.n_price_valid_2.sum() / totals.loc[key, "n_price_valid_2"],
                     "rate_gap_on_same_segments_pp": np.average(part.rate_gap[has_rate], weights=part.weight[has_rate])})
    rate = both.loc[both.rate_weight.ge(hr.MIN_MATCHED_PRICES) & both.rate_gap.notna()]
    for key, part in rate.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "scenario": "interest_rate_minimum_10_each", "n_strata": len(part),
                     "weighted_mean_of_median_gaps_pp": np.average(part.rate_gap, weights=part.rate_weight),
                     "share_strata_positive": part.rate_gap.gt(0).mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = read_cohort(args.staged, COLUMNS)
    cohort["spread"] = number(cohort["rate_spread"])
    cohort["interest"] = number(cohort["interest_rate"])
    bands = case_mix_bands(cohort)
    keep_cols = ["lei", "county_code", "derived_msa_md", "loan_type", "action_taken", "spread", "interest"]
    cohort = cohort[keep_cols]

    segments, cells, band_segments = [], [], []
    for geo, column in (("county", "county_code"), ("msa", "derived_msa_md")):
        frame = geo_frame(cohort, bands, geo, column)
        segments.append(segment_table(frame))
        band_segments.append(segment_table(frame, ["amount_band"]))
        cells.append(cell_table(frame))
        del frame
    del cohort, bands
    segments = pd.concat(segments, ignore_index=True)
    band_segments = pd.concat(band_segments, ignore_index=True)
    cells = pd.concat(cells, ignore_index=True)
    for col in [*GROUP, *BANDS]:
        cells[col] = cells[col].astype(str).astype("category")
    segments.to_parquet(args.out / "segments_pandas.parquet", index=False)
    cells.to_parquet(args.out / "cells_pandas.parquet", index=False)

    # Original Pandas diagnostics, now fed by Pandas-built inputs
    adjusted = hr.standardize(cells, "n", "n_fallout")
    priced = hr.standardize(cells, "n_price_valid", "n_fallout_price_valid", "_priced")
    adjusted = adjusted.merge(priced, on=GROUP, validate="one_to_one")
    del cells
    for col in GROUP:
        adjusted[col] = adjusted[col].astype(str)
    prices = hr.price_diagnostics(segments)
    adjusted = adjusted.merge(prices, on=GROUP, how="left", validate="one_to_one")
    adjusted["primary_eligible"] = hr.eligible(adjusted)
    adjusted["primary_candidate"] = (adjusted["primary_eligible"] & adjusted["relative_spread_pp"].round(10).gt(0)
                                     & adjusted["adjusted_gap_pp"].round(10).gt(0))
    robustness, associations = hr.robust_results(adjusted)
    robustness.to_csv(args.out / "robustness_pandas.csv", index=False)
    associations.to_csv(args.out / "association_pandas.csv", index=False)
    hr.matched_summary(prices).to_csv(args.out / "matched_pandas.csv", index=False)
    extra_matched(segments).to_csv(args.out / "matched_extra_pandas.csv", index=False)
    extra_matched(band_segments, ["amount_band"]).to_csv(args.out / "matched_by_amount_band_pandas.csv", index=False)

    adjusted["material_candidate"] = adjusted["candidate_material_gaps_10bp_and_1pp"]
    adjusted["priced_eligible"] = adjusted["eligible_price_available_outcome_cohort"]
    adjusted["priced_candidate"] = adjusted["candidate_price_available_outcome_cohort"]
    columns = [*GROUP, "n", "n_peer", "reference_support_share", "adjusted_gap_pp", "crude_gap_pp",
               "n_priced", "n_peer_priced", "reference_support_share_priced", "adjusted_gap_pp_priced",
               "relative_spread_pp", "n_price_valid_originated", "peer_n_price_valid",
               "primary_eligible", "primary_candidate", "material_candidate", "priced_eligible", "priced_candidate"]
    adjusted.loc[adjusted["n"].ge(50) & adjusted["n_peer"].ge(50), columns].sort_values(GROUP).to_csv(
        args.out / "diagnostics_pandas.csv", index=False)
    summary = {}
    for geo, part in adjusted.groupby("geo"):
        a = associations.set_index(["scenario", "geo", "loan_type"])
        summary[geo] = {
            "segments": len(part),
            "primary_eligible": int(part.primary_eligible.sum()),
            "primary_candidate": int(part.primary_candidate.sum()),
            "material_candidate": int(part.material_candidate.sum()),
            "priced_eligible": int(part.priced_eligible.sum()),
            "priced_candidate": int(part.priced_candidate.sum()),
            "spearman_primary": float(a.loc[("primary", geo, "All"), "spearman_price_vs_adjusted_gap"]),
            "spearman_price_available_outcome": float(
                a.loc[("price_available_outcome_cohort", geo, "All"), "spearman_price_vs_adjusted_gap"]),
        }
    (args.out / "diagnostics_pandas.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

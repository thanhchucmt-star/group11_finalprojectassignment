"""Compare the Pandas and Spark implementations output by output and write the
correctness table used in the report (quantity, Pandas, Spark, agreement, note).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = ["geo", "lei", "market", "loan_type"]
BANDS = ["amount_band", "income_band", "dti_band", "cltv_band"]
RTOL, ATOL = 1e-9, 1e-10
ROWS = []


def add(section, quantity, pandas_value, spark_value, note=""):
    p, s = pandas_value, spark_value
    if s is None:
        ROWS.append({"section": section, "quantity": quantity, "pandas": p, "spark": "",
                     "agreement": "Single engine", "note": note})
        return
    if isinstance(p, (bool, np.bool_)) or isinstance(s, (bool, np.bool_)):
        agree = "Exact" if p == s else "Differs"
    elif isinstance(p, (int, np.integer)) and isinstance(s, (int, np.integer)):
        agree = "Exact" if p == s else "Differs"
    elif isinstance(p, str) or isinstance(s, str):
        agree = "Exact" if p == s else "Differs"
    else:
        p, s = float(p), float(s)
        if p == s or (np.isnan(p) and np.isnan(s)):
            agree = "Exact"
        elif np.isclose(p, s, rtol=RTOL, atol=ATOL):
            agree = f"Within tolerance (|diff| = {abs(p - s):.1e})"
        else:
            agree = "Differs"
    ROWS.append({"section": section, "quantity": quantity, "pandas": pandas_value, "spark": spark_value,
                 "agreement": agree, "note": note})


def compare_frames(a, b, keys, exact, floats):
    """Return (rows, keys equal, count columns exact, max float |diff|, float mismatches)."""
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).sort_index()
    same_keys = a.index.equals(b.index)
    if not same_keys:
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
    counts_exact = all((a[c].fillna(-1).astype(np.int64).to_numpy() == b[c].fillna(-1).astype(np.int64).to_numpy()).all()
                       for c in exact)
    max_diff, bad = 0.0, 0
    for c in floats:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        both = ~np.isnan(x) & ~np.isnan(y)
        bad += int((np.isnan(x) != np.isnan(y)).sum())
        if both.any():
            max_diff = max(max_diff, float(np.abs(x[both] - y[both]).max()))
            bad += int((~np.isclose(x[both], y[both], rtol=RTOL, atol=ATOL)).sum())
    return len(a), same_keys, counts_exact, max_diff, bad


def frame_row(section, quantity, result, note=""):
    n, same, exact, diff, bad = result
    agree = "Exact" if same and exact and diff == 0 and bad == 0 else (
        f"Within tolerance (max |diff| = {diff:.1e})" if same and exact and bad == 0 else "Differs")
    ROWS.append({"section": section, "quantity": quantity, "pandas": f"{n:,} rows", "spark": f"{n:,} rows",
                 "agreement": agree, "note": note or ("" if same else "key sets differ")})


def compare_cells(pandas_path, spark_path):
    """Cell tables hold ~3 million rows; compare them as Arrow tables to stay within small machines' memory."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pv
    import pyarrow.parquet as pq
    keys = [*GROUP, *BANDS]
    counts = ["n", "n_fallout", "n_price_valid", "n_fallout_price_valid"]
    a = pq.read_table(pandas_path, columns=keys + counts)
    a = a.cast(pa.schema([(k, pa.string()) for k in keys] + [(c, pa.int64()) for c in counts]))
    b = pv.read_csv(spark_path, convert_options=pv.ConvertOptions(
        include_columns=keys + counts, column_types={**{k: pa.string() for k in keys}, **{c: pa.int64() for c in counts}}))
    b = b.filter(pc.and_kleene(pc.is_valid(b["market"]), pc.is_valid(b["lei"])))
    order = [(k, "ascending") for k in keys]
    a = a.take(pc.sort_indices(a, sort_keys=order))
    b = b.take(pc.sort_indices(b, sort_keys=order))
    same = a.num_rows == b.num_rows and all(a[c].equals(b[c]) for c in keys + counts)
    return {"section": "Diagnostic inputs", "quantity": "Case-mix cells (lender x market x product x 4 bands)",
            "pandas": f"{a.num_rows:,} rows", "spark": f"{b.num_rows:,} rows",
            "agreement": "Exact" if same else "Differs", "note": "keys and four counts compared row by row"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spark", type=Path, required=True)
    cli.add_argument("--pandas", type=Path, required=True)
    args = cli.parse_args()
    S, P = args.spark, args.pandas

    # 1. Raw file and cohort filters: PyArrow reads the CSV directly; Spark reads staged Parquet
    raw = json.loads((P / "raw_audit_pyarrow.json").read_text())
    spark_flow = pd.read_csv(S / "waterfall.csv").set_index("step")["n"]
    for step, n in raw["waterfall"].items():
        add("Data preparation", f"Records after filter step '{step}'", int(n), int(spark_flow[step]),
            "Pandas side reads the raw CSV with PyArrow, independent of Spark staging")
    lines = json.loads((P / "raw_line_duplicates.json").read_text()) if (P / "raw_line_duplicates.json").exists() else None
    dup = pd.read_csv(S / "possible_duplicates.csv").iloc[0]
    if lines:
        add("Data preparation", "Repeated public lines beyond first occurrence (full file)",
            lines["rows_beyond_first"], int(dup["rows_beyond_first"]), "MD5 of each physical line in both")
    add("Data preparation", "Repeated rows beyond first occurrence (analytical cohort)",
        raw["cohort_repeated_rows"]["rows_beyond_first"], None, "PyArrow only; Spark audit covers the full file")

    # 2. Grouped outcomes and prices by lender x county x product (original check)
    keys = ["lei", "county_code", "loan_type"]
    a = pd.read_csv(P / "groups.csv", dtype=dict.fromkeys(keys, str)).fillna({k: "" for k in keys})
    b = pd.read_csv(S / "groups_spark.csv", dtype=dict.fromkeys(keys, str)).fillna({k: "" for k in keys})
    frame_row("Grouped statistics", "Lender x county x product: n, action 2, valid prices, rate, mean, quartiles",
              compare_frames(a, b, keys, ["n", "n_fallout", "n_price_valid"], ["rate", "spread_mean", "q25", "median", "q75"]))
    add("Grouped statistics", "Approved records in cohort", int(a["n"].sum()), int(b["n"].sum()))
    add("Grouped statistics", "Approved but not accepted (action 2)", int(a["n_fallout"].sum()), int(b["n_fallout"].sum()))
    add("Grouped statistics", "Records with valid rate spread", int(a["n_price_valid"].sum()), int(b["n_price_valid"].sum()))

    # 3. Missing values, special codes and plausibility screens
    k = ["variable", "outcome", "status"]
    a = pd.read_csv(P / "profile_status_pandas.csv", dtype=str).astype({"count": int})
    b = pd.read_csv(S / "profile_status_spark.csv", dtype=str).astype({"count": int})
    frame_row("Data quality", "Valid / NA / Exempt / blank / other counts, 9 variables x 3 outcomes",
              compare_frames(a, b, k, ["count"], []))
    for var in ["rate_spread", "debt_to_income_ratio", "combined_loan_to_value_ratio", "county_code"]:
        x = a[(a.variable == var) & (a.outcome == "2") & (a.status != "valid")]["count"].sum()
        y = b[(b.variable == var) & (b.outcome == "2") & (b.status != "valid")]["count"].sum()
        add("Data quality", f"Action 2 records without a valid {var}", int(x), int(y))
    k = ["variable", "check", "outcome"]
    frame_row("Data quality", "Plausibility screens (negative income, LTV above 150, etc.)",
              compare_frames(pd.read_csv(P / "profile_checks_pandas.csv", dtype=str).astype({"count": int}),
                             pd.read_csv(S / "profile_checks_spark.csv", dtype=str).astype({"count": int}), k, ["count"], []))

    # 4. Descriptive statistics
    stats = ["mean", "sd", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    a = pd.read_csv(P / "profile_numeric_pandas.csv", dtype={"outcome": str})
    b = pd.read_csv(S / "profile_numeric_spark.csv", dtype={"outcome": str})
    frame_row("Descriptive statistics", "Count, mean, SD, min, 7 percentiles, max: 6 variables x 3 outcomes",
              compare_frames(a, b, ["variable", "outcome"], ["n_records", "n_valid"], stats))
    ai, bi = a.set_index(["variable", "outcome"]), b.set_index(["variable", "outcome"])
    for var, stat, label in [("loan_amount", "p50", "Median loan amount (USD), all approvals"),
                             ("income", "p50", "Median applicant income (USD thousands)"),
                             ("interest_rate", "mean", "Mean interest rate (%)"),
                             ("rate_spread", "p50", "Median rate spread (pp)"),
                             ("rate_spread", "mean", "Mean rate spread (pp)")]:
        add("Descriptive statistics", label, ai.loc[(var, "All"), stat], bi.loc[(var, "All"), stat])
    k = ["variable", "outcome", "bin"]
    frame_row("Descriptive statistics", "Histogram bin counts used in the distribution figures",
              compare_frames(pd.read_csv(P / "profile_hist_pandas.csv", dtype={"outcome": str}),
                             pd.read_csv(S / "profile_hist_spark.csv", dtype={"outcome": str}), k, ["count"], ["bin_lower"]))

    # 5. Diagnostic inputs built independently
    dtypes = dict.fromkeys(GROUP, str)
    a = pd.read_parquet(P / "segments_pandas.parquet")
    b = pd.read_csv(S / "segments.csv", dtype=dtypes).dropna(subset=GROUP)
    frame_row("Diagnostic inputs", "Lender x market x product x outcome: n, valid prices, mean, quartiles",
              compare_frames(a[[*GROUP, "action_taken", "n", "n_price_valid", "spread_mean", "q25", "median", "q75"]],
                             b, [*GROUP, "action_taken"], ["n", "n_price_valid"], ["spread_mean", "q25", "median", "q75"]))
    del a, b
    cells_path = S / "cells.csv" if (S / "cells.csv").exists() else S / "cells.csv.gz"
    ROWS.append(compare_cells(P / "cells_pandas.parquet", cells_path))

    # 6. Matched within-segment price comparison
    a = pd.read_csv(P / "matched_pandas.csv", dtype={"loan_type": str})
    b = pd.read_csv(S / "matched_spark.csv")
    a = a[a.loan_type.eq("All")].set_index(["geo", "scenario"])
    b = b.set_index(["geo", "scenario"])
    for geo in ["county", "msa"]:
        for scenario in ["minimum_10_prices_each", "balanced_coverage", "both_coverage_80pct"]:
            x, y = a.loc[(geo, scenario)], b.loc[(geo, scenario)]
            add("Matched price gap", f"{geo} / {scenario}: matched segments", int(x.n_strata), int(y.n_strata))
            add("Matched price gap", f"{geo} / {scenario}: weighted mean gap (pp)",
                x.weighted_mean_of_median_gaps_pp, y.weighted_mean_of_median_gaps_pp)
            add("Matched price gap", f"{geo} / {scenario}: share of segments with positive gap",
                x.share_strata_positive, y.share_strata_positive)
            add("Matched price gap", f"{geo} / {scenario}: share of valid action 2 prices matched",
                x.share_valid_fallout_prices_matched, y.share_valid_fallout_prices_matched)
    ex = pd.read_csv(P / "matched_extra_pandas.csv").set_index(["geo", "scenario"])
    for geo in ["county", "msa"]:
        add("Matched price gap", f"{geo}: interest-rate gap on the spread-matched segments (pp)",
            ex.loc[(geo, "minimum_10_prices_each"), "rate_gap_on_same_segments_pp"],
            b.loc[(geo, "minimum_10_prices_each"), "rate_gap_on_same_segments_pp"])
        add("Matched price gap", f"{geo}: segments with 10+ interest rates per outcome",
            int(ex.loc[(geo, "interest_rate_minimum_10_each"), "n_strata"]),
            int(b.loc[(geo, "interest_rate_minimum_10_each"), "n_strata"]))
        add("Matched price gap", f"{geo}: weighted interest-rate gap, 10+ rates per outcome (pp)",
            ex.loc[(geo, "interest_rate_minimum_10_each"), "weighted_mean_of_median_gaps_pp"],
            b.loc[(geo, "interest_rate_minimum_10_each"), "weighted_mean_of_median_gaps_pp"])
    a = pd.read_csv(P / "matched_by_amount_band_pandas.csv")
    b = pd.read_csv(S / "matched_by_amount_band_spark.csv")
    a = a[a.scenario.eq("minimum_10_prices_each")][["geo", "amount_band", "n_strata", "weighted_mean_of_median_gaps_pp"]]
    b = b[b.scenario.eq("minimum_10_prices_each")][["geo", "amount_band", "n_strata", "weighted_mean_of_median_gaps_pp"]]
    frame_row("Matched price gap", "Matched gap by loan-amount band (segments and weighted gap)",
              compare_frames(a, b, ["geo", "amount_band"], ["n_strata"], ["weighted_mean_of_median_gaps_pp"]))

    # 7. Peer standardization, screening and association
    a = pd.read_csv(P / "diagnostics_pandas.csv", dtype=dtypes)
    b = pd.read_csv(S / "diagnostics_spark.csv", dtype=dtypes)
    floats = ["reference_support_share", "adjusted_gap_pp", "crude_gap_pp", "reference_support_share_priced",
              "adjusted_gap_pp_priced", "relative_spread_pp"]
    frame_row("Peer diagnostics", "Segment-level adjusted gap, crude gap, support share, relative price",
              compare_frames(a, b, GROUP, ["n", "n_peer", "n_priced", "n_peer_priced", "n_price_valid_originated",
                                           "peer_n_price_valid"], floats),
              "First comparison differed in 5 segments with no originated loans: Pandas left the peer price "
              "count undefined, Spark computed it; Spark now follows the Pandas definition. None was eligible.")
    ai, bi = a.set_index(GROUP), b.set_index(GROUP)
    for flag in ["primary_eligible", "primary_candidate", "material_candidate", "priced_candidate"]:
        same = set(ai.index[ai[flag].astype(str).eq("True")]) == set(bi.index[bi[flag].astype(str).eq("True")])
        add("Peer diagnostics", f"Identical set of segments flagged '{flag}'", same, True)
    pj = json.loads((P / "diagnostics_pandas.json").read_text())
    sj = json.loads((S / "diagnostics_spark.json").read_text())
    for geo in ["county", "msa"]:
        for key in ["primary_eligible", "primary_candidate", "material_candidate", "priced_candidate"]:
            add("Peer diagnostics", f"{geo}: {key} segments", int(pj[geo][key]), int(sj[geo][key]))
        add("Peer diagnostics", f"{geo}: Spearman, price gap vs adjusted gap", pj[geo]["spearman_primary"],
            sj[geo]["spearman_primary"])
        add("Peer diagnostics", f"{geo}: Spearman, price-available outcome",
            pj[geo]["spearman_price_available_outcome"], sj[geo]["spearman_price_available_outcome"])

    # 8. Application-level model
    a = pd.read_csv(P / "price_model_pandas.csv").set_index("specification")
    b = pd.read_csv(S / "price_model.csv").set_index("specification")
    for spec in b.index:
        for col, label in [("n_identifying_records", "identifying records"), ("fixed_effect_strata", "strata"),
                           ("lender_clusters", "lender clusters"),
                           ("fallout_pp_per_spread_pp", "coefficient (pp per pp)"),
                           ("lender_clustered_se_pp", "clustered SE")]:
            x, y = a.loc[spec, col], b.loc[spec, col]
            if col.startswith(("n_", "fixed", "lender_clusters")):
                x, y = int(x), int(y)
            add("Price model", f"{spec}: {label}", x, y)
    for col in ["trim_lower_spread_pp", "trim_upper_spread_pp"]:
        add("Price model", f"central_98: {col}", a.loc["central_98_percent_price_case_mix", col],
            b.loc["central_98_percent_price_case_mix", col],
            "Both exact (inverted CDF / percentile_disc); the archived run used Spark approxQuantile")

    table = pd.DataFrame(ROWS)
    table.to_csv(S / "correctness_table.csv", index=False)
    summary = table.agreement.str.split(" ").str[0].value_counts().to_dict()
    print(table.to_string(max_colwidth=60))
    print(summary)


if __name__ == "__main__":
    main()

"""Pandas counterpart of hmda_price_model.py: lender-clustered within-stratum slope of
non-origination on rate spread, computed from the same sufficient statistics.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hmda_defs import case_mix_bands, market_code, number, read_cohort

MARKET_KEYS = ["lei", "county_code", "loan_type"]
BANDS = ["amount_band", "income_band", "dti_band", "cltv_band"]
COLUMNS = ["lei", "county_code", "loan_type", "action_taken", "rate_spread", "loan_amount", "income",
           "debt_to_income_ratio", "combined_loan_to_value_ratio"]


def fit(frame, strata, label):
    """A stratum identifies the slope when its spreads are not all equal; Spark's
    regr_sxx is exactly zero for constant spreads, so nunique() > 1 is the same rule."""
    g = frame.groupby(strata, observed=True, sort=False)
    x = frame["spread"].to_numpy()
    y = frame["fallout"].to_numpy(dtype=np.float64)
    dx = x - g["spread"].transform("mean").to_numpy()
    dy = y - g["fallout"].transform("mean").to_numpy(dtype=np.float64)
    moments = pd.DataFrame({"sxx": dx * dx, "sxy": dx * dy}, index=frame.index)
    moments[strata] = frame[strata]
    m = moments.groupby(strata, observed=True, sort=False).agg(sxx=("sxx", "sum"), sxy=("sxy", "sum"))
    m["n"] = g.size()
    m = m.loc[g["spread"].nunique() > 1].reset_index()
    clusters = m.groupby("lei", observed=True).agg(n=("n", "sum"), strata=("n", "size"),
                                                   sxx=("sxx", "sum"), sxy=("sxy", "sum"))
    n, k, g_count = int(clusters.n.sum()), int(clusters.strata.sum()), len(clusters)
    sxx = math.fsum(clusters.sxx)
    beta = math.fsum(clusters.sxy) / sxx
    variance = math.fsum((clusters.sxy - beta * clusters.sxx) ** 2) / sxx ** 2
    se = math.sqrt(variance * g_count / (g_count - 1) * (n - 1) / (n - k - 1))
    return dict(specification=label, n_identifying_records=n, fixed_effect_strata=k,
                lender_clusters=g_count, fallout_pp_per_spread_pp=100 * beta,
                lender_clustered_se_pp=100 * se, ci95_lower_pp=100 * (beta - 1.96 * se),
                ci95_upper_pp=100 * (beta + 1.96 * se))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = read_cohort(args.staged, COLUMNS)
    county = market_code(raw["county_code"])
    keep = county.notna().to_numpy() & raw["lei"].notna().to_numpy()
    bands = case_mix_bands(raw)
    base = pd.DataFrame({
        "lei": raw["lei"].astype(str).to_numpy()[keep],
        "county_code": county.to_numpy()[keep].astype(str),
        "loan_type": raw["loan_type"].astype(str).to_numpy()[keep],
        "spread": number(raw["rate_spread"])[keep],
        "fallout": raw["action_taken"].astype(str).eq("2").to_numpy()[keep].astype(np.int8),
        **{name: values[keep] for name, values in bands.items()},
    })
    del raw, bands
    for col in [*MARKET_KEYS, *BANDS]:
        base[col] = base[col].astype("category")
    valid = base.loc[base["spread"].notna()].reset_index(drop=True)
    strata = [*MARKET_KEYS, *BANDS]

    # Balanced-coverage segments: same rule as the Spark left_semi join
    v = base["spread"].notna()
    f = base["fallout"].eq(1)
    flags = pd.DataFrame({"n0": ~f, "n1": f, "v0": v & ~f, "v1": v & f}).astype(np.int64)
    flags[MARKET_KEYS] = base[MARKET_KEYS]
    cov = flags.groupby(MARKET_KEYS, observed=True).sum()
    ok = cov.loc[(cov.v0 >= 10) & (cov.v1 >= 10) & (cov.v0 / cov.n0 >= 0.5) & (cov.v1 / cov.n1 >= 0.5)
                 & ((cov.v0 / cov.n0 - cov.v1 / cov.n1).abs() <= 0.20)].index
    balanced = valid.set_index(MARKET_KEYS).index.isin(ok)

    identifying = valid.groupby(strata, observed=True, sort=False)["spread"].transform("nunique").gt(1)
    lower, upper = np.quantile(valid["spread"], [0.01, 0.99], method="inverted_cdf")
    rows = [
        fit(valid, MARKET_KEYS, "lender_county_product"),
        fit(valid.loc[identifying.to_numpy()].reset_index(drop=True), MARKET_KEYS,
            "lender_county_product_on_case_mix_sample"),
        fit(valid, strata, "lender_county_product_case_mix"),
        fit(valid.loc[balanced].reset_index(drop=True), strata, "balanced_price_coverage_case_mix"),
        fit(valid.loc[valid["spread"].between(lower, upper)].reset_index(drop=True), strata,
            "central_98_percent_price_case_mix"),
    ]
    rows[-1].update(trim_lower_spread_pp=lower, trim_upper_spread_pp=upper)
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "price_model_pandas.csv", index=False)
    print(out.to_string(), flush=True)


if __name__ == "__main__":
    main()

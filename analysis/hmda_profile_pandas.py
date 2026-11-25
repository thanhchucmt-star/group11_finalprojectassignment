"""Pandas counterpart of hmda_profile_spark.py on the same staged records."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmda_defs import number, read_cohort
from hmda_profile_defs import CHECKS, DTI_CATEGORIES, NUMERIC, OPS, PCT_NAMES, PCTS, STATUS_COLUMNS


def outcomes(frame):
    yield "All", frame
    for action, part in frame.groupby("action_taken", sort=True, observed=True):
        yield action, part


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    columns = ["action_taken", *NUMERIC, "debt_to_income_ratio", "county_code", "derived_msa_md"]
    cohort = read_cohort(args.staged, columns)
    values = {v: number(cohort[v]) for v in NUMERIC}

    stats, hist, status_rows, checks, dti = [], [], [], [], []
    for outcome, part in outcomes(cohort):
        idx = part.index.to_numpy()
        for v, (lo, hi, width) in NUMERIC.items():
            x = values[v][idx]
            x = x[~np.isnan(x)]
            q = np.quantile(x, PCTS) if len(x) else [np.nan] * len(PCTS)
            stats.append({"variable": v, "outcome": outcome, "n_records": len(part), "n_valid": len(x),
                          "mean": x.mean() if len(x) else np.nan,
                          "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                          "min": x.min() if len(x) else np.nan,
                          **dict(zip(PCT_NAMES, q)), "max": x.max() if len(x) else np.nan})
            nbins = int(round((hi - lo) / width))
            bins = np.floor((x - lo) / width)
            bins = np.where(bins < 0, -1, np.where(bins >= nbins, nbins, bins)).astype(int)
            uniq, counts = np.unique(bins, return_counts=True)
            hist += [{"variable": v, "outcome": outcome, "bin": int(b), "bin_lower": lo + b * width,
                      "bin_width": width, "count": int(c)} for b, c in zip(uniq, counts)]
            for label, op, threshold in CHECKS[v]:
                checks.append({"variable": v, "check": label, "outcome": outcome,
                               "count": int(OPS[op](x, threshold).sum())})
        for name in STATUS_COLUMNS:
            raw = part[name]
            upper = raw.str.upper()
            if name == "debt_to_income_ratio":
                valid = raw.isin(DTI_CATEGORIES).to_numpy() | ~np.isnan(number(raw))
            elif name in ("county_code", "derived_msa_md"):
                valid = (raw.str.fullmatch(r"[0-9]{5}").fillna(False) & raw.ne("99999")).to_numpy()
            else:
                valid = ~np.isnan(values[name][idx])
            status = np.select(
                [valid, upper.eq("NA").fillna(False).to_numpy(), upper.eq("EXEMPT").fillna(False).to_numpy(),
                 (raw.isna() | raw.eq("")).to_numpy(), raw.eq("99999").fillna(False).to_numpy()],
                ["valid", "na", "exempt", "blank", "code_99999"], "other")
            for key, count in pd.Series(status).value_counts().items():
                status_rows.append({"variable": name, "outcome": outcome, "status": key, "count": int(count)})
        for key, count in part["debt_to_income_ratio"].value_counts(dropna=False).items():
            dti.append({"outcome": outcome, "value": key, "count": int(count)})

    pd.DataFrame(stats).to_csv(args.out / "profile_numeric_pandas.csv", index=False)
    pd.DataFrame(hist).to_csv(args.out / "profile_hist_pandas.csv", index=False)
    pd.DataFrame(status_rows).to_csv(args.out / "profile_status_pandas.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "profile_checks_pandas.csv", index=False)
    pd.DataFrame(dti).to_csv(args.out / "profile_dti_pandas.csv", index=False)
    print("Saved Pandas profile", flush=True)


if __name__ == "__main__":
    main()

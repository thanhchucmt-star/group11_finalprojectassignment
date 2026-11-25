"""Who the closed-end and non-reverse filters remove: records coded 1111 (Exempt) by
institutions using the HMDA partial exemption, compared with the records the cohort keeps."""
import argparse
from pathlib import Path

import pandas as pd

from hmda_defs import FILTERS, dataset


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--staged", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    expr = None
    for _, column, values in FILTERS[:6]:  # filters before closed-end and non-reverse
        import pyarrow.dataset as ds
        term = ds.field(column).isin(list(values))
        expr = term if expr is None else expr & term
    cols = ["action_taken", "open_end_line_of_credit", "reverse_mortgage", "lei", "rate_spread"]
    t = dataset(args.staged).to_table(columns=cols, filter=expr).to_pandas()
    exempt = t.open_end_line_of_credit.eq("1111") | t.reverse_mortgage.eq("1111")
    kept = t.open_end_line_of_credit.eq("2") & t.reverse_mortgage.eq("2")
    other = ~exempt & ~kept
    lenders_kept, lenders_exempt = set(t.lei[kept]), set(t.lei[exempt])
    rows = []
    for name, mask in (("kept in cohort", kept), ("exempt code (1111)", exempt), ("open-end or reverse", other)):
        part = t[mask]
        size = part.groupby("lei").size()
        rows.append({
            "group": name, "records": len(part), "lenders": part.lei.nunique(),
            "share_not_accepted": part.action_taken.eq("2").mean(),
            "median_records_per_lender": size.median(),
            "share_price_reported_exempt": part.rate_spread.eq("Exempt").mean(),
        })
    out = pd.DataFrame(rows)
    out["records_before_filters"] = len(t)
    out["lenders_only_exempt"] = len(lenders_exempt - lenders_kept)
    out["lenders_in_both"] = len(lenders_exempt & lenders_kept)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "exempt_audit.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()

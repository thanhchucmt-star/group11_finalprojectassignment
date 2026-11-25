"""Independent Pandas check and benchmark for staged HMDA 2024 data."""

import argparse
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


KEYS = ["lei", "county_code", "loan_type"]
FILTERS = [
    ("year", "activity_year", ("2024",)),
    ("action", "action_taken", ("1", "2")),
    ("purpose", "loan_purpose", ("1",)),
    ("lien", "lien_status", ("1",)),
    ("dwelling", "derived_dwelling_category", ("Single Family (1-4 Units):Site-Built",)),
    ("occupancy", "occupancy_type", ("1",)),
    ("closed_end", "open_end_line_of_credit", ("2",)),
    ("non_reverse", "reverse_mortgage", ("2",)),
]
COLUMNS = KEYS + [column for _, column, _ in FILTERS] + ["rate_spread"]
COUNT_COLUMNS = ["n", "n_fallout", "n_price_valid"]


def cohort(frame, counts):
    """Apply cumulative filters and construct common features within a batch."""
    if counts is not None:
        counts[0] += len(frame)
    eligible = np.ones(len(frame), dtype=bool)
    for index, (_, column, valid_values) in enumerate(FILTERS, 1):
        eligible &= frame[column].isin(valid_values).to_numpy(dtype=bool)
        if counts is not None:
            counts[index] += np.count_nonzero(eligible)

    selected = frame.loc[eligible, KEYS].fillna("")
    selected["fallout"] = frame.loc[eligible, "action_taken"].eq("2").to_numpy(dtype=np.int8)
    spreads = pd.to_numeric(frame.loc[eligible, "rate_spread"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    spreads[~np.isfinite(spreads)] = np.nan
    selected["spread"] = spreads
    return selected


def scan(dataset, buckets, batch_size, counts=None):
    scanner = dataset.scanner(
        columns=COLUMNS,
        filter=ds.field("bucket") < buckets,
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas(types_mapper=pd.ArrowDtype)
        yield cohort(frame, counts)


def aggregate(frame, price_stat):
    return frame.groupby(KEYS, sort=False).agg(
        n=("fallout", "size"),
        n_fallout=("fallout", "sum"),
        n_price_valid=("spread", "count"),
        **{f"spread_{price_stat}": ("spread", price_stat)},
    )


def write_checks(dataset, out, batch_size):
    counts = np.zeros(len(FILTERS) + 1, dtype=np.int64)
    batches = list(scan(dataset, 4, batch_size, counts))
    eligible = pd.concat(batches, ignore_index=True)
    del batches
    groups = aggregate(eligible, "mean")
    price_data = eligible.loc[eligible["spread"].notna()]
    quantiles = (
        price_data.groupby(KEYS, sort=False)["spread"]
        .quantile([0.25, 0.5, 0.75], interpolation="linear")
        .unstack()
        .rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
        .reindex(columns=["q25", "median", "q75"])
    )
    groups["rate"] = groups["n_fallout"] / groups["n"]
    groups.join(quantiles).sort_index().to_csv(out / "groups.csv")
    pd.DataFrame({"step": ["input"] + [step for step, _, _ in FILTERS], "n": counts}).to_csv(
        out / "waterfall.csv", index=False
    )


def benchmark_run(dataset, buckets, batch_size):
    partial = [aggregate(frame, "sum") for frame in scan(dataset, buckets, batch_size)]
    combined = pd.concat(partial).groupby(level=KEYS, sort=False).sum()
    combined["rate"] = combined["n_fallout"] / combined["n"]
    combined["spread_mean"] = combined["spread_sum"] / combined["n_price_valid"].replace(0, np.nan)
    return {
        "n_groups": len(combined),
        "spread_sum": combined["spread_sum"].sum(),
        **combined[COUNT_COLUMNS].sum().to_dict(),
    }


def write_benchmark(dataset, out, batch_size, repeats):
    benchmark_run(dataset, 1, batch_size)  # discarded warm-up, mirrors the Spark benchmark
    rows = []
    for percent, buckets in [(25, 1), (50, 2), (100, 4)]:
        input_rows = dataset.count_rows(filter=ds.field("bucket") < buckets)
        for run in range(1, repeats + 1):
            started = perf_counter()
            checksum = benchmark_run(dataset, buckets, batch_size)
            elapsed = perf_counter() - started
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mib = peak / (1024 ** (2 if sys.platform == "darwin" else 1))
            rows.append({
                "engine": "Pandas",
                "mode": "uncached",
                "size_pct": percent,
                "repeat": run,
                "seconds": elapsed,
                "n_input": input_rows,
                "peak_rss_mib": rss_mib,
                **checksum,
            })
            print(f"Pandas {percent}% run {run}: {elapsed:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "benchmark_pandas.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(args.staged, format="parquet", partitioning="hive", exclude_invalid_files=True)
    if not args.benchmark_only:
        write_checks(dataset, args.out, args.batch_size)
    if args.benchmark_repeats:
        write_benchmark(dataset, args.out, args.batch_size, args.benchmark_repeats)


if __name__ == "__main__":
    main()

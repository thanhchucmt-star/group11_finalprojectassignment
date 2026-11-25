"""Assert agreement across independently calculated HMDA results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["lei", "county_code", "loan_type"]
COUNTS = ["n", "n_fallout", "n_price_valid"]
FLOATS = ["rate", "spread_mean", "q25", "median", "q75"]

def benchmark_table(spark, pandas):
    """Report table: medians, ranges, speedup = T_Pandas / T_Spark (uncached), memory notes."""
    rows = []
    for size, p in pandas.groupby("size_pct"):
        s = spark[spark["size_pct"] == size]
        su = s.loc[s["mode"] == "uncached", "seconds"]
        sc = s.loc[s["mode"] == "cached", "seconds"]
        sm = s.loc[s["mode"] == "cache_materialization"]
        speedup = p["seconds"].median() / su.median()
        rows.append({
            "size_pct": size, "input_rows": int(p["n_input"].iloc[0]),
            "pandas_median_s": p["seconds"].median(), "pandas_min_s": p["seconds"].min(),
            "pandas_max_s": p["seconds"].max(),
            "spark_uncached_median_s": su.median(), "spark_uncached_min_s": su.min(),
            "spark_uncached_max_s": su.max(),
            "spark_cached_median_s": sc.median(), "spark_cache_build_median_s": sm["seconds"].median(),
            "speedup_pandas_over_spark_uncached": speedup,
            "faster": "Spark" if speedup > 1 else "Pandas",
            "pandas_process_peak_rss_mib": p["peak_rss_mib"].max() if "peak_rss_mib" in p else np.nan,
            "spark_cache_memory_mib": pd.to_numeric(sm.get("cache_memory_bytes"), errors="coerce").max() / 2**20,
            "spark_cache_disk_mib": pd.to_numeric(sm.get("cache_disk_bytes"), errors="coerce").max() / 2**20,
        })
    return pd.DataFrame(rows)


def verify(spark_dir, pandas_dir):
    a = pd.read_csv(spark_dir / "waterfall.csv")
    b = pd.read_csv(pandas_dir / "waterfall.csv")
    pd.testing.assert_frame_equal(a, b)

    dtypes = dict.fromkeys(KEYS, str)
    a = pd.read_csv(spark_dir / "groups_spark.csv", dtype=dtypes)
    b = pd.read_csv(pandas_dir / "groups.csv", dtype=dtypes)
    a[KEYS] = a[KEYS].fillna("")
    b[KEYS] = b[KEYS].fillna("")
    a = a.set_index(KEYS).sort_index()
    b = b.set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    pd.testing.assert_frame_equal(a[COUNTS], b[COUNTS])
    np.testing.assert_allclose(a[FLOATS], b[FLOATS], rtol=1e-9, atol=1e-10, equal_nan=True)

    spark = pd.read_csv(spark_dir / "benchmark_spark.csv")
    pandas = pd.read_csv(pandas_dir / "benchmark_pandas.csv")
    keys = ["size_pct", "repeat"]
    count_cols = ["n_input", "n_groups", *COUNTS]
    pandas_checks = pandas.set_index(keys).sort_index()
    for mode in ["uncached", "cached"]:
        spark_checks = spark.loc[spark["mode"] == mode].set_index(keys).sort_index()
        pd.testing.assert_index_equal(spark_checks.index, pandas_checks.index)
        pd.testing.assert_frame_equal(spark_checks[count_cols], pandas_checks[count_cols], check_dtype=False)
        np.testing.assert_allclose(spark_checks["spread_sum"], pandas_checks["spread_sum"], rtol=1e-9, atol=1e-8)

    benchmark = pd.concat([spark, pandas], ignore_index=True)
    summary = benchmark.groupby(["size_pct", "engine", "mode"]).agg(
        input_rows=("n_input", "first"), repeats=("seconds", "size"),
        median_seconds=("seconds", "median"), min_seconds=("seconds", "min"),
        max_seconds=("seconds", "max"),
    ).reset_index()
    summary["actual_pct"] = summary["input_rows"] * 100 / summary["input_rows"].max()
    summary.to_csv(spark_dir / "benchmark_summary.csv", index=False)
    benchmark_table(spark, pandas).to_csv(spark_dir / "benchmark_table.csv", index=False)
    differences = (a[FLOATS] - b[FLOATS]).abs().max().to_dict()
    checks = {
        "status": "passed",
        "waterfall_steps": 9,
        "groups_compared": len(a),
        "cohort_records": int(a["n"].sum()),
        "group_count_fields_exact": COUNTS,
        "float_fields": FLOATS,
        "float_absolute_tolerance": 1e-10,
        "float_relative_tolerance": 1e-9,
        "max_abs_differences": differences,
        "benchmark_repeated_checks": 2 * len(pandas_checks),
        "benchmark_spread_sum_absolute_tolerance": 1e-8,
        "benchmark_spread_sum_relative_tolerance": 1e-9,
        "benchmark_scope": (
            "Both engines use the same nested samples of all source records on projected Parquet. "
            "Uncached means no application-level data cache; operating system disk cache was not flushed. "
            "Spark cache materialization times the read and cache action separately. "
            "One discarded warm-up run precedes timing in each engine. Medians are based on "
            f"{int(pandas_checks.groupby(level=0).size().iloc[0])} repeats per condition. "
            "The Spark cache holds only the columns the workload reads. Pandas streams Arrow batches."
        ),
    }
    (spark_dir / "engine_agreement.json").write_text(json.dumps(checks, indent=2))

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spark", required=True, type=Path)
    cli.add_argument("--pandas", required=True, type=Path)
    args = cli.parse_args()
    verify(args.spark, args.pandas)

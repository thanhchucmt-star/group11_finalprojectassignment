"""Schema record, value audit and descriptive statistics of the analytical cohort (Spark)."""
import argparse
import json
from pathlib import Path

from pyspark.sql import functions as F

from hmda_profile_defs import (CHECKS, DTI_CATEGORIES, NUMERIC, OPS, PCT_NAMES, PCTS,
                               STATUS_COLUMNS)
from hmda_spark import COLUMNS, build_session, export, features, number


def with_all(frame):
    """Stack an 'All' outcome under the per-action rows."""
    return frame.withColumn("outcome", F.col("action_taken")).unionByName(
        frame.withColumn("outcome", F.lit("All")))


def numeric_stats(cohort):
    values = cohort.withColumns({f"x_{v}": number(v) for v in NUMERIC})
    stacked = with_all(values)
    parts = []
    for v in NUMERIC:
        x = F.col(f"x_{v}")
        part = stacked.groupBy("outcome").agg(
            F.count("*").alias("n_records"), F.count(x).alias("n_valid"),
            F.avg(x).alias("mean"), F.stddev_samp(x).alias("sd"),
            F.min(x).alias("min"), F.max(x).alias("max"),
            F.expr(f"percentile(x_{v}, array({', '.join(map(str, PCTS))}))").alias("pct"),
        ).withColumn("variable", F.lit(v))
        parts.append(part.select("variable", "outcome", "n_records", "n_valid", "mean", "sd", "min",
                                 *(F.col("pct")[i].alias(name) for i, name in enumerate(PCT_NAMES)), "max"))
    result = parts[0]
    for part in parts[1:]:
        result = result.unionByName(part)
    return result


def histograms(cohort):
    stacked = with_all(cohort)
    parts = []
    for v, (lo, hi, width) in NUMERIC.items():
        x = number(v)
        nbins = int(round((hi - lo) / width))
        raw_bin = F.floor((x - F.lit(lo)) / F.lit(width))
        b = F.when(raw_bin < 0, -1).when(raw_bin >= nbins, nbins).otherwise(raw_bin)
        parts.append(stacked.where(x.isNotNull()).groupBy("outcome", b.cast("int").alias("bin"))
                     .count().withColumn("variable", F.lit(v))
                     .withColumn("bin_lower", F.lit(lo) + F.col("bin") * F.lit(width))
                     .withColumn("bin_width", F.lit(width)))
    result = parts[0]
    for part in parts[1:]:
        result = result.unionByName(part)
    return result.select("variable", "outcome", "bin", "bin_lower", "bin_width", "count")


def value_status(cohort):
    stacked = with_all(cohort)
    rows = []
    for name in STATUS_COLUMNS:
        raw = F.upper(F.col(name))
        if name == "debt_to_income_ratio":
            valid = F.col(name).isin(DTI_CATEGORIES) | number(name).isNotNull()
        elif name in ("county_code", "derived_msa_md"):
            valid = F.col(name).rlike(r"^[0-9]{5}$") & (F.col(name) != "99999")
        else:
            valid = number(name).isNotNull()
        status = (F.when(valid, "valid").when(raw == "NA", "na").when(raw == "EXEMPT", "exempt")
                  .when(raw.isNull() | (raw == ""), "blank")
                  .when(F.col(name) == "99999", "code_99999").otherwise("other"))
        rows.append(stacked.groupBy("outcome", status.alias("status")).count()
                    .withColumn("variable", F.lit(name)))
    result = rows[0]
    for part in rows[1:]:
        result = result.unionByName(part)
    return result.select("variable", "outcome", "status", "count")


def plausibility(cohort):
    stacked = with_all(cohort)
    aggs = []
    for v, checks in CHECKS.items():
        x = number(v)
        for label, op, threshold in checks:
            aggs.append(F.count_if(x.isNotNull() & OPS[op](x, F.lit(threshold))).alias(f"{v}__{label}"))
    wide = stacked.groupBy("outcome").agg(*aggs)
    long = wide.unpivot(["outcome"], [c for c in wide.columns if c != "outcome"], "check", "count")
    return (long.withColumn("variable", F.split("check", "__")[0])
            .withColumn("check", F.split("check", "__")[1]).select("variable", "check", "outcome", "count"))


def dti_categories(cohort):
    return with_all(cohort).groupBy("outcome", F.col("debt_to_income_ratio").alias("value")).count()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spark = build_session("HMDA profile")
    try:
        spark.sparkContext.setLogLevel("ERROR")
        header = spark.read.option("header", True).csv(str(args.raw)).schema
        staged = spark.read.parquet(str(args.staged))
        (args.out / "schema_spark.json").write_text(json.dumps({
            "raw_columns": len(header.fields),
            "raw_schema": [[f.name, f.dataType.simpleString()] for f in header.fields],
            "staged_schema": [[f.name, f.dataType.simpleString()] for f in staged.schema.fields],
            "analytic_columns": COLUMNS,
        }, indent=2))
        cohort = features(staged).cache()
        sort = ("variable", "outcome")
        export(numeric_stats(cohort), args.out / "profile_numeric_spark.csv", sort)
        export(histograms(cohort), args.out / "profile_hist_spark.csv", (*sort, "bin"))
        export(value_status(cohort), args.out / "profile_status_spark.csv", (*sort, "status"))
        export(plausibility(cohort), args.out / "profile_checks_spark.csv", ("variable", "check", "outcome"))
        export(dti_categories(cohort), args.out / "profile_dti_spark.csv", ("outcome", "value"))
        cohort.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

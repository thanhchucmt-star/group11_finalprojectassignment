"""HMDA 2024 cohort, Spark diagnostics, and matched Pandas/Spark benchmark."""
import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

COLUMNS = """activity_year action_taken loan_purpose lien_status derived_dwelling_category
occupancy_type open_end_line_of_credit reverse_mortgage lei county_code derived_msa_md
state_code loan_type loan_amount income debt_to_income_ratio combined_loan_to_value_ratio
rate_spread interest_rate loan_term""".split()
KEYS = ["lei", "county_code", "loan_type"]
FILTER_COLUMNS = ["activity_year", "action_taken", "loan_purpose", "lien_status",
                  "derived_dwelling_category", "occupancy_type", "open_end_line_of_credit",
                  "reverse_mortgage"]
BENCHMARK_COLUMNS = FILTER_COLUMNS + KEYS + ["rate_spread"]


def build_session(app):
    """Local Spark session. Defaults reproduce the archived run; env vars allow smaller machines."""
    return (SparkSession.builder.appName(app)
            .master(os.environ.get("HMDA_SPARK_MASTER", "local[6]"))
            .config("spark.driver.memory", os.environ.get("HMDA_DRIVER_MEMORY", "8g"))
            .config("spark.sql.shuffle.partitions", os.environ.get("HMDA_SHUFFLE_PARTITIONS", "32"))
            .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())
def filters():
    return [
        ("year", F.col("activity_year") == "2024"),
        ("action", F.col("action_taken").isin("1", "2")),
        ("purpose", F.col("loan_purpose") == "1"),
        ("lien", F.col("lien_status") == "1"),
        ("dwelling", F.col("derived_dwelling_category") == "Single Family (1-4 Units):Site-Built"),
        ("occupancy", F.col("occupancy_type") == "1"),
        ("closed_end", F.col("open_end_line_of_credit") == "2"),
        ("non_reverse", F.col("reverse_mortgage") == "2"),
    ]

def number(name):
    value = F.expr(f"try_cast(`{name}` as double)")
    return F.when(F.abs(value) < float("inf"), value)

def eligibility():
    keep = F.lit(True)
    for _, condition in filters():
        keep = keep & condition
    return keep

def features(frame):
    return frame.where(eligibility()).withColumns({
        "spread": number("rate_spread"),
        "fallout": (F.col("action_taken") == "2").cast("int"),
    })

def summaries():
    return [
        F.count("*").alias("n"), F.sum("fallout").alias("n_fallout"),
        F.count("spread").alias("n_price_valid"), F.avg("spread").alias("spread_mean"),
    ]

def quartiles():
    return [
        F.expr(f"percentile(spread, {p})").alias(name)
        for p, name in [(0.25, "q25"), (0.5, "median"), (0.75, "q75")]
    ]

def export(frame, destination, sort=()):
    if sort:
        frame = frame.orderBy(*sort)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(frame.columns)
        writer.writerows(frame.toLocalIterator())
    print(f"Saved {destination.name}", flush=True)

def rows_csv(destination, rows):
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def stage(spark, raw, staged):
    started = perf_counter()
    source = spark.read.option("header", True).option("mode", "FAILFAST").csv(str(raw))
    projected = source.select(*(F.trim(F.col(name)).alias(name) for name in COLUMNS))
    bucket = F.pmod(F.xxhash64(*COLUMNS), F.lit(4)).cast("int")
    projected.withColumn("bucket", bucket).write.partitionBy("bucket").parquet(str(staged))
    return perf_counter() - started

def duplicate_audit(spark, raw, out):
    # HMDA has one public record per line. Repeated public lines are retained.
    repeats = (
        spark.read.text(str(raw)).groupBy(F.md5("value").alias("row_hash"))
        .count().where("count > 1")
        .agg(F.count("*"), F.sum("count"), F.sum(F.col("count") - 1)).first()
    )
    rows_csv(out / "possible_duplicates.csv", [{
        "basis": "Identical public line (MD5 fingerprint), full input; not a loan identifier",
        "repeated_patterns": repeats[0], "rows_in_repeated_patterns": repeats[1] or 0,
        "rows_beyond_first": repeats[2] or 0,
    }])

def band(value, edges):
    result = F.lit(f"{edges[-1]}+")
    for lo, hi in reversed(list(zip(edges[:-1], edges[1:]))):
        result = F.when(value < hi, f"{lo}-<{hi}").otherwise(result)
    return F.when(value.isNull(), "Unknown").when(value < edges[0], f"<{edges[0]}").otherwise(result)

def case_mix_bands():
    dti = F.col("debt_to_income_ratio")
    dti_numeric = (F.when(dti == "<20%", 19).when(dti == "20%-<30%", 25)
                   .when(dti == "30%-<36%", 33).when(dti == "50%-60%", 55)
                   .when(dti == ">60%", 61).otherwise(number("debt_to_income_ratio")))
    return {
        "amount_band": band(number("loan_amount"), [150000, 250000, 400000, 650000]),
        "income_band": band(number("income"), [50, 100, 150, 250]),
        "dti_band": band(dti_numeric, [36, 43, 50]),
        "cltv_band": band(number("combined_loan_to_value_ratio"), [70, 80, 90, 97]),
    }

def analyze(frame, out):
    keep = F.lit(True)
    counts = [F.count("*").alias("input")]
    for name, condition in filters():
        keep = keep & condition
        counts.append(F.count_if(keep).alias(name))
    waterfall = frame.agg(*counts).first().asDict()
    rows_csv(out / "waterfall.csv", [{"step": k, "n": v} for k, v in waterfall.items()])
    before = frame.where(eligibility() | (
        (F.col("activity_year") == "2024") & F.col("action_taken").isin("1", "2")
        & (F.col("loan_purpose") == "1") & (F.col("lien_status") == "1")
        & (F.col("derived_dwelling_category") == "Single Family (1-4 Units):Site-Built")
        & (F.col("occupancy_type") == "1")
    ))
    export(before.groupBy("open_end_line_of_credit", "reverse_mortgage").count(),
           out / "status_audit.csv", ("open_end_line_of_credit", "reverse_mortgage"))
    cohort = features(frame).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        raw_price = F.upper(F.col("rate_spread"))
        status = (
            F.when(F.col("spread").isNotNull(), "valid")
            .when(raw_price == "NA", "na").when(raw_price == "EXEMPT", "exempt")
            .when(raw_price.isNull() | (raw_price == ""), "blank").otherwise("other")
        )
        coverage = cohort.withColumn("price_status", status).unpivot(
            ["action_taken", "price_status"],
            [F.lit("All").alias("overall"), F.col("loan_type").alias("product"),
             F.col("lei").alias("lender")], "dimension", "group",
        )
        export(coverage.groupBy("action_taken", "dimension", "group").agg(
            F.count("*").alias("n"),
            *(F.count_if(F.col("price_status") == k).alias("n_price_valid" if k == "valid" else "n_" + k)
              for k in ["valid", "na", "exempt", "blank", "other"]),
        ), out / "coverage.csv", ("dimension", "group", "action_taken"))
        export(cohort.groupBy(*KEYS).agg(*summaries(), *quartiles())
               .withColumn("rate", F.col("n_fallout") / F.col("n")),
               out / "groups_spark.csv", KEYS)
        fields = ["lei", "loan_type", "action_taken", "spread", "fallout",
                  "loan_amount", "income", "debt_to_income_ratio", "combined_loan_to_value_ratio"]
        geo = cohort.unpivot(fields, [F.col("county_code").alias("county"),
                                      F.col("derived_msa_md").alias("msa")], "geo", "market")
        geo = geo.withColumn("market", F.when(F.col("market").rlike(r"^[0-9]{5}$")
                                               & (F.col("market") != "99999"), F.col("market")))
        group = ["geo", "lei", "market", "loan_type"]
        export(geo.groupBy(*group, "action_taken").agg(
            F.count("*").alias("n"), F.count("spread").alias("n_price_valid"),
            F.avg("spread").alias("spread_mean"), *quartiles(),
        ), out / "segments.csv", [*group, "action_taken"])
        bands = case_mix_bands()
        export(geo.where(F.col("market").isNotNull()).withColumns(bands)
               .groupBy(*group, *bands).agg(
                   F.count("*").alias("n"), F.sum("fallout").alias("n_fallout"),
                   F.count("spread").alias("n_price_valid"),
                   F.count_if((F.col("fallout") == 1) & F.col("spread").isNotNull())
                   .alias("n_fallout_price_valid"),
               ), out / "cells.csv")
        missing = []
        for name in ["county_code", "derived_msa_md", "loan_amount", "income",
                     "debt_to_income_ratio", "combined_loan_to_value_ratio", "loan_term",
                     "interest_rate", "rate_spread"]:
            raw = F.upper(F.col(name))
            invalid = raw.isNull() | raw.isin("", "NA", "EXEMPT")
            if name == "derived_msa_md":
                invalid = invalid | (raw == "99999")
            missing.append(F.count_if(invalid).alias(name))
        export(cohort.groupBy("action_taken").agg(F.count("*").alias("n"), *missing),
               out / "missingness.csv", ("action_taken",))
    finally:
        cohort.unpersist()

def execute_workload(frame):
    aggregated = features(frame).groupBy(*KEYS).agg(*summaries())
    check = aggregated.agg(
        F.count("*"), F.sum("n"), F.sum("n_fallout"), F.sum("n_price_valid"),
        F.sum(F.col("spread_mean") * F.col("n_price_valid")),
    ).first()
    return check

def cache_storage(spark):
    """Memory and disk bytes held by cached blocks; empty when the JVM API is unavailable."""
    try:
        infos = spark.sparkContext._jsc.sc().getRDDStorageInfo()
        return sum(i.memSize() for i in infos), sum(i.diskSize() for i in infos)
    except Exception:
        return "", ""


def benchmark(spark, staged, out, repeats=int(os.environ.get("HMDA_BENCH_REPEATS", "5"))):
    """Time the shared workload. One discarded warm-up precedes all timed runs, and the
    cached frame holds only the columns the workload reads."""
    warm = [str(staged / "bucket=0")]
    execute_workload(spark.read.parquet(*warm))
    rows = []
    for size, buckets in [(25, 1), (50, 2), (100, 4)]:
        inputs = [str(staged / f"bucket={i}") for i in range(buckets)]
        actual = spark.read.parquet(*inputs).count()
        base = None
        for mode in ["uncached", "cache_materialization", "cached"]:
            for repeat in range(1, repeats + 1):
                mem_bytes = disk_bytes = ""
                if mode == "cache_materialization":
                    base = (spark.read.parquet(*inputs).select(*BENCHMARK_COLUMNS)
                            .persist(StorageLevel.MEMORY_AND_DISK))
                    started = perf_counter()
                    base.count()
                    seconds = perf_counter() - started
                    mem_bytes, disk_bytes = cache_storage(spark)
                    check = ("", "", "", "", "")
                    if repeat < repeats:
                        base.unpersist(blocking=True)
                else:
                    started = perf_counter()
                    check = execute_workload(spark.read.parquet(*inputs) if mode == "uncached" else base)
                    seconds = perf_counter() - started
                rows.append(dict(engine="Spark", size_pct=size, mode=mode, repeat=repeat,
                                 seconds=seconds, n_input=actual, n_groups=check[0], n=check[1],
                                 n_fallout=check[2], n_price_valid=check[3], spread_sum=check[4],
                                 cache_memory_bytes=mem_bytes, cache_disk_bytes=disk_bytes))
                print(f"Spark {size}% {mode} {repeat}: {seconds:.2f}s", flush=True)
        if base is not None:
            base.unpersist(blocking=True)
    rows_csv(out / "benchmark_spark.csv", rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "stage", "analysis", "benchmark", "run"], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    spark = build_session("HMDA 2024")
    try:
        spark.sparkContext.setLogLevel("ERROR")
        spark.range(1).count()
        startup = perf_counter() - started
        metadata = {
            "spark_version": spark.version, "spark_startup_plus_first_action_seconds": startup,
            "benchmark_repeats": int(os.environ.get("HMDA_BENCH_REPEATS", "5")),
            "benchmark_warmup": "one discarded run on bucket 0 before timing",
            "master": spark.sparkContext.master, "spark_driver_memory": spark.conf.get("spark.driver.memory"),
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "raw_name": args.raw.name, "raw_size_bytes": args.raw.stat().st_size,
            "logical_cpus": os.cpu_count(),
        }
        if args.phase in {"stage", "all"}:
            metadata["parquet_stage_seconds"] = stage(spark, args.raw, args.staged)
            print(f"Stage ready in {metadata['parquet_stage_seconds']:.2f}s", flush=True)
        if args.phase in {"analysis", "all", "run"}:
            analyze(spark.read.parquet(str(args.staged)), args.out)
            duplicate_audit(spark, args.raw, args.out)
        if args.phase in {"benchmark", "all", "run"}:
            benchmark(spark, args.staged, args.out)
        (args.out / f"spark_{args.phase}_metadata.json").write_text(json.dumps(metadata, indent=2))
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

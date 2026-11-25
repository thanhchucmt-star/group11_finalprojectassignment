"""Descriptive loan-level fixed-effects model on records with valid rate spread."""
import argparse
import math
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from hmda_spark import build_session, case_mix_bands, features, rows_csv

MARKET_KEYS = ["lei", "county_code", "loan_type"]

def fit(frame, strata, label):
    moments = frame.groupBy(*strata).agg(
        F.count("*").alias("n"),
        F.regr_sxx("fallout", "spread").alias("sxx"),
        F.regr_sxy("fallout", "spread").alias("sxy"),
    )
    clusters = moments.where("sxx > 0").groupBy("lei").agg(
        F.sum("n").alias("n"), F.count("*").alias("strata"), F.sum("sxx").alias("sxx"),
        F.sum("sxy").alias("sxy"),
    ).collect()
    n = sum(r.n for r in clusters)
    k = sum(r.strata for r in clusters)
    sxx = math.fsum(r.sxx for r in clusters)
    beta = math.fsum(r.sxy for r in clusters) / sxx
    variance = math.fsum((r.sxy - beta * r.sxx) ** 2 for r in clusters) / sxx**2
    g = len(clusters)
    se = math.sqrt(variance * g / (g - 1) * (n - 1) / (n - k - 1))
    return dict(specification=label, n_identifying_records=n, fixed_effect_strata=k,
                lender_clusters=g, fallout_pp_per_spread_pp=100 * beta,
                lender_clustered_se_pp=100 * se, ci95_lower_pp=100 * (beta - 1.96 * se),
                ci95_upper_pp=100 * (beta + 1.96 * se))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    spark = build_session("HMDA price sensitivity")
    try:
        spark.sparkContext.setLogLevel("ERROR")
        bands = case_mix_bands()
        base = features(spark.read.parquet(args.staged)).where(
            F.col("county_code").rlike("^[0-9]{5}$") & (F.col("county_code") != "99999")
            & F.col("lei").isNotNull()
        ).withColumns(bands).select(*MARKET_KEYS, *bands, "spread", "fallout").persist()
        valid = base.where("spread is not null")
        strata = [*MARKET_KEYS, *bands]
        coverage = base.groupBy(*MARKET_KEYS).agg(
            F.count_if(F.expr("fallout = 0")).alias("n0"), F.count_if(F.expr("fallout = 1")).alias("n1"),
            F.count_if(F.expr("fallout = 0 and spread is not null")).alias("v0"),
            F.count_if(F.expr("fallout = 1 and spread is not null")).alias("v1"),
        ).where("v0 >= 10 and v1 >= 10").where(
            "v0/n0 >= 0.5 and v1/n1 >= 0.5 and abs(v0/n0 - v1/n1) <= 0.20"
        ).select(*MARKET_KEYS)
        # Exact discrete percentiles so the Pandas check can reproduce the trimmed sample.
        # The archived run used approxQuantile(relative error 1e-4); both are recorded.
        approx_lower, approx_upper = valid.approxQuantile("spread", [0.01, 0.99], 0.0001)
        lower, upper = valid.selectExpr(
            "percentile_disc(0.01) WITHIN GROUP (ORDER BY spread)",
            "percentile_disc(0.99) WITHIN GROUP (ORDER BY spread)").first()
        # Records that identify the case-mix model: strata with within-stratum price variation.
        identifying = (valid.groupBy(*strata).agg(F.regr_sxx("fallout", "spread").alias("sxx"))
                       .where("sxx > 0").select(*strata))
        same_sample = valid.join(identifying, strata, "left_semi")
        rows = [
            fit(valid, MARKET_KEYS, "lender_county_product"),
            fit(same_sample, MARKET_KEYS, "lender_county_product_on_case_mix_sample"),
            fit(valid, strata, "lender_county_product_case_mix"),
            fit(valid.join(coverage, MARKET_KEYS, "left_semi"), strata, "balanced_price_coverage_case_mix"),
            fit(valid.where(F.col("spread").between(lower, upper)), strata, "central_98_percent_price_case_mix"),
        ]
        rows[-1].update(trim_lower_spread_pp=lower, trim_upper_spread_pp=upper,
                        approx_lower_spread_pp=approx_lower, approx_upper_spread_pp=approx_upper)
        for row in rows[:-1]:
            row.update(trim_lower_spread_pp="", trim_upper_spread_pp="",
                       approx_lower_spread_pp="", approx_upper_spread_pp="")
        rows_csv(args.out / "price_model.csv", rows)
        print("Saved price_model.csv", flush=True)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

"""Spark implementation of the diagnostic layer that hmda_results.py computes in Pandas.

Covers the matched within-segment price comparison, case-mix standardization against
local peers, the screening rules, and the Spearman association, computed from the
staged records with Spark DataFrame operations only.
"""
import argparse
import json
from pathlib import Path

from pyspark.sql import Window, functions as F

from hmda_results import (MAX_PRICE_COVERAGE_GAP, MIN_MATCHED_PRICES, MIN_ORIGINATED_PRICES,
                          MIN_PRICE_COVERAGE, MIN_SUPPORT, PRIMARY_VOLUME)
from hmda_spark import build_session, case_mix_bands, export, features, number

GROUP = ["geo", "lei", "market", "loan_type"]
MARKET = ["geo", "market", "loan_type"]
BANDS = ["amount_band", "income_band", "dti_band", "cltv_band"]


def safe_div(num, den):
    return F.when(den != 0, num / den)


def geo_records(cohort):
    fields = ["lei", "loan_type", "action_taken", "spread", "fallout", "interest",
              "loan_amount", "income", "debt_to_income_ratio", "combined_loan_to_value_ratio"]
    geo = cohort.withColumn("interest", number("interest_rate")).unpivot(
        fields, [F.col("county_code").alias("county"), F.col("derived_msa_md").alias("msa")],
        "geo", "market")
    market_ok = F.col("market").rlike(r"^[0-9]{5}$") & (F.col("market") != "99999")
    return geo.where(market_ok & F.col("lei").isNotNull() & F.col("loan_type").isNotNull())


def segments(geo, extra=(), medians=True):
    """Per-outcome segment counts; medians are skipped when only the peer layer is needed."""
    keys = [*GROUP, *extra, "action_taken"]
    median = (lambda c: F.expr(f"percentile({c}, 0.5)")) if medians else (lambda c: F.lit(None).cast("double"))
    return geo.groupBy(*keys).agg(
        F.count("*").alias("n"), F.count("spread").alias("n_price_valid"),
        F.sum("spread").alias("spread_sum"),
        median("spread").alias("median"),
        F.count("interest").alias("n_rate_valid"),
        median("interest").alias("rate_median"),
    )


def by_action(seg, keys):
    cols = ["n", "n_price_valid", "spread_sum", "median", "n_rate_valid", "rate_median"]
    one = seg.where("action_taken = '1'").select(*keys, *(F.col(c).alias(f"{c}_1") for c in cols))
    two = seg.where("action_taken = '2'").select(*keys, *(F.col(c).alias(f"{c}_2") for c in cols))
    both = one.join(two, keys, "full")
    for c in ("n", "n_price_valid", "n_rate_valid"):
        both = both.fillna(0, subset=[f"{c}_1", f"{c}_2"])
    return both.withColumns({
        "coverage_1": safe_div(F.col("n_price_valid_1"), F.col("n_1")),
        "coverage_2": safe_div(F.col("n_price_valid_2"), F.col("n_2")),
        "weight": F.least("n_price_valid_1", "n_price_valid_2"),
        "gap": F.col("median_2") - F.col("median_1"),
        "rate_weight": F.least("n_rate_valid_1", "n_rate_valid_2"),
        "rate_gap": F.col("rate_median_2") - F.col("rate_median_1"),
    })


def matched_summary(prices, extra=()):
    min_count = F.col("weight") >= MIN_MATCHED_PRICES
    balanced = (F.col("coverage_1") >= MIN_PRICE_COVERAGE) & (F.col("coverage_2") >= MIN_PRICE_COVERAGE) \
        & (F.abs(F.col("coverage_1") - F.col("coverage_2")) <= MAX_PRICE_COVERAGE_GAP)
    both80 = (F.col("coverage_1") >= 0.80) & (F.col("coverage_2") >= 0.80)
    scenarios = {
        "minimum_10_prices_each": min_count,
        "balanced_coverage": min_count & F.coalesce(balanced, F.lit(False)),
        "both_coverage_80pct": min_count & F.coalesce(both80, F.lit(False)),
    }
    totals = prices.groupBy("geo", *extra).agg(F.sum("n_price_valid_1").alias("all_valid_1"),
                                               F.sum("n_price_valid_2").alias("all_valid_2"))
    parts = []
    for name, rule in scenarios.items():
        m = prices.where(rule & F.col("gap").isNotNull())
        parts.append(m.groupBy("geo", *extra).agg(
            F.count("*").alias("n_strata"),
            F.sum("n_price_valid_1").alias("n_originated_prices"),
            F.sum("n_price_valid_2").alias("n_fallout_prices"),
            (F.sum(F.col("weight") * F.col("gap")) / F.sum("weight")).alias("weighted_mean_of_median_gaps_pp"),
            F.expr("percentile(gap, 0.5)").alias("median_of_stratum_gaps_pp"),
            F.avg((F.col("gap") > 0).cast("double")).alias("share_strata_positive"),
            # Interest-rate medians on the same matched segments, weighted by the price weights
            (F.sum(F.col("weight") * F.col("rate_gap")) / F.sum(F.when(F.col("rate_gap").isNotNull(), F.col("weight"))))
            .alias("rate_gap_on_same_segments_pp"),
        ).withColumn("scenario", F.lit(name)))
    rate = prices.where((F.col("rate_weight") >= MIN_MATCHED_PRICES) & F.col("rate_gap").isNotNull())
    parts.append(rate.groupBy("geo", *extra).agg(
        F.count("*").alias("n_strata"),
        F.sum("n_rate_valid_1").alias("n_originated_prices"),
        F.sum("n_rate_valid_2").alias("n_fallout_prices"),
        (F.sum(F.col("rate_weight") * F.col("rate_gap")) / F.sum("rate_weight")).alias("weighted_mean_of_median_gaps_pp"),
        F.expr("percentile(rate_gap, 0.5)").alias("median_of_stratum_gaps_pp"),
        F.avg((F.col("rate_gap") > 0).cast("double")).alias("share_strata_positive"),
        F.lit(None).cast("double").alias("rate_gap_on_same_segments_pp"),
    ).withColumn("scenario", F.lit("interest_rate_minimum_10_each")))
    result = parts[0]
    for part in parts[1:]:
        result = result.unionByName(part)
    return result.join(totals, ["geo", *extra], "left").withColumns({
        "share_valid_originated_prices_matched": F.col("n_originated_prices") / F.col("all_valid_1"),
        "share_valid_fallout_prices_matched": F.col("n_fallout_prices") / F.col("all_valid_2"),
    })


def standardize(cells, n_col, out_col, suffix=""):
    w = Window.partitionBy(*MARKET, *BANDS)
    c = cells.select(*GROUP, *BANDS, F.col(n_col).alias("n"), F.col(out_col).alias("o"))
    c = c.withColumns({"ref_n": F.sum("n").over(w), "ref_o": F.sum("o").over(w)})
    c = c.withColumns({"peer_n": F.col("ref_n") - F.col("n"), "peer_o": F.col("ref_o") - F.col("o")})
    support = (F.col("n") > 0) & (F.col("peer_n") > 0)
    weight = F.when(support, F.col("ref_n")).otherwise(F.lit(0))
    c = c.withColumns({
        "support": support,
        "weight": weight,
        "own_w": F.when(support, weight * (F.col("o") / F.col("n"))).otherwise(F.lit(0.0)),
        "peer_w": F.when(support, weight * (F.col("peer_o") / F.col("peer_n"))).otherwise(F.lit(0.0)),
    })
    r = c.groupBy(*GROUP).agg(
        F.sum("n").alias("n"), F.sum("o").alias("n_fallout"),
        F.sum(F.when(F.col("support"), F.col("n")).otherwise(0)).alias("n_own_supported"),
        F.sum(F.when(F.col("support"), F.col("peer_n")).otherwise(0)).alias("n_peer_supported"),
        F.sum("weight").alias("n_reference_supported"),
        F.sum("own_w").alias("own_weighted"), F.sum("peer_w").alias("peer_weighted"),
    )
    wm = Window.partitionBy(*MARKET)
    r = r.withColumn("reference_total", F.sum("n").over(wm))
    r = r.withColumns({
        "n_peer": F.col("reference_total") - F.col("n"),
        "reference_support_share": safe_div(F.col("n_reference_supported"), F.col("reference_total")),
        "adjusted_rate": safe_div(F.col("own_weighted"), F.col("n_reference_supported")),
        "peer_adjusted_rate": safe_div(F.col("peer_weighted"), F.col("n_reference_supported")),
        "own_total": F.sum("n_fallout").over(wm),
    })
    r = r.withColumns({
        "adjusted_gap_pp": 100 * (F.col("adjusted_rate") - F.col("peer_adjusted_rate")),
        "peer_crude_rate": safe_div(F.col("own_total") - F.col("n_fallout"), F.col("n_peer")),
    })
    r = r.withColumn("crude_gap_pp", 100 * (safe_div(F.col("n_fallout"), F.col("n")) - F.col("peer_crude_rate")))
    keep = ["n", "n_peer", "reference_support_share", "adjusted_gap_pp", "crude_gap_pp"]
    return r.select(*GROUP, *(F.col(k).alias(k + suffix) for k in keep))


def originated_prices(prices):
    wm = Window.partitionBy(*MARKET)
    p = prices.select(*GROUP, F.col("n_1").alias("n_originated"),
                      F.col("n_price_valid_1").alias("n_price_valid_originated"),
                      F.coalesce(F.col("spread_sum_1"), F.lit(0.0)).alias("spread_sum"),
                      "n_price_valid_2", "coverage_1", "coverage_2", "weight")
    # Peer quantities are defined only for lenders with originated records in the market,
    # as in hmda_results.price_diagnostics (which builds them from action 1 rows).
    has_originations = F.col("n_originated") > 0
    p = p.withColumns({
        "spread_mean_originated": safe_div(F.col("spread_sum"), F.col("n_price_valid_originated")),
        "peer_n_originated": F.when(has_originations, F.sum("n_originated").over(wm) - F.col("n_originated")),
        "peer_n_price_valid": F.when(has_originations,
                                     F.sum("n_price_valid_originated").over(wm) - F.col("n_price_valid_originated")),
        "peer_spread_sum": F.when(has_originations, F.sum("spread_sum").over(wm) - F.col("spread_sum")),
    })
    return p.withColumns({
        "peer_originated_coverage": safe_div(F.col("peer_n_price_valid"), F.col("peer_n_originated")),
        "peer_spread_mean": safe_div(F.col("peer_spread_sum"), F.col("peer_n_price_valid")),
    }).withColumn("relative_spread_pp", F.col("spread_mean_originated") - F.col("peer_spread_mean"))


def eligible(volume=PRIMARY_VOLUME, support=MIN_SUPPORT, coverage=MIN_PRICE_COVERAGE, suffix=""):
    rule = ((F.col("n" + suffix) >= volume) & (F.col("n_peer" + suffix) >= volume)
            & (F.col("reference_support_share" + suffix) >= support)
            & (F.col("n_price_valid_originated") >= MIN_ORIGINATED_PRICES)
            & (F.col("peer_n_price_valid") >= MIN_ORIGINATED_PRICES)
            & (F.col("coverage_1") >= coverage) & (F.col("peer_originated_coverage") >= coverage))
    return F.coalesce(rule, F.lit(False))


def positive(column, threshold=None):
    value = F.bround(F.col(column), 10)
    return F.coalesce(value > 0 if threshold is None else value >= threshold, F.lit(False))


def spearman(frame, x, y):
    """Average-rank Spearman correlation within each geography (ties share their mean rank)."""
    ranked = frame
    for name in (x, y):
        order = Window.partitionBy("geo").orderBy(name)
        ties = Window.partitionBy("geo", name)
        ranked = ranked.withColumn(f"r_{name}", F.rank().over(order) + (F.count("*").over(ties) - 1) / 2)
    return {r["geo"]: r["rho"] for r in ranked.groupBy("geo").agg(F.corr(f"r_{x}", f"r_{y}").alias("rho")).collect()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", choices=["all", "matched", "peer"], default="all",
                        help="matched price comparison, peer standardization, or both")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spark = build_session("HMDA diagnostics")
    try:
        spark.sparkContext.setLogLevel("ERROR")
        geo = geo_records(features(spark.read.parquet(str(args.staged)))).withColumns(case_mix_bands())
        geo = geo.cache()
        prices = by_action(segments(geo, medians=args.part != "peer"), GROUP).cache()
        if args.part in ("all", "matched"):
            export(matched_summary(prices), args.out / "matched_spark.csv", ("geo", "scenario"))
            export(matched_summary(prices, ["loan_type"]), args.out / "matched_by_product_spark.csv",
                   ("geo", "loan_type", "scenario"))
            band_prices = by_action(segments(geo, ["amount_band"]), [*GROUP, "amount_band"])
            export(matched_summary(band_prices, ["amount_band"]), args.out / "matched_by_amount_band_spark.csv",
                   ("geo", "amount_band", "scenario"))
        if args.part == "matched":
            return

        cells = geo.groupBy(*GROUP, *BANDS).agg(
            F.count("*").alias("n"), F.sum("fallout").alias("n_fallout"),
            F.count("spread").alias("n_price_valid"),
            F.count_if((F.col("fallout") == 1) & F.col("spread").isNotNull()).alias("n_fallout_price_valid"),
        ).cache()
        adjusted = standardize(cells, "n", "n_fallout").join(
            standardize(cells, "n_price_valid", "n_fallout_price_valid", "_priced"), GROUP, "inner")
        diag = adjusted.join(originated_prices(prices), GROUP, "left")
        diag = diag.withColumns({
            "primary_eligible": eligible(),
            "priced_eligible": eligible(suffix="_priced"),
        }).withColumns({
            "primary_candidate": F.col("primary_eligible") & positive("relative_spread_pp") & positive("adjusted_gap_pp"),
            "material_candidate": F.col("primary_eligible") & positive("relative_spread_pp", 0.10)
            & positive("adjusted_gap_pp", 1.00),
            "priced_candidate": F.col("priced_eligible") & positive("relative_spread_pp")
            & positive("adjusted_gap_pp_priced"),
        }).cache()
        columns = [*GROUP, "n", "n_peer", "reference_support_share", "adjusted_gap_pp", "crude_gap_pp",
                   "n_priced", "n_peer_priced", "reference_support_share_priced", "adjusted_gap_pp_priced",
                   "relative_spread_pp", "n_price_valid_originated", "peer_n_price_valid",
                   "primary_eligible", "primary_candidate", "material_candidate", "priced_eligible",
                   "priced_candidate"]
        export(diag.where((F.col("n") >= 50) & (F.col("n_peer") >= 50)).select(*columns),
               args.out / "diagnostics_spark.csv", GROUP)
        counts = {r["geo"]: r.asDict() for r in diag.groupBy("geo").agg(
            F.count("*").alias("segments"),
            F.sum(F.col("primary_eligible").cast("int")).alias("primary_eligible"),
            F.sum(F.col("primary_candidate").cast("int")).alias("primary_candidate"),
            F.sum(F.col("material_candidate").cast("int")).alias("material_candidate"),
            F.sum(F.col("priced_eligible").cast("int")).alias("priced_eligible"),
            F.sum(F.col("priced_candidate").cast("int")).alias("priced_candidate"),
        ).collect()}
        rho = spearman(diag.where("primary_eligible"), "relative_spread_pp", "adjusted_gap_pp")
        rho_priced = spearman(diag.where("priced_eligible"), "relative_spread_pp", "adjusted_gap_pp_priced")
        for geo_name in counts:
            counts[geo_name]["spearman_primary"] = rho.get(geo_name)
            counts[geo_name]["spearman_price_available_outcome"] = rho_priced.get(geo_name)
        (args.out / "diagnostics_spark.json").write_text(json.dumps(counts, indent=2))
        print(json.dumps(counts, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Spark side of the predictive extension (predictive_plan.md).

Reads the raw CSV with Spark, applies the eight cohort filters, assigns folds, computes the
within-segment cross-products with DataFrame aggregations, solves for the coefficients on the
driver, and writes metrics, coefficients and segment medians. Phase "gbt" adds the boosted-tree
sensitivity model (M3), which runs in Spark only.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import functions as F

import hmda_predict_defs as P
from hmda_defs import BAND_EDGES
from hmda_spark import band, build_session, filters, number

SEG = ["lei", "county_code", "loan_type"]
MKT = ["county_code", "loan_type"]
BATCH = 200  # aggregate expressions per pass


# ------------------------------------------------------------------------------ extraction
def extract(spark, raw, destination):
    source = spark.read.option("header", True).option("mode", "FAILFAST").csv(str(raw))
    wanted = list(dict.fromkeys(P.FILTER_COLUMNS + P.EXTRACT))
    trimmed = source.select(*(F.trim(F.col(c)).alias(c) for c in wanted))
    keep = F.lit(True)
    for _, condition in filters():
        keep = keep & condition
    key = F.concat_ws("|", *(F.coalesce(F.col(c), F.lit("")) for c in P.EXTRACT))
    out = trimmed.where(keep).select(*P.EXTRACT).withColumn(
        "fold", F.pmod(F.crc32(key.cast("binary")), F.lit(P.FOLDS)).cast("int"))
    out.write.mode("overwrite").parquet(str(destination))
    return spark.read.parquet(str(destination)).count()


# ------------------------------------------------------------------------------- features
def text(c):
    return F.coalesce(F.col(c), F.lit(""))


def features(ext):
    dti = text("debt_to_income_ratio")
    dti_num = (F.when(dti == "<20%", 19.0).when(dti == "20%-<30%", 25.0).when(dti == "30%-<36%", 33.0)
               .when(dti == "50%-60%", 55.0).when(dti == ">60%", 61.0).otherwise(number("debt_to_income_ratio")))
    term = number("loan_term")
    intro = number("intro_rate_period")
    flags = [text(c) for c in P.NON_AMORTIZING]
    yes = flags[0] == "1"
    no = flags[0] == "2"
    for f in flags[1:]:
        yes, no = yes | (f == "1"), no & (f == "2")
    conf = text("conforming_loan_limit")
    county = text("county_code")
    valid = county.rlike(r"^[0-9]{5}$") & (county != "99999") & (text("lei") != "") & (text("loan_type") != "")
    df = ext.where(valid).withColumns({
        "loan_amount_num": number("loan_amount"), "income_num": number("income"), "dti_num": dti_num,
        "cltv_num": number("combined_loan_to_value_ratio"), "spread": number("rate_spread"),
        "action": F.col("action_taken").cast("int"),
        "term_cat": F.when(term.isNull(), "Unknown").when(term == 360, "360").when(term == 180, "180")
        .otherwise("other"),
        "arm_cat": F.when(text("intro_rate_period") == "NA", "fixed").when(intro.isNull(), "Unknown")
        .when(intro > 0, "ARM").otherwise("fixed"),
        "conforming_cat": F.when(conf == "C", "C").when(conf == "NC", "NC").otherwise("other"),
        "nonamort_cat": F.when(yes, "yes").when(no, "no").otherwise("Unknown"),
    })
    sources = {"amount_bin": "loan_amount_num", "income_bin": "income_num", "dti_bin": "dti_num",
               "cltv_bin": "cltv_num"}
    df = df.withColumns({name: band(F.col(sources[name]), edges) for name, (_, edges, _) in P.CORE.items()})
    df = df.withColumn("amount_band", band(F.col("loan_amount_num"), BAND_EDGES["amount_band"][1]))
    keep = [*SEG, "action", "fold", "spread", "amount_band", *P.CORE, *P.TERMS, *P.GBT_NUMERIC]
    return df.select(*keep)


def reporting_audit(df):
    exprs = []
    for name in P.TERMS:
        missing = (F.col(name) == "other") if name == "conforming_cat" else (F.col(name) == "Unknown")
        exprs.append(F.avg(missing.cast("double")).alias(name))
    shares = {r["action"]: r for r in df.groupBy("action").agg(*exprs).collect()}
    rows, used = [], []
    for name in P.TERMS:
        s1, s2 = shares[1][name], shares[2][name]
        ok = abs(s2 - s1) <= P.MAX_REPORTING_GAP
        rows.append(dict(feature=name, missing_share_originated=s1, missing_share_not_accepted=s2,
                         gap=abs(s2 - s1), used=ok))
        if ok:
            used.append(name)
    return pd.DataFrame(rows), used


def trim_bounds(df):
    r = df.where("action = 1 AND spread IS NOT NULL").selectExpr(
        f"percentile_disc({P.TRIM[0]}) WITHIN GROUP (ORDER BY spread) AS lo",
        f"percentile_disc({P.TRIM[1]}) WITHIN GROUP (ORDER BY spread) AS hi").first()
    return float(r["lo"]), float(r["hi"])


# -------------------------------------------------------------------------------- fitting
def dummies(columns):
    out = []
    for col in columns:
        name, cat = col.split("=", 1)
        out.append(F.when(F.col(name) == cat, 1.0).otherwise(0.0))
    return out


def batched_agg(frame, group, named_exprs):
    """Run many aggregate expressions in passes of BATCH and merge the results by group."""
    result = None
    items = list(named_exprs.items())
    for start in range(0, len(items), BATCH):
        part = frame.groupBy(*group).agg(*(e.alias(n) for n, e in items[start:start + BATCH])).toPandas()
        result = part if result is None else result.merge(part, on=group, how="outer")
    return result.set_index(group[0]) if len(group) == 1 else result


def gram_exprs(d, y, weight=None):
    p = len(d)
    w = (lambda e: e) if weight is None else (lambda e: e * weight)
    exprs = {f"g_{i}_{j}": F.sum(w(d[i] * d[j])) for i in range(p) for j in range(i, p)}
    exprs.update({f"c_{i}": F.sum(w(d[i] * y)) for i in range(p)})
    return exprs


def unpack(row, p):
    g = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            g[i, j] = g[j, i] = row[f"g_{i}_{j}"]
    return g, np.array([row[f"c_{i}"] for i in range(p)], dtype=np.float64)


def beta_expr(betas, j):
    e = F.lit(float(betas[0][j]))
    for k in range(1, P.FOLDS):
        e = F.when(F.col("fold") == k, float(betas[k][j])).otherwise(e)
    return F.when(F.col("fold") == 0, float(betas[0][j])).otherwise(e)


def fit_variant(spark, df, columns, train_cond):
    p = len(columns)
    base = df.withColumn("train", train_cond).persist(StorageLevel.MEMORY_AND_DISK)
    d = dummies(columns)
    y = F.col("spread")
    tr = base.where("train")
    # 1. Raw cross-products per fold
    per_fold = batched_agg(tr, ["fold"], gram_exprs(d, y))
    g_fold, c_fold = {}, {}
    for k in range(P.FOLDS):
        g_fold[k], c_fold[k] = unpack(per_fold.loc[k], p)
    g_tot, c_tot = sum(g_fold.values()), sum(c_fold.values())
    # 2. Segment sums per fold, totals, and training sums for each held-out fold k
    sums = {"n": F.count("*").cast("double"), "sy": F.sum(y), **{f"sx_{j}": F.sum(d[j]) for j in range(p)}}
    seg_fold = tr.groupBy(*SEG, "fold").agg(*(e.alias(n) for n, e in sums.items())).persist()
    seg_tot = seg_fold.groupBy(*SEG).agg(*(F.sum(n).alias(n) for n in sums))
    ks = spark.range(P.FOLDS).select(F.col("id").cast("int").alias("k"))
    held = seg_fold.select(*SEG, F.col("fold").alias("k"), *(F.col(n).alias(f"{n}_k") for n in sums))
    seg_tr = (seg_tot.crossJoin(ks).join(held, [*SEG, "k"], "left")
              .select(*SEG, "k", *((F.col(n) - F.coalesce(F.col(f"{n}_k"), F.lit(0.0))).alias(n) for n in sums))
              .persist())
    pos = seg_tr.where("n > 0")
    sx = [F.col(f"sx_{j}") for j in range(p)]
    ss = {f"g_{i}_{j}": F.sum(sx[i] * sx[j] / F.col("n")) for i in range(p) for j in range(i, p)}
    ss.update({f"c_{i}": F.sum(sx[i] * F.col("sy") / F.col("n")) for i in range(p)})
    ss_k = batched_agg(pos, ["k"], ss)
    betas = np.zeros((P.FOLDS, p))
    for k in range(P.FOLDS):
        s_g, s_c = unpack(ss_k.loc[k], p)
        betas[k] = P.solve((g_tot - g_fold[k]) - s_g, (c_tot - c_fold[k]) - s_c)
    # 3. Shrunken segment effects for M1 (beta = 0) and M2, by held-out fold k
    def beta_k(j):
        e = F.lit(float(betas[0][j]))
        for k in range(1, P.FOLDS):
            e = F.when(F.col("k") == k, float(betas[k][j])).otherwise(e)
        return e
    xb_tr = sum((F.col(f"sx_{j}") * beta_k(j) for j in range(p)), F.lit(0.0)) if p else F.lit(0.0)
    effects = seg_tr.select(*SEG, "k", "n", F.col("sy").alias("a1"), (F.col("sy") - xb_tr).alias("a2"))
    mkt = effects.groupBy(*MKT, "k").agg(F.sum("a1").alias("m1"), F.sum("a2").alias("m2"), F.sum("n").alias("nm"))
    prod = mkt.groupBy("loan_type", "k").agg((F.sum("m1") / F.sum("nm")).alias("p1"),
                                              (F.sum("m2") / F.sum("nm")).alias("p2"))
    mkt = mkt.join(prod, ["loan_type", "k"]).select(
        *MKT, "k", "p1", "p2",
        ((F.col("m1") + P.SHRINK * F.col("p1")) / (F.col("nm") + P.SHRINK)).alias("am1"),
        ((F.col("m2") + P.SHRINK * F.col("p2")) / (F.col("nm") + P.SHRINK)).alias("am2"))
    seg_alpha = effects.join(mkt.select(*MKT, "k", "am1", "am2"), [*MKT, "k"]).select(
        *SEG, "k",
        ((F.col("a1") + P.SHRINK * F.col("am1")) / (F.col("n") + P.SHRINK)).alias("as1"),
        ((F.col("a2") + P.SHRINK * F.col("am2")) / (F.col("n") + P.SHRINK)).alias("as2"))
    # 4. Out-of-fold predictions for every record
    xb = sum((dj * beta_expr(betas, j) for j, dj in enumerate(d)), F.lit(0.0)) if p else F.lit(0.0)
    rows = (base.withColumn("k", F.col("fold"))
            .join(seg_alpha, [*SEG, "k"], "left")
            .join(mkt.select(*MKT, "k", "am1", "am2"), [*MKT, "k"], "left")
            .join(prod.select("loan_type", "k", "p1", "p2"), ["loan_type", "k"], "left")
            .withColumn("xb", xb)
            .withColumns({"pred_M0": F.col("p1"),
                          "pred_M1": F.coalesce("as1", "am1", "p1"),
                          "pred_M2": F.coalesce("as2", "am2", "p2") + F.col("xb"),
                          "resid": F.col("spread") - F.col("xb")}))
    seg_n = seg_tot.select(*SEG, F.col("n").alias("seg_train_n"))
    rows = rows.join(seg_n, SEG, "left").fillna(0.0, subset=["seg_train_n"]).persist(StorageLevel.MEMORY_AND_DISK)
    return rows, betas, [base, seg_fold, seg_tr]


def metrics(variant, rows, models=("M0", "M1", "M2")):
    ev = rows.where("train")
    agg = [F.count("*").alias("n"), F.sum("spread").alias("sy"), F.sum(F.col("spread") ** 2).alias("syy")]
    for m in models:
        e = F.col("spread") - F.col(f"pred_{m}")
        agg += [F.sum(e * e).alias(f"sse_{m}"), F.sum(F.abs(e)).alias(f"sae_{m}")]
    r = ev.agg(*agg).first()
    n = r["n"]
    sst = r["syy"] - r["sy"] ** 2 / n
    out = []
    for m in models:
        sse = r[f"sse_{m}"]
        out.append(dict(variant=variant, model=m, n=n, sse=sse, sae=r[f"sae_{m}"],
                        rmse_bp=100 * np.sqrt(sse / n), mae_bp=100 * r[f"sae_{m}"] / n, r2=1 - sse / sst,
                        r2_vs_M0=1 - sse / r["sse_M0"] if "sse_M0" in r.asDict() else np.nan,
                        r2_within_vs_M1=1 - sse / r["sse_M1"] if m in ("M2", "M3") and "sse_M1" in r.asDict() else np.nan))
    return out


def segment_medians(rows, condition, variant, by_band=False, resid="resid"):
    keys = [*SEG] + (["amount_band"] if by_band else []) + ["action"]
    out = (rows.where(condition).groupBy(*keys).agg(
        F.count("*").alias("n_valid"), F.expr("percentile(spread, 0.5)").alias("median_spread"),
        F.expr(f"percentile({resid}, 0.5)").alias("median_resid")).toPandas())
    if not by_band:
        out.insert(3, "amount_band", "All")
    out.insert(0, "variant", variant)
    return out.rename(columns={"action": "action_taken"})


def support_condition(rows):
    thin = F.col("seg_train_n") < 20
    tr = rows.where("train")
    for name in P.CORE:
        counts = tr.groupBy("loan_type", name).agg(F.count("*").alias(f"c_{name}"))
        rows = rows.join(counts, ["loan_type", name], "left")
        thin = thin | (F.coalesce(F.col(f"c_{name}"), F.lit(0)) < 100)
    return rows, thin & (F.col("action") == 2)


# ------------------------------------------------------------------------------------ GBT
def gbt(spark, rows, terms):
    """M3: boosted trees on spread minus the M1 segment effect, cross-fitted by fold.
    The prediction for a record in fold k is its M1 effect (computed without fold k) plus the
    tree output of the model trained without fold k."""
    cats = [c for c in P.design_columns(terms) if c.split("=")[0] in P.TERMS]
    ind = {f"i_{j}": e for j, e in enumerate(dummies(cats))}
    lt = {f"lt_{v}": (F.col("loan_type") == v).cast("double") for v in ["1", "2", "3", "4"]}
    feats = [*P.GBT_NUMERIC, *ind, *lt]
    keep = [*SEG, "fold", "action", "spread", "train", "amount_band", "pred_M0", "pred_M1"]
    base = rows.select(*keep, *P.GBT_NUMERIC, *P.TERMS).withColumns({**ind, **lt})
    assembler = VectorAssembler(inputCols=feats, outputCol="features")
    seg_alpha, mkt_alpha, prod_alpha = m1_effect_tables(spark, rows)
    preds, checks = [], {}
    for k in range(P.FOLDS):
        t0 = perf_counter()
        train = (base.where(F.col("train") & (F.col("fold") != k)).withColumn("k", F.lit(k))
                 .join(seg_alpha, [*SEG, "k"], "left").join(mkt_alpha, [*MKT, "k"], "left")
                 .join(prod_alpha, ["loan_type", "k"], "left")
                 .withColumn("label", F.col("spread") - F.coalesce("as1", "am1", "p1")))
        train = assembler.transform(train).select("features", "label").repartition(32).persist()
        model = GBTRegressor(featuresCol="features", labelCol="label", **P.GBT).fit(train)
        test = assembler.transform(base.where(F.col("fold") == k))
        scored = model.transform(test)
        if k == 0:
            again = GBTRegressor(featuresCol="features", labelCol="label", **P.GBT).fit(train)
            twice = again.setPredictionCol("p2").transform(scored)
            checks["refit_fold0_max_abs_difference"] = float(
                twice.agg(F.max(F.abs(F.col("prediction") - F.col("p2")))).first()[0])
            checks["refit_fold0_trees_identical"] = (model.toDebugString.split("\n", 1)[1]
                                                       == again.toDebugString.split("\n", 1)[1])
        preds.append(scored.select(*keep, F.col("prediction").alias("f")))
        train.unpersist()
        print(f"GBT fold {k}: {perf_counter() - t0:.1f}s", flush=True)
    out = preds[0]
    for part in preds[1:]:
        out = out.unionByName(part)
    return out.withColumns({"pred_M3": F.col("pred_M1") + F.col("f"),
                            "resid_M3": F.col("spread") - F.col("f")}), checks


def m1_effect_tables(spark, rows):
    """Re-derive the M1 effect tables from the stored predictions: for a record in fold k,
    pred_M1 is the shrunken segment effect computed without fold k. GBT training rows (fold != k)
    need the effect computed without fold k, so the tables are rebuilt here from training sums."""
    tr = rows.where("train")
    seg_fold = tr.groupBy(*SEG, "fold").agg(F.count("*").cast("double").alias("n"), F.sum("spread").alias("sy"))
    seg_tot = seg_fold.groupBy(*SEG).agg(F.sum("n").alias("n"), F.sum("sy").alias("sy"))
    ks = spark.range(P.FOLDS).select(F.col("id").cast("int").alias("k"))
    held = seg_fold.select(*SEG, F.col("fold").alias("k"), F.col("n").alias("n_k"), F.col("sy").alias("sy_k"))
    seg_tr = (seg_tot.crossJoin(ks).join(held, [*SEG, "k"], "left")
              .select(*SEG, "k", (F.col("n") - F.coalesce("n_k", F.lit(0.0))).alias("n"),
                      (F.col("sy") - F.coalesce("sy_k", F.lit(0.0))).alias("a1")))
    mkt = seg_tr.groupBy(*MKT, "k").agg(F.sum("a1").alias("m1"), F.sum("n").alias("nm"))
    prod = mkt.groupBy("loan_type", "k").agg((F.sum("m1") / F.sum("nm")).alias("p1"))
    mkt = mkt.join(prod, ["loan_type", "k"]).select(
        *MKT, "k", ((F.col("m1") + P.SHRINK * F.col("p1")) / (F.col("nm") + P.SHRINK)).alias("am1"))
    seg = seg_tr.join(mkt, [*MKT, "k"]).select(
        *SEG, "k", ((F.col("a1") + P.SHRINK * F.col("am1")) / (F.col("n") + P.SHRINK)).alias("as1"))
    return seg.persist(), mkt.persist(), prod.persist()


# ----------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "extract", "fit", "gbt"], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    extract_path = args.out / "predict_extract_spark"
    meta = {"plan_sha256": P.plan_sha256()}
    print(f"Plan SHA-256 {meta['plan_sha256']}", flush=True)
    spark = build_session("HMDA 2024 prediction")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if args.phase in ("all", "extract"):
            t = perf_counter()
            meta["extract_rows"] = extract(spark, args.raw, extract_path)
            meta["extract_seconds"] = perf_counter() - t
            print(f"Extracted {meta['extract_rows']:,} rows in {meta['extract_seconds']:.1f}s", flush=True)
            (args.out / "predict_extract_meta_spark.json").write_text(json.dumps(meta, indent=2))
        if args.phase == "extract":
            return
        t = perf_counter()
        df = features(spark.read.parquet(str(extract_path))).persist(StorageLevel.MEMORY_AND_DISK)
        audit, terms = reporting_audit(df)
        lo, hi = trim_bounds(df)
        complete = F.lit(True)
        for name in P.CORE:
            complete = complete & (F.col(name) != "Unknown")
        in_range = F.col("spread").isNotNull() & (F.col("spread") >= lo) & (F.col("spread") <= hi)
        train = (F.col("action") == 1) & in_range
        fold_counts = df.groupBy("fold", "action").count().toPandas()
        meta.update(model_rows=df.count(), trim_lower=lo, trim_upper=hi, terms_used=terms,
                    complete_rows=df.where(complete).count(),
                    fold_counts={f"{r.fold}_{r.action}": int(r["count"]) for _, r in fold_counts.iterrows()})
        if args.phase in ("all", "fit"):
            audit.to_csv(args.out / "predict_reporting_spark.csv", index=False)
            metric_rows, coef_rows, seg_tables = [], [], []
            for variant, unknown_core in (("main", False), ("unknown_core", True)):
                sub = df.where(complete) if not unknown_core else df
                cols = P.design_columns(terms, unknown_core=unknown_core)
                rows, betas, cached = fit_variant(spark, sub, cols, train)
                metric_rows += metrics(variant, rows)
                for k in range(P.FOLDS):
                    coef_rows += [dict(variant=variant, fold=k, column=c, beta=b) for c, b in zip(cols, betas[k])]
                valid = F.col("spread").isNotNull()
                seg_tables.append(segment_medians(rows, valid, variant))
                if variant == "main":
                    seg_tables.append(segment_medians(rows, valid, "main_by_amount_band", by_band=True))
                    seg_tables.append(segment_medians(rows, valid & in_range, "central_98"))
                    flagged, thin = support_condition(rows)
                    seg_tables.append(segment_medians(flagged, valid & ~thin, "support"))
                    meta.update(main_rows=rows.count(), main_train_rows=rows.where("train").count(),
                                support_dropped=flagged.where(valid & thin).count(),
                                main_valid_not_accepted=rows.where(valid & (F.col("action") == 2)).count())
                    rows.write.mode("overwrite").parquet(str(args.out / "predict_rows_main_spark"))
                rows.unpersist()
                for c in cached:
                    c.unpersist()
            pd.DataFrame(metric_rows).to_csv(args.out / "predict_metrics_spark.csv", index=False)
            pd.DataFrame(coef_rows).to_csv(args.out / "predict_coef_spark.csv", index=False)
            pd.concat(seg_tables, ignore_index=True).to_parquet(args.out / "predict_segments_spark.parquet", index=False)
            meta["fit_seconds"] = perf_counter() - t
            (args.out / "predict_meta_spark.json").write_text(json.dumps(meta, indent=2))
            print(pd.DataFrame(metric_rows).to_string(), flush=True)
        if args.phase in ("all", "gbt"):
            t = perf_counter()
            rows = spark.read.parquet(str(args.out / "predict_rows_main_spark")).persist(StorageLevel.MEMORY_AND_DISK)
            gbt_rows, checks = gbt(spark, rows, terms)
            gbt_rows = gbt_rows.persist(StorageLevel.MEMORY_AND_DISK)
            met = metrics("main", gbt_rows, models=("M0", "M1", "M3"))
            pd.DataFrame(met).to_csv(args.out / "predict_metrics_gbt_spark.csv", index=False)
            seg = segment_medians(gbt_rows, F.col("spread").isNotNull(), "gbt", resid="resid_M3")
            seg.to_parquet(args.out / "predict_segments_gbt_spark.parquet", index=False)
            checks.update(gbt_seconds=perf_counter() - t, plan_sha256=meta["plan_sha256"], settings=P.GBT)
            (args.out / "predict_gbt_meta_spark.json").write_text(json.dumps(checks, indent=2))
            print(pd.DataFrame(met).to_string(), json.dumps(checks, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

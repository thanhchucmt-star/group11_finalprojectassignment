"""Combine the Spark and pandas outputs of the predictive extension: engine agreement,
within-segment raw and residual gaps with segment bootstrap intervals, and the pre-specified
decision rule (predictive_plan.md, sections 7 and 8)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hmda_predict_defs as P

KEYS = ["variant", "lei", "county_code", "loan_type", "amount_band", "action_taken"]
RTOL, ATOL = 1e-9, 1e-10


def matched(seg):
    """Segments with at least MIN_MATCHED valid records per outcome; one row per segment."""
    seg = seg.copy()
    seg["action_taken"] = seg["action_taken"].astype(int)
    keys = [k for k in KEYS if k != "action_taken"]
    one = seg[seg.action_taken.eq(1)].set_index(keys)
    two = seg[seg.action_taken.eq(2)].set_index(keys)
    both = one.join(two, how="inner", lsuffix="_1", rsuffix="_2").reset_index()
    both = both[both.n_valid_1.ge(P.MIN_MATCHED) & both.n_valid_2.ge(P.MIN_MATCHED)].copy()
    both["weight"] = both[["n_valid_1", "n_valid_2"]].min(axis=1)
    both["raw_gap"] = both.median_spread_2 - both.median_spread_1
    both["resid_gap"] = both.median_resid_2 - both.median_resid_1
    # Fixed order so the seeded bootstrap draws the same segments in both engines
    return both.sort_values(keys, kind="mergesort").reset_index(drop=True)


def summarize(both, rng_seed=P.SEED):
    rows = []
    for (variant, amount_band), g in both.groupby(["variant", "amount_band"], sort=True):
        w, raw, res = g.weight.to_numpy(float), g.raw_gap.to_numpy(), g.resid_gap.to_numpy()
        point_raw, point_res = np.average(raw, weights=w), np.average(res, weights=w)
        rng = np.random.default_rng(rng_seed)
        idx = rng.integers(0, len(g), size=(P.BOOT, len(g)))
        ww = w[idx]
        b_raw = (ww * raw[idx]).sum(1) / ww.sum(1)
        b_res = (ww * res[idx]).sum(1) / ww.sum(1)
        b_share = 1 - b_res / b_raw
        q = lambda v: np.percentile(v, [2.5, 97.5])
        rows.append(dict(
            variant=variant, amount_band=amount_band, n_segments=len(g),
            n_originated=int(g.n_valid_1.sum()), n_not_accepted=int(g.n_valid_2.sum()),
            raw_gap_bp=100 * point_raw, raw_lo_bp=100 * q(b_raw)[0], raw_hi_bp=100 * q(b_raw)[1],
            resid_gap_bp=100 * point_res, resid_lo_bp=100 * q(b_res)[0], resid_hi_bp=100 * q(b_res)[1],
            share_explained=1 - point_res / point_raw, share_lo=q(b_share)[0], share_hi=q(b_share)[1],
            share_segments_resid_positive=float((res > 0).mean())))
    return pd.DataFrame(rows)


def decide(row):
    lo, hi, share = row["resid_lo_bp"], row["resid_hi_bp"], row["share_explained"]
    if lo > 0 or hi < 0:
        return "persists" if share < 0.5 else "partly_explained"
    return "not_distinguishable"


def compare(a, b, name, keys, exact=(), close=()):
    m = a.merge(b, on=keys, how="outer", suffixes=("_pandas", "_spark"), indicator=True)
    out = [dict(output=name, quantity="rows matched on keys", pandas=int((m._merge != "right_only").sum()),
                spark=int((m._merge != "left_only").sum()), agree=bool((m._merge == "both").all()),
                max_abs_difference=0.0)]
    m = m[m._merge == "both"]
    for c in exact:
        x, y = m[f"{c}_pandas"].astype(float), m[f"{c}_spark"].astype(float)
        out.append(dict(output=name, quantity=c, pandas=float(x.sum()), spark=float(y.sum()),
                        agree=bool((x == y).all()), max_abs_difference=float(np.abs(x - y).max())))
    for c in close:
        x, y = m[f"{c}_pandas"].to_numpy(float), m[f"{c}_spark"].to_numpy(float)
        ok = np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True)
        out.append(dict(output=name, quantity=c, pandas=float(np.nansum(x)), spark=float(np.nansum(y)),
                        agree=bool(ok.all()), max_abs_difference=float(np.nanmax(np.abs(x - y))) if len(x) else 0.0))
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spark", type=Path, required=True)
    cli.add_argument("--pandas", type=Path, required=True)
    cli.add_argument("--tables", type=Path, required=True, help="main-analysis tables (matched_spark.csv)")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seg_s = pd.read_parquet(args.spark / "predict_segments_spark.parquet")
    seg_p = pd.read_parquet(args.pandas / "predict_segments_pandas.parquet")
    for s in (seg_s, seg_p):
        s["action_taken"] = s["action_taken"].astype(int)
    gbt_path = args.spark / "predict_segments_gbt_spark.parquet"
    seg_all = pd.concat([seg_s, pd.read_parquet(gbt_path)] if gbt_path.exists() else [seg_s], ignore_index=True)
    seg_all["action_taken"] = seg_all["action_taken"].astype(int)

    summary = summarize(matched(seg_all))
    summary_p = summarize(matched(seg_p))
    original = pd.read_csv(args.tables / "matched_spark.csv")
    original = original[(original.geo == "county") & (original.scenario == "minimum_10_prices_each")]
    original_bp = 100 * float(original["weighted_mean_of_median_gaps_pp"].iloc[0])
    main = summary[(summary.variant == "main") & (summary.amount_band == "All")].iloc[0]
    decision = dict(rule=decide(main), original_gap_bp=original_bp, **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                                                         for k, v in main.items()})
    full = summary[(summary.variant == "unknown_core") & (summary.amount_band == "All")]
    decision["unknown_core_raw_gap_bp"] = float(full.raw_gap_bp.iloc[0]) if len(full) else None
    decision["unknown_core_reproduces_original"] = bool(len(full) and abs(full.raw_gap_bp.iloc[0] - original_bp) < 1e-9)

    met_s = pd.read_csv(args.spark / "predict_metrics_spark.csv")
    met_p = pd.read_csv(args.pandas / "predict_metrics_pandas.csv")
    met_g = args.spark / "predict_metrics_gbt_spark.csv"
    metrics = pd.concat([met_s, pd.read_csv(met_g).query("model == 'M3'")] if met_g.exists() else [met_s])
    coef_s = pd.read_csv(args.spark / "predict_coef_spark.csv")
    coef_p = pd.read_csv(args.pandas / "predict_coef_pandas.csv")
    meta_s = json.loads((args.spark / "predict_meta_spark.json").read_text())
    meta_p = json.loads((args.pandas / "predict_meta_pandas.json").read_text())
    rep_s = pd.read_csv(args.spark / "predict_reporting_spark.csv")
    rep_p = pd.read_csv(args.pandas / "predict_reporting_pandas.csv")

    agree = []
    for q in ("plan_sha256", "model_rows", "complete_rows", "main_rows", "main_train_rows", "support_dropped",
              "main_valid_not_accepted", "trim_lower", "trim_upper", "terms_used"):
        agree.append(dict(output="metadata", quantity=q, pandas=str(meta_p.get(q)), spark=str(meta_s.get(q)),
                          agree=meta_p.get(q) == meta_s.get(q), max_abs_difference=0.0))
    agree += compare(rep_p, rep_s, "reporting audit", ["feature"], exact=("used",),
                     close=("missing_share_originated", "missing_share_not_accepted"))
    agree += compare(coef_p, coef_s, "coefficients", ["variant", "fold", "column"], close=("beta",))
    agree += compare(met_p, met_s, "prediction metrics", ["variant", "model"], exact=("n",),
                     close=("sse", "sae", "rmse_bp", "mae_bp", "r2"))
    agree += compare(seg_p, seg_s, "segment medians", KEYS, exact=("n_valid",), close=("median_spread", "median_resid"))
    agree += compare(summary_p, summary[summary.variant != "gbt"], "matched gaps", ["variant", "amount_band"],
                     exact=("n_segments", "n_originated", "n_not_accepted"),
                     close=("raw_gap_bp", "resid_gap_bp", "raw_lo_bp", "raw_hi_bp", "resid_lo_bp", "resid_hi_bp"))
    agree = pd.DataFrame(agree)

    coef_mean = (coef_s[coef_s.variant == "main"].groupby("column", sort=False)["beta"]
                 .agg(["mean", "min", "max"]).reset_index())
    summary.to_csv(args.out / "predict_matched.csv", index=False)
    metrics.to_csv(args.out / "predict_metrics.csv", index=False)
    agree.to_csv(args.out / "predict_agreement.csv", index=False)
    coef_mean.to_csv(args.out / "predict_coef_main.csv", index=False)
    rep_s.to_csv(args.out / "predict_reporting.csv", index=False)
    gbt_meta = args.spark / "predict_gbt_meta_spark.json"
    decision["gbt"] = json.loads(gbt_meta.read_text()) if gbt_meta.exists() else None
    decision["meta_spark"] = meta_s
    (args.out / "predict_decision.json").write_text(json.dumps(decision, indent=2, default=str))
    print(summary.to_string())
    print(metrics.to_string())
    print(agree.groupby("output")["agree"].all().to_string())
    print(json.dumps({k: decision[k] for k in ("rule", "original_gap_bp", "raw_gap_bp", "resid_gap_bp",
                                                "share_explained", "unknown_core_reproduces_original")}, indent=2))


if __name__ == "__main__":
    main()

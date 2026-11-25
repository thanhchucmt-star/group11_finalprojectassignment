"""Pandas/PyArrow side of the predictive extension (predictive_plan.md).

Reads the raw CSV itself with PyArrow, applies the eight cohort filters, assigns folds,
fits the cross-fitted models with NumPy and writes metrics, coefficients and segment medians.
Nothing here reads a Spark output.
"""
import argparse
import csv
import json
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

import hmda_predict_defs as P
from hmda_defs import BAND_EDGES, FILTERS, band

CHUNK = 250_000


# ------------------------------------------------------------------------------ extraction
def extract(raw, destination):
    with raw.open(newline="") as handle:
        names = next(csv.reader(handle))
    wanted = list(dict.fromkeys(P.FILTER_COLUMNS + P.EXTRACT))
    convert = pv.ConvertOptions(column_types={n: pa.string() for n in names}, strings_can_be_null=False,
                                include_columns=wanted)
    reader = pv.open_csv(raw, read_options=pv.ReadOptions(block_size=64 << 20), convert_options=convert)
    parts = []
    for batch in reader:
        keep = None
        for _, column, values in FILTERS:
            term = pc.is_in(pc.utf8_trim(batch.column(column), " "), value_set=pa.array(values))
            keep = term if keep is None else pc.and_(keep, term)
        chosen = batch.filter(keep)
        if chosen.num_rows:
            parts.append(pa.table([pc.utf8_trim(chosen.column(c), " ") for c in P.EXTRACT], names=P.EXTRACT))
    table = pa.concat_tables(parts)
    joined = pc.binary_join_element_wise(*[table.column(c) for c in P.EXTRACT], "|").to_pylist()
    crc = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in joined), dtype=np.int64, count=len(joined))
    table = table.append_column("fold", pa.array((crc % P.FOLDS).astype(np.int32)))
    pq.write_table(table, destination)
    return table.num_rows


# --------------------------------------------------------------------------------- sample
def model_sample(ext):
    feats = P.pandas_features(ext)
    nonempty = lambda s: P.cat_apply(s, lambda t: t != "", bool)
    valid = P.market_ok(ext["county_code"]) & nonempty(ext["lei"]) & nonempty(ext["loan_type"])
    df = feats.assign(lei=ext["lei"], county_code=ext["county_code"], loan_type=ext["loan_type"],
                      action=P.cat_apply(ext["action_taken"], lambda t: int(t) if t else -1, np.int64), fold=ext["fold"].astype(int))
    df["amount_band"] = pd.Categorical(band(df["loan_amount_num"].to_numpy(), BAND_EDGES["amount_band"][1]))
    df = df.loc[valid].reset_index(drop=True)
    for c in ("lei", "county_code", "loan_type"):
        df[c] = df[c].cat.remove_unused_categories()
    return df


def reporting_audit(df):
    rows, used = [], []
    for name in P.TERMS:
        missing = df[name].eq("other") if name == "conforming_cat" else df[name].eq("Unknown")
        share = {a: float(missing[df["action"].eq(a)].mean()) for a in (1, 2)}
        gap = abs(share[2] - share[1])
        ok = gap <= P.MAX_REPORTING_GAP
        rows.append(dict(feature=name, missing_share_originated=share[1], missing_share_not_accepted=share[2],
                         gap=gap, used=ok))
        if ok:
            used.append(name)
    return pd.DataFrame(rows), used


def trim_bounds(df):
    y = df.loc[df["action"].eq(1), "spread"].dropna().to_numpy()
    lo, hi = np.quantile(y, list(P.TRIM), method="inverted_cdf")
    return float(lo), float(hi)


# ------------------------------------------------------------------------------- fitting
def chunked_gram(x, y, rows):
    p = x.shape[1]
    gram, cross = np.zeros((p, p)), np.zeros(p)
    idx = np.flatnonzero(rows)
    for start in range(0, len(idx), CHUNK):
        part = idx[start:start + CHUNK]
        xf = x[part].astype(np.float64)
        gram += xf.T @ xf
        cross += xf.T @ y[part]
    return gram, cross


def group_sums(codes, size, x, y, rows):
    idx = np.flatnonzero(rows)
    c = codes[idx]
    n = np.bincount(c, minlength=size).astype(np.float64)
    sy = np.bincount(c, weights=y[idx], minlength=size)
    sx = np.column_stack([np.bincount(c, weights=x[idx, j].astype(np.float64), minlength=size)
                          for j in range(x.shape[1])]) if x.shape[1] else np.zeros((size, 0))
    return n, sy, sx


def fit_variant(df, columns, train, lo, hi):
    """Cross-fitted M0, M1, M2. Returns per-row predictions, residuals and coefficients."""
    x = P.pandas_design(df, columns)
    y = df["spread"].to_numpy(dtype=np.float64)
    y0 = np.where(train, y, 0.0)
    fold = df["fold"].to_numpy()
    seg = df.groupby(["lei", "county_code", "loan_type"], observed=True, sort=False).ngroup().to_numpy()
    mkt_of_row = df.groupby(["county_code", "loan_type"], observed=True, sort=False).ngroup().to_numpy()
    prod_of_row = df.groupby(["loan_type"], observed=True, sort=False).ngroup().to_numpy()
    nseg, nmkt, nprod = seg.max() + 1, mkt_of_row.max() + 1, prod_of_row.max() + 1
    seg_mkt = np.zeros(nseg, dtype=np.int64); seg_mkt[seg] = mkt_of_row
    mkt_prod = np.zeros(nmkt, dtype=np.int64); mkt_prod[mkt_of_row] = prod_of_row

    g_tot, c_tot = chunked_gram(x, y0, train)
    n_tot, sy_tot, sx_tot = group_sums(seg, nseg, x, y0, train)
    betas = np.zeros((P.FOLDS, x.shape[1]))
    alpha = {m: np.zeros((P.FOLDS, nseg)) for m in ("M1", "M2")}
    alpha0 = np.zeros((P.FOLDS, nprod))
    for k in range(P.FOLDS):
        in_k = train & (fold == k)
        g_k, c_k = chunked_gram(x, y0, in_k)
        n_k, sy_k, sx_k = group_sums(seg, nseg, x, y0, in_k)
        n_tr, sy_tr, sx_tr = n_tot - n_k, sy_tot - sy_k, sx_tot - sx_k
        pos = n_tr > 0
        scaled = sx_tr[pos] / np.sqrt(n_tr[pos])[:, None]
        within_gram = (g_tot - g_k) - scaled.T @ scaled
        within_cross = (c_tot - c_k) - (sx_tr[pos] * (sy_tr[pos] / n_tr[pos])[:, None]).sum(axis=0)
        betas[k] = P.solve(within_gram, within_cross)
        for model, beta in (("M1", np.zeros(x.shape[1])), ("M2", betas[k])):
            a_seg = sy_tr - sx_tr @ beta
            a_mkt = np.bincount(seg_mkt, weights=a_seg, minlength=nmkt)
            n_mkt = np.bincount(seg_mkt, weights=n_tr, minlength=nmkt)
            a_prod = np.bincount(mkt_prod, weights=a_mkt, minlength=nprod)
            n_prod = np.bincount(mkt_prod, weights=n_mkt, minlength=nprod)
            alpha_prod = a_prod / n_prod
            alpha_mkt = P.shrink(a_mkt, n_mkt, alpha_prod[mkt_prod])
            alpha[model][k] = P.shrink(a_seg, n_tr, alpha_mkt[seg_mkt])
            if model == "M1":
                alpha0[k] = alpha_prod
    xb = np.zeros(len(df))
    for k in range(P.FOLDS):
        rows = np.flatnonzero(fold == k)
        for start in range(0, len(rows), CHUNK):
            part = rows[start:start + CHUNK]
            xb[part] = x[part].astype(np.float64) @ betas[k]
    pred = {"M0": alpha0[fold, prod_of_row], "M1": alpha["M1"][fold, seg], "M2": alpha["M2"][fold, seg] + xb}
    seg_train_n = n_tot[seg]
    return pred, y - xb, betas, seg_train_n, x


def metrics(variant, y, pred, evaluate):
    yv = y[evaluate]
    sst = ((yv - yv.mean()) ** 2).sum()
    sse = {m: float(((yv - p[evaluate]) ** 2).sum()) for m, p in pred.items()}
    rows = []
    for m, p in pred.items():
        e = yv - p[evaluate]
        rows.append(dict(variant=variant, model=m, n=int(evaluate.sum()), sse=sse[m], sae=float(np.abs(e).sum()),
                         rmse_bp=100 * np.sqrt(sse[m] / len(yv)), mae_bp=100 * np.abs(e).mean(),
                         r2=1 - sse[m] / sst, r2_vs_M0=1 - sse[m] / sse["M0"],
                         r2_within_vs_M1=1 - sse[m] / sse["M1"] if m == "M2" else np.nan))
    return rows


def segment_medians(df, resid, rows, variant, by_band=False):
    keys = ["lei", "county_code", "loan_type"] + (["amount_band"] if by_band else []) + ["action"]
    work = df.loc[rows, keys].assign(spread=df.loc[rows, "spread"].to_numpy(), resid=resid[rows])
    g = work.groupby(keys, sort=False, observed=True)
    out = g.agg(n_valid=("spread", "size"), median_spread=("spread", "median"),
                median_resid=("resid", "median")).reset_index()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
    if not by_band:
        out.insert(3, "amount_band", "All")
    out.insert(0, "variant", variant)
    return out.rename(columns={"action": "action_taken"})


def support_flags(df, x_cols, train, seg_train_n):
    """Approvals not accepted in thin bins (< 100 training originations within loan type)
    or in segments with < 20 training originations."""
    thin = np.zeros(len(df), dtype=bool)
    for name in P.CORE:
        cell = df.groupby(["loan_type", name], observed=True, sort=False).ngroup().to_numpy()
        n = np.bincount(cell[train], minlength=cell.max() + 1)
        thin |= n[cell] < 100
    thin |= seg_train_n < 20
    return thin & df["action"].eq(2).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "extract", "fit"], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    extract_path = args.out / "predict_extract_pandas.parquet"
    meta = {"plan_sha256": P.plan_sha256()}
    print(f"Plan SHA-256 {meta['plan_sha256']}", flush=True)
    if args.phase in ("all", "extract"):
        t = perf_counter()
        meta["extract_rows"] = extract(args.raw, extract_path)
        meta["extract_seconds"] = perf_counter() - t
        print(f"Extracted {meta['extract_rows']:,} rows in {meta['extract_seconds']:.1f}s", flush=True)
        (args.out / "predict_extract_meta_pandas.json").write_text(json.dumps(meta, indent=2))
        if args.phase == "extract":
            return
    t = perf_counter()
    ext = pq.read_table(extract_path).to_pandas(strings_to_categorical=True)
    df = model_sample(ext)
    del ext
    audit, terms = reporting_audit(df)
    audit.to_csv(args.out / "predict_reporting_pandas.csv", index=False)
    lo, hi = trim_bounds(df)
    y = df["spread"].to_numpy()
    complete = np.logical_and.reduce([(df[n] != "Unknown").to_numpy() for n in P.CORE])
    finite = np.isfinite(y)
    in_range = finite & (y >= lo) & (y <= hi)
    meta.update(model_rows=len(df), trim_lower=lo, trim_upper=hi, terms_used=terms,
                complete_rows=int(complete.sum()))
    metric_rows, coef_rows, seg_tables = [], [], []
    for variant, unknown_core in (("main", False), ("unknown_core", True)):
        rows = complete if not unknown_core else np.ones(len(df), dtype=bool)
        sub = df.loc[rows].reset_index(drop=True)
        cols = P.design_columns(terms, unknown_core=unknown_core)
        ys = sub["spread"].to_numpy()
        train = sub["action"].eq(1).to_numpy() & np.isfinite(ys) & (ys >= lo) & (ys <= hi)
        pred, resid, betas, seg_n, x = fit_variant(sub, cols, train, lo, hi)
        metric_rows += metrics(variant, ys, pred, train)
        for k in range(P.FOLDS):
            coef_rows += [dict(variant=variant, fold=k, column=c, beta=b) for c, b in zip(cols, betas[k])]
        valid = np.isfinite(ys)
        seg_tables.append(segment_medians(sub, resid, valid, variant))
        if variant == "main":
            seg_tables.append(segment_medians(sub, resid, valid, "main_by_amount_band", by_band=True))
            seg_tables.append(segment_medians(sub, resid, valid & (ys >= lo) & (ys <= hi), "central_98"))
            thin = support_flags(sub, cols, train, seg_n)
            seg_tables.append(segment_medians(sub, resid, valid & ~thin, "support"))
            meta.update(main_rows=len(sub), main_train_rows=int(train.sum()), support_dropped=int((valid & thin).sum()),
                        main_valid_not_accepted=int((valid & sub["action"].eq(2).to_numpy()).sum()))
        del x
    pd.DataFrame(metric_rows).to_csv(args.out / "predict_metrics_pandas.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(args.out / "predict_coef_pandas.csv", index=False)
    seg = pd.concat(seg_tables, ignore_index=True)
    seg.to_parquet(args.out / "predict_segments_pandas.parquet", index=False)
    meta["fit_seconds"] = perf_counter() - t
    (args.out / "predict_meta_pandas.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(metric_rows).to_string(), flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

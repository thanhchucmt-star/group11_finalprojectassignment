"""State-level geography of non-acceptance and price gaps, from county segment tables.

Both engines already produce lender x county x loan type x outcome segments
(Spark: hmda_tables/segments.csv; pandas read_csv: segments_county_pandas_csv.parquet).
The first two digits of a county FIPS code give the state, so states are sums of counties.
Medians are not additive, so price gaps use the matched-segment rule of the main analysis
(>=10 valid prices per outcome in a lender x county x loan type segment, weighted by the
smaller count) and are averaged within each state.

Outputs (tables): geography_state.csv (Spark), geography_state_pandas.csv,
geography_agreement.csv, pandas_csv_agreement.csv (pandas read_csv vs Spark).
Outputs (figures): geography_states.png/.pdf (report) and, with --slides, s_geo_map.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
        "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
        "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
        "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
        "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
        "54": "WV", "55": "WI", "56": "WY", "72": "PR", "60": "AS", "66": "GU", "69": "MP", "78": "VI"}
GRID = {"AK": (0, 0), "ME": (10, 0), "VT": (9, 1), "NH": (10, 1),
        "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2), "IL": (5, 2), "WI": (6, 2),
        "MI": (7, 2), "NY": (8, 2), "RI": (9, 2), "MA": (10, 2),
        "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3), "IN": (5, 3), "OH": (6, 3),
        "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
        "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4), "WV": (6, 4),
        "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
        "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5), "SC": (7, 5),
        "DC": (8, 5), "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
        "HI": (0, 7), "TX": (3, 7), "FL": (8, 7), "PR": (10, 7)}
MIN_PRICES = 10
INK, BLUE, VERM, NAVY = "#111111", "#0072B2", "#D55E00", "#0B3C8C"


def load(path):
    path = Path(path)
    if path.suffix == ".parquet":
        s = pd.read_parquet(path)
    else:
        s = pd.read_csv(path, dtype={"market": str, "lei": str, "loan_type": str, "action_taken": str})
    s = s[(s["geo"] == "county") & s["market"].notna()].copy()   # county unknown: no state
    for c in ["lei", "market", "loan_type", "action_taken"]:
        s[c] = s[c].astype(str)
    s["state"] = s["market"].str[:2]
    return s[["lei", "market", "state", "loan_type", "action_taken", "n", "n_price_valid", "median"]]


def matched(seg):
    """Lender x county x loan type segments with >=10 valid prices in both outcomes."""
    k = ["lei", "market", "state", "loan_type"]
    w = seg.pivot_table(index=k, columns="action_taken", values=["n_price_valid", "median"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna(subset=["median_1", "median_2"])
    w = w[(w["n_price_valid_1"] >= MIN_PRICES) & (w["n_price_valid_2"] >= MIN_PRICES)].reset_index()
    w["gap_bp"] = (w["median_2"] - w["median_1"]) * 100
    w["weight"] = np.minimum(w["n_price_valid_1"], w["n_price_valid_2"])
    return w


def states(seg):
    n = seg.pivot_table(index="state", columns="action_taken", values="n", aggfunc="sum", fill_value=0)
    t = pd.DataFrame({"n": n.get("1", 0) + n.get("2", 0), "n_fallout": n.get("2", 0)})
    t["rate_pct"] = t["n_fallout"] / t["n"] * 100
    fall = seg[seg["action_taken"] == "2"].groupby(["state", "lei"])["n"].sum()
    t["lenders_with_fallout"] = fall[fall > 0].groupby("state").size()
    t["top_lender_share_pct"] = fall.groupby("state").max() / t["n_fallout"].replace(0, np.nan) * 100
    top = fall.groupby("state").idxmax().dropna()
    tot = seg.groupby(["state", "lei"])["n"].sum()
    top_n = pd.Series({st: tot[(st, le)] for st, (_, le) in top.items()})
    top_f = fall.groupby("state").max()
    t["rate_without_top_lender_pct"] = (t["n_fallout"] - top_f) / (t["n"] - top_n) * 100
    t["top3_lender_share_pct"] = (fall.groupby("state", group_keys=False).nlargest(3).groupby("state").sum()
                                  / t["n_fallout"].replace(0, np.nan) * 100)
    m = matched(seg)
    g = m.groupby("state").apply(lambda d: pd.Series({
        "matched_segments": len(d), "gap_bp": np.average(d["gap_bp"], weights=d["weight"])}),
        include_groups=False)
    t = t.join(g)
    t["matched_segments"] = t["matched_segments"].fillna(0).astype(int)
    t.insert(0, "abbr", [FIPS.get(s, s) for s in t.index])
    t.index.name = "state_fips"
    national = np.average(m["gap_bp"], weights=m["weight"])
    return t.reset_index(), m, national


def compare(a, b, keys, cols, tol):
    j = a.merge(b, on=keys, how="outer", suffixes=("_spark", "_pandas"), indicator=True)
    rows = [("rows only in one engine", int((j["_merge"] != "both").sum()), 0, True)]
    both = j[j["_merge"] == "both"]
    for c in cols:
        x, y = both[f"{c}_spark"].astype(float), both[f"{c}_pandas"].astype(float)
        d = (x - y).abs()
        d = d[~(x.isna() & y.isna())]
        mx = float(d.max()) if len(d) else 0.0
        rows.append((c, int(len(d)), mx, bool(mx <= tol.get(c, 0) and not d.isna().any())))
    return pd.DataFrame(rows, columns=["quantity", "rows_compared", "max_abs_diff", "agree"])


GEOJSON = Path(__file__).with_name("us_states.geojson")  # PublicaMundi MappingAPI, US Census-derived


def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    """Albers equal-area conic projection (US National Atlas parameters), in degrees in, units out."""
    r = np.radians
    n = (np.sin(r(p1)) + np.sin(r(p2))) / 2
    c = np.cos(r(p1)) ** 2 + 2 * n * np.sin(r(p1))
    rho0 = np.sqrt(c - 2 * n * np.sin(r(lat0))) / n
    rho = np.sqrt(c - 2 * n * np.sin(r(np.asarray(lat)))) / n
    th = n * r(np.asarray(lon) - lon0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


INSET = {"02": dict(scale=0.35, dx=-0.30, dy=-0.33, proj=dict(lon0=-154, lat0=50, p1=55, p2=65)),
         "15": dict(scale=1.0, dx=-0.08, dy=-0.36, proj=dict(lon0=-157, lat0=20, p1=8, p2=18)),
         "72": dict(scale=1.0, dx=0.37, dy=-0.27, proj=dict(lon0=-66.5, lat0=18, p1=8, p2=18))}


def state_shapes():
    gj = json.loads(GEOJSON.read_text())
    shapes = {}
    for f in gj["features"]:
        fips = str(f["id"]).zfill(2)
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = []
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            if fips == "02":
                ring[:, 0] = np.where(ring[:, 0] > 0, ring[:, 0] - 360, ring[:, 0])
            if fips in INSET:
                o = INSET[fips]
                x, y = albers(ring[:, 0], ring[:, 1], **o["proj"])
                x, y = x * o["scale"] + o["dx"], y * o["scale"] + o["dy"]
            else:
                x, y = albers(ring[:, 0], ring[:, 1])
            rings.append(np.column_stack([x, y]))
        shapes[fips] = rings
    return shapes


OFFSET = {"12": (0.035, -0.01), "22": (-0.02, 0.005), "26": (0.018, 0.0), "39": (0.004, -0.012), "16": (0.0, -0.03),
          "06": (-0.01, 0.0), "24": (0.0, 0.0), "51": (0.01, 0.0), "36": (0.01, 0.0), "15": (0.0, 0.02),
          "02": (0.0, 0.0), "72": (0.0, 0.025), "45": (0.005, 0.0), "54": (-0.006, -0.006)}


def centroid(ring):
    x, y = ring[:, 0], ring[:, 1]
    a = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = a.sum() / 2
    if abs(area) < 1e-12:
        return x.mean(), y.mean(), 0.0
    return ((x + np.roll(x, -1)) * a).sum() / (6 * area), ((y + np.roll(y, -1)) * a).sum() / (6 * area), abs(area)


def tile_map(ax, t, value="rate_pct", fs=9, label_fmt="{:.1f}", cax=None, lab_fs=None):
    """Choropleth of real state boundaries (Albers equal-area; Alaska, Hawaii, Puerto Rico inset).
    Large states carry abbreviation and value, mid-size states the abbreviation; the small
    north-eastern states are read from the table."""
    from matplotlib.patches import Polygon
    lab_fs = lab_fs or fs - 2
    v = t.set_index("state_fips")[value]
    lo, hi = np.nanpercentile(v.values, 2), np.nanpercentile(v.values, 98)
    cmap = matplotlib.colormaps["Blues"]
    for fips, rings in state_shapes().items():
        val = v.get(fips, np.nan)
        x = 0.12 + 0.88 * np.clip((val - lo) / (hi - lo), 0, 1) if np.isfinite(val) else None
        col = cmap(x) if x is not None else "white"
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=col, edgecolor="white", lw=0.6))
        if not np.isfinite(val):
            continue
        cx, cy, area = max((centroid(r) for r in rings), key=lambda c: c[2])
        if fips in INSET:
            area = 1.0
        dx, dy = OFFSET.get(fips, (0, 0))
        txt = "white" if x is not None and x > 0.55 else INK
        if fips in ("15", "72"):
            txt = INK
            cy += 0.02
        if area > 0.0026 or fips in ("15", "72", "41", "16", "47"):
            ax.text(cx + dx, cy + dy, f"{FIPS[fips]}\n{label_fmt.format(val)}", ha="center", va="center",
                    fontsize=lab_fs, color=txt, linespacing=0.95)
    ax.set_xlim(-0.40, 0.43); ax.set_ylim(-0.44, 0.27); ax.set_aspect("equal"); ax.axis("off")
    if cax is not None:
        sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "b", [cmap(0.12), cmap(1.0)]), norm=matplotlib.colors.Normalize(lo, hi))
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=fs - 2)
        cb.set_label("Approvals not accepted (%)", fontsize=fs - 2)
    return lo, hi, cmap


def dumbbell(ax, t, fs=9, nat=3.11, k=10):
    """Ten highest-rate states (2,000+ approvals): rate with and without the state's largest lender."""
    d = t[(t["n"] >= 2000) & t["abbr"].isin(GRID)].nlargest(k, "rate_pct")[::-1]
    y = np.arange(len(d))
    ax.hlines(y, d["rate_without_top_lender_pct"], d["rate_pct"], color=BLUE, lw=1.5)
    ax.scatter(d["rate_pct"], y, s=fs * 4, color=NAVY, zorder=3, label="All lenders")
    ax.scatter(d["rate_without_top_lender_pct"], y, s=fs * 4, facecolor="white", edgecolor=BLUE, lw=1.5,
               zorder=3, label="Without largest lender")
    ax.axvline(nat, color=INK, lw=1, ls="--")
    ax.set_yticks(y, d["abbr"], fontsize=fs)
    ax.set_xlabel("Approvals not accepted (%)", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.legend(fontsize=fs - 1, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return float(d["rate_pct"].std())


def figures(t, out, slides=None):
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
    fig = plt.figure(figsize=(7.1, 2.3))
    ax = fig.add_axes([0.0, 0.0, 0.52, 0.89])
    lo, hi, cmap = tile_map(ax, t, fs=7, lab_fs=5, cax=fig.add_axes([0.34, 0.13, 0.17, 0.03]))
    ax.set_title("(a) Approvals not accepted by state (%)", fontsize=8, loc="left")
    bx = fig.add_axes([0.6, 0.19, 0.385, 0.69])
    dumbbell(bx, t, fs=7)
    bx.set_title("(b) 10 highest states, largest lender removed", fontsize=8, loc="left")
    big = t[t["n"] >= 2000]
    rho = big[["top_lender_share_pct", "rate_pct"]].corr(method="spearman").iloc[0, 1]
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"geography_states.{ext}", dpi=300)
    plt.close(fig)
    if slides:
        fig = plt.figure(figsize=(12.08, 4.6), dpi=100)
        ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
        tile_map(ax, t, fs=15, lab_fs=11, cax=fig.add_axes([0.40, 0.16, 0.19, 0.03]))
        bx = fig.add_axes([0.66, 0.14, 0.33, 0.84])
        dumbbell(bx, t, fs=15)
        fig.savefig(Path(slides) / "s_geo_map.png", dpi=200)
        plt.close(fig)
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True, help="hmda_tables/segments.csv")
    ap.add_argument("--pandas", required=True, help="segments_county_pandas_csv.parquet")
    ap.add_argument("--pandas-waterfall", help="waterfall_pandas_csv.csv")
    ap.add_argument("--spark-waterfall", help="hmda_tables/waterfall.csv")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--slides")
    a = ap.parse_args()
    tables, figs = Path(a.tables), Path(a.figures)
    for d in [tables, figs] + ([Path(a.slides)] if a.slides else []):
        d.mkdir(parents=True, exist_ok=True)
    sp, pd_ = load(a.spark), load(a.pandas)
    ts, ms, nat_s = states(sp)
    tp, mp, nat_p = states(pd_)
    ts.to_csv(tables / "geography_state.csv", index=False)
    tp.to_csv(tables / "geography_state_pandas.csv", index=False)
    seg_cmp = compare(sp, pd_, ["lei", "market", "loan_type", "action_taken"], ["n", "n_price_valid", "median"],
                      {"median": 1e-9})
    seg_cmp.insert(0, "table", "county segments")
    st_cmp = compare(ts, tp, ["state_fips"], ["n", "n_fallout", "rate_pct", "top_lender_share_pct", "rate_without_top_lender_pct",
                                               "matched_segments", "gap_bp"],
                     {"rate_pct": 1e-9, "top_lender_share_pct": 1e-9, "rate_without_top_lender_pct": 1e-9, "gap_bp": 1e-6})
    st_cmp.insert(0, "table", "state geography")
    extra = [("national matched gap (bp)", 1, abs(nat_s - nat_p), abs(nat_s - nat_p) < 1e-6),
             ("matched segments", 1, abs(len(ms) - len(mp)), len(ms) == len(mp))]
    if a.pandas_waterfall and a.spark_waterfall:
        w = pd.read_csv(a.spark_waterfall).merge(pd.read_csv(a.pandas_waterfall), on="step", suffixes=("_s", "_p"))
        extra.append(("waterfall steps", len(w), float((w.n_s - w.n_p).abs().max()), bool((w.n_s == w.n_p).all())))
    ex = pd.DataFrame(extra, columns=["quantity", "rows_compared", "max_abs_diff", "agree"])
    ex.insert(0, "table", "pandas read_csv vs Spark")
    agree = pd.concat([seg_cmp, st_cmp, ex], ignore_index=True)
    agree.to_csv(tables / "geography_agreement.csv", index=False)
    rho = figures(ts, figs, a.slides)
    summary = {"states_with_counties": int(len(ts)), "records_with_county": int(ts["n"].sum()),
               "national_matched_gap_bp": round(nat_s, 3), "matched_segments": int(len(ms)),
               "spearman_rate_vs_top_lender_share_2000plus": round(float(rho), 3),
               "all_agree": bool(agree["agree"].all())}
    (tables / "geography_summary.json").write_text(json.dumps(summary, indent=2))
    print(agree.to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

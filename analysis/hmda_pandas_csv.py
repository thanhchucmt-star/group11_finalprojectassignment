"""Pandas-only pipeline from the raw CSV: pd.read_csv in chunks, no PyArrow reader.

It answers one question: does pandas' own CSV reader give the same cohort and the same
segment results as the Spark pipeline? It rebuilds, with plain pandas,
  * the eight-step cohort waterfall,
  * counts by outcome and loan type,
  * county segments (lender x county x loan type x outcome): n, valid prices, median spread,
and writes them next to the other pandas checks. hmda_geography.py compares them with Spark.

Usage: python hmda_pandas_csv.py --raw 2024_public_lar_one_year_csv.csv --out hmda_pandas_checks
On a machine that limits run time, split the read: --parts 8 --part 0 ... --part 7, then --combine.
Each part reads one byte range of the CSV (cut at line ends) with the same pd.read_csv call.
"""
import io
import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

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
KEEP = ["lei", "county_code", "state_code", "loan_type", "action_taken", "rate_spread"]
USECOLS = sorted({c for _, c, _ in FILTERS} | set(KEEP))


def filter_chunks(source, chunksize, names=None):
    counts = np.zeros(len(FILTERS) + 1, dtype=np.int64)
    parts = []
    reader = pd.read_csv(source, usecols=USECOLS, dtype=str, keep_default_na=False, na_filter=False,
                         chunksize=chunksize, engine="c", header=None if names else "infer", names=names)
    for chunk in reader:
        chunk = chunk.apply(lambda s: s.str.strip())          # Spark staging trims every field
        counts[0] += len(chunk)
        keep = np.ones(len(chunk), dtype=bool)
        for i, (_, column, values) in enumerate(FILTERS, 1):
            keep &= chunk[column].isin(values).to_numpy()
            counts[i] += keep.sum()
        parts.append(chunk.loc[keep, KEEP].astype("category"))
    return parts, counts


def read_part(raw, part, nparts, chunksize):
    """Rows whose line starts inside byte range part/nparts of the file (header excluded)."""
    with open(raw, "rb") as f:
        header = f.readline()
        start0, size = f.tell(), Path(raw).stat().st_size
        bounds = [start0 + (size - start0) * k // nparts for k in range(nparts + 1)]
        def align(pos):
            if pos <= start0:
                return start0
            f.seek(pos - 1)
            f.readline()
            return f.tell()
        lo, hi = align(bounds[part]), align(bounds[part + 1]) if part + 1 < nparts else size
        f.seek(lo)
        block = f.read(hi - lo)
    names = header.decode().strip().split(",")
    return filter_chunks(io.BytesIO(block), chunksize, names)


def finish(parts, counts):
    cohort = pd.concat(parts, ignore_index=True)
    for c in KEEP:
        cohort[c] = cohort[c].astype(str).astype("category")
    spread = pd.to_numeric(cohort["rate_spread"].astype(str), errors="coerce")
    cohort["spread"] = spread.where(np.isfinite(spread))      # try_cast with non-finite set missing
    county = cohort["county_code"].astype(str)
    valid = county.str.fullmatch(r"[0-9]{5}") & county.ne("99999")
    cohort["market"] = county.where(valid)
    waterfall = pd.DataFrame({"step": ["input"] + [s for s, _, _ in FILTERS], "n": counts})
    return cohort, waterfall


def read_cohort(raw, chunksize):
    parts, counts = filter_chunks(raw, chunksize)
    return finish(parts, counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    work = out / "pandas_csv_parts"
    if a.part is not None:
        work.mkdir(exist_ok=True)
        parts, counts = read_part(a.raw, a.part, a.parts, a.chunksize)
        frame = pd.concat(parts, ignore_index=True).astype(str)
        frame.to_parquet(work / f"part_{a.part:02d}.parquet", index=False)
        np.save(work / f"counts_{a.part:02d}.npy", counts)
        (work / f"seconds_{a.part:02d}.txt").write_text(f"{perf_counter() - t0:.2f}")
        print(f"part {a.part}: {counts[0]:,} rows, {counts[-1]:,} kept, {perf_counter() - t0:.0f}s")
        return
    if a.combine:
        files = sorted(work.glob("part_*.parquet"))
        assert len(files) == a.parts, f"expected {a.parts} parts, found {len(files)}"
        parts = [pd.read_parquet(f).astype("category") for f in files]
        counts = sum(np.load(f) for f in sorted(work.glob("counts_*.npy")))
        cohort, waterfall = finish(parts, counts)
        part_seconds = [float(f.read_text()) for f in sorted(work.glob("seconds_*.txt"))]
    else:
        part_seconds = []
        cohort, waterfall = read_cohort(a.raw, a.chunksize)
    t_read = perf_counter() - t0
    waterfall.to_csv(out / "waterfall_pandas_csv.csv", index=False)

    outcome = cohort.groupby(["loan_type", "action_taken"], observed=True).agg(
        n=("spread", "size"), n_price_valid=("spread", "count"),
        spread_median=("spread", "median")).reset_index()
    outcome.to_csv(out / "outcomes_pandas_csv.csv", index=False)

    seg = cohort[cohort["market"].notna()]
    keys = ["lei", "market", "loan_type", "action_taken"]
    g = seg.groupby(keys, observed=True)["spread"]
    segments = pd.concat([g.size().rename("n"), g.count().rename("n_price_valid"),
                          g.quantile(0.5, interpolation="linear").rename("median")], axis=1).reset_index()
    segments.insert(0, "geo", "county")
    segments.to_parquet(out / "segments_county_pandas_csv.parquet", index=False)

    meta = {"reader": f"pandas {pd.__version__} read_csv(engine='c', dtype=str, chunksize={a.chunksize})",
            "python": sys.version.split()[0], "rows_read": int(waterfall.n.iloc[0]),
            "cohort": int(len(cohort)), "read_and_filter_seconds": round(t_read, 1),
            "total_seconds": round(perf_counter() - t0, 1),
            "parts": a.parts if a.combine else 1,
            "part_read_seconds_sum": round(sum(part_seconds), 1),
            "note": ("parts read in separate processes (byte ranges, 180 s device limit); "
                     "read_and_filter_seconds covers the combine step only") if a.combine else "single pass",
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    (out / "pandas_csv_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

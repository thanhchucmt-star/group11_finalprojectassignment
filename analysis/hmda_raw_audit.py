"""Read the raw CSV with PyArrow, without Spark staging, to record the file schema,
recount the cohort filters, and count repeated public rows in the file and in the cohort.
"""
import argparse
import csv
import hashlib
from array import array
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

from hmda_defs import FILTERS

NUMERIC = r"^-?[0-9]+(\.[0-9]+)?$"


def line_duplicates(raw):
    """MD5 of every physical line, as in hmda_spark.duplicate_audit (header included)."""
    digests = array("Q")
    with raw.open("rb", buffering=16 << 20) as handle:
        for line in handle:
            digests.append(int.from_bytes(hashlib.md5(line.rstrip(b"\r\n")).digest()[:8], "little"))
    _, counts = np.unique(np.frombuffer(digests, dtype=np.uint64), return_counts=True)
    repeated = counts[counts > 1]
    return {"lines": len(digests), "repeated_patterns": int(len(repeated)),
            "rows_in_repeated_patterns": int(repeated.sum()), "rows_beyond_first": int((repeated - 1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", choices=["all", "parse", "profile", "lines"], default="all",
                        help="parse: filters and cohort repeats; profile: per-column codes; lines: line hashes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "lines":
        lines = line_duplicates(args.raw)
        (args.out / "raw_line_duplicates.json").write_text(json.dumps(lines, indent=2))
        print(json.dumps(lines, indent=2), flush=True)
        return
    with args.raw.open(newline="") as handle:
        names = next(csv.reader(handle))
    convert = pv.ConvertOptions(column_types={n: pa.string() for n in names}, strings_can_be_null=False)
    reader = pv.open_csv(args.raw, read_options=pv.ReadOptions(block_size=64 << 20), convert_options=convert)

    do_parse = args.part in ("all", "parse")
    do_profile = args.part in ("all", "profile")
    counts = np.zeros(len(FILTERS) + 1, dtype=np.int64)
    profile = {n: dict(blank=0, na=0, exempt=0, numeric=0, other=0) for n in names}
    examples = {n: set() for n in names}
    cohort_hashes = []
    for batch in reader:
        counts[0] += batch.num_rows
        keep = pa.array(np.ones(batch.num_rows, dtype=bool))
        for i, (_, column, values) in enumerate(FILTERS if do_parse else [], 1):
            value = pc.utf8_trim(batch.column(column), " ")
            keep = pc.and_(keep, pc.is_in(value, value_set=pa.array(values)))
            counts[i] += pc.sum(keep).as_py() or 0
        for name in (names if do_profile else []):
            col = batch.column(name)
            blank = pc.equal(col, "")
            na = pc.equal(pc.utf8_upper(col), "NA")
            exempt = pc.equal(pc.utf8_upper(col), "EXEMPT")
            numeric = pc.match_substring_regex(col, NUMERIC)
            p = profile[name]
            p["blank"] += pc.sum(blank).as_py() or 0
            p["na"] += pc.sum(na).as_py() or 0
            p["exempt"] += pc.sum(exempt).as_py() or 0
            p["numeric"] += pc.sum(numeric).as_py() or 0
            if len(examples[name]) < 5:
                other = pc.filter(col, pc.invert(pc.or_(pc.or_(blank, na), pc.or_(exempt, numeric))))
                examples[name].update(other.slice(0, 50).to_pylist()[: 5 - len(examples[name])])
        eligible = batch.filter(keep) if do_parse else None
        if eligible is not None and eligible.num_rows:
            joined = pc.binary_join_element_wise(*eligible.columns, "\x1f")
            cohort_hashes.append(pd.util.hash_array(np.asarray(joined.to_numpy(zero_copy_only=False), dtype=object)))
    if do_profile:
        for name in names:
            p = profile[name]
            p["other"] = int(counts[0] - p["blank"] - p["na"] - p["exempt"] - p["numeric"])
        pd.DataFrame([{"column": n, "position": i + 1, **profile[n],
                       "other_examples": "; ".join(sorted(map(str, examples[n])))}
                      for i, n in enumerate(names)]).to_csv(args.out / "raw_schema_profile.csv", index=False)
        print(f"Saved raw_schema_profile.csv ({len(names)} columns)", flush=True)
    if do_parse:
        _, dup_counts = np.unique(np.concatenate(cohort_hashes), return_counts=True)
        repeated = dup_counts[dup_counts > 1]
        result = {
            "file": args.raw.name, "bytes": args.raw.stat().st_size, "columns": len(names),
            "records": int(counts[0]),
            "waterfall": dict(zip(["input", *[s for s, _, _ in FILTERS]], map(int, counts))),
            "cohort_repeated_rows": {"repeated_patterns": int(len(repeated)),
                                     "rows_in_repeated_patterns": int(repeated.sum()),
                                     "rows_beyond_first": int((repeated - 1).sum())},
        }
        (args.out / "raw_audit_pyarrow.json").write_text(json.dumps(result, indent=2))
        pd.DataFrame({"step": result["waterfall"].keys(), "n": result["waterfall"].values()}).to_csv(
            args.out / "waterfall_raw_pyarrow.csv", index=False)
        print(json.dumps(result, indent=2), flush=True)
    if args.part == "all":
        lines = line_duplicates(args.raw)
        (args.out / "raw_line_duplicates.json").write_text(json.dumps(lines, indent=2))


if __name__ == "__main__":
    main()

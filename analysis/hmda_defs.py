"""Pandas versions of the cohort, market and case-mix definitions used in hmda_spark.py.

Each function mirrors one Spark expression so the two engines apply the same rules.
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

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
GROUP = ["geo", "lei", "market", "loan_type"]
MARKET = ["geo", "market", "loan_type"]
BANDS = ["amount_band", "income_band", "dti_band", "cltv_band"]
BAND_EDGES = {
    "amount_band": ("loan_amount", [150000, 250000, 400000, 650000]),
    "income_band": ("income", [50, 100, 150, 250]),
    "dti_band": ("debt_to_income_ratio", [36, 43, 50]),
    "cltv_band": ("combined_loan_to_value_ratio", [70, 80, 90, 97]),
}
DTI_CODES = {"<20%": 19, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 61}


def dataset(staged):
    return ds.dataset(staged, format="parquet", partitioning="hive", exclude_invalid_files=True)


def cohort_filter():
    """Pushdown expression equal to the conjunction of the eight Spark filters."""
    expr = None
    for _, column, values in FILTERS:
        term = ds.field(column).isin(list(values))
        expr = term if expr is None else expr & term
    return expr


def read_cohort(staged, columns, batch_size=500_000):
    """Read eligible records only. String columns arrive as pandas Categoricals to save memory."""
    scanner = dataset(staged).scanner(columns=columns, filter=cohort_filter(), batch_size=batch_size)
    return scanner.to_table().to_pandas(strings_to_categorical=True)


def _numeric_values(values):
    out = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def number(series):
    """try_cast(x as double) with non-finite values set to missing (Spark: number())."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = np.append(_numeric_values(series.cat.categories.to_numpy()), np.nan)
        return cats[series.cat.codes.to_numpy()]  # code -1 (missing) picks the trailing NaN
    return _numeric_values(series.to_numpy())


def market_code(series):
    """Five-digit county or MSA/MD code, excluding 99999; otherwise missing."""
    text = series.astype("object")
    ok = text.str.fullmatch(r"[0-9]{5}").fillna(False).astype(bool) & text.ne("99999")
    return text.where(ok)


def dti_numeric(series):
    values = number(series)
    mapped = series.astype("object").map(DTI_CODES)
    return np.where(mapped.notna(), mapped.to_numpy(dtype=np.float64, na_value=np.nan), values)


def band(values, edges):
    """Same labels and boundaries as hmda_spark.band()."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.full(values.shape, f"{edges[-1]}+", dtype=object)
    for lo, hi in reversed(list(zip(edges[:-1], edges[1:]))):
        labels = np.where(values < hi, f"{lo}-<{hi}", labels)
    labels = np.where(values < edges[0], f"<{edges[0]}", labels)
    labels = np.where(np.isnan(values), "Unknown", labels)
    return labels


def case_mix_bands(frame):
    out = {}
    for name, (column, edges) in BAND_EDGES.items():
        raw = frame[column]
        values = dti_numeric(raw) if name == "dti_band" else number(raw)
        out[name] = band(values, edges)
    return out

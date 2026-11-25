"""Shared constants for the predictive extension (see predictive_plan.md).

Both engines import these lists, so bins, reference categories, folds and model settings
cannot drift apart. Feature construction itself is coded separately in each engine.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from hmda_defs import DTI_CODES, FILTERS, band, number

PLAN = Path(__file__).with_name("predictive_plan.md")

# Raw fields extracted for the model; the fold key is CRC32 of these values joined by "|".
EXTRACT = ["lei", "county_code", "loan_type", "action_taken", "rate_spread", "loan_amount", "income",
           "debt_to_income_ratio", "combined_loan_to_value_ratio", "loan_term", "intro_rate_period",
           "conforming_loan_limit", "interest_only_payment", "balloon_payment", "negative_amortization",
           "other_nonamortizing_features"]
FILTER_COLUMNS = [column for _, column, _ in FILTERS]
NON_AMORTIZING = ["interest_only_payment", "balloon_payment", "negative_amortization",
                  "other_nonamortizing_features"]

FOLDS = 5
SHRINK = 20.0               # pseudo-count for shrinking segment and market effects
TRIM = (0.01, 0.99)         # training range of origination spreads (exact percentile_disc)
MIN_MATCHED = 10            # valid residuals per outcome in a segment
MAX_REPORTING_GAP = 0.05    # loan-term features: missing-share difference between outcomes
BOOT = 2000
SEED = 2024

CORE = {
    "amount_bin": ("loan_amount", [100000, 150000, 200000, 250000, 300000, 400000, 500000,
                                   650000, 800000, 1000000], "300000-<400000"),
    "income_bin": ("income", [40, 60, 80, 100, 150, 200, 300], "100-<150"),
    "dti_bin": ("debt_to_income_ratio", [20, 30, 36, 40, 45, 50, 60], "36-<40"),
    "cltv_bin": ("combined_loan_to_value_ratio", [60, 70, 75, 80, 85, 90, 95, 97, 100], "95-<97"),
}
TERMS = {
    "term_cat": (["360", "180", "other", "Unknown"], "360"),
    "arm_cat": (["fixed", "ARM", "Unknown"], "fixed"),
    "conforming_cat": (["C", "NC", "other"], "C"),
    "nonamort_cat": (["no", "yes", "Unknown"], "no"),
}
GBT = dict(maxIter=50, maxDepth=5, stepSize=0.1, maxBins=64, seed=SEED, subsamplingRate=1.0)
GBT_NUMERIC = ["loan_amount_num", "income_num", "dti_num", "cltv_num"]


def plan_sha256():
    return hashlib.sha256(PLAN.read_bytes()).hexdigest()


def labels(edges):
    """All labels band() can produce, in order, excluding Unknown."""
    out = [f"<{edges[0]}"] + [f"{lo}-<{hi}" for lo, hi in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    return out


def design_columns(terms_used, unknown_core=False):
    """Dummy-variable names: one column per non-reference category, in a fixed order."""
    cols = []
    for name, (_, edges, ref) in CORE.items():
        cats = labels(edges) + (["Unknown"] if unknown_core else [])
        cols += [f"{name}={c}" for c in cats if c != ref]
    for name in terms_used:
        cats, ref = TERMS[name]
        cols += [f"{name}={c}" for c in cats if c != ref]
    return cols


# ---------------------------------------------------------------- pandas feature builders
def _cat(series):
    """Categorical view of a raw string column (missing values become the empty string)."""
    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")
    cats = series.cat.categories.astype(str)
    return np.append(cats.to_numpy(dtype=object), ""), np.where(series.cat.codes.to_numpy() < 0,
                                                              len(cats), series.cat.codes.to_numpy())


def cat_apply(series, func, dtype=object):
    """Apply func to each distinct value once and expand by codes (memory-light)."""
    cats, codes = _cat(series)
    return np.asarray([func(c) for c in cats], dtype=dtype)[codes]


def _num(text):
    try:
        v = float(text)
    except ValueError:
        return np.nan
    return v if np.isfinite(v) else np.nan


def pandas_features(frame):
    """Bins and categories from raw strings; mirrors hmda_predict_spark.features()."""
    out = pd.DataFrame(index=frame.index)
    out["loan_amount_num"] = cat_apply(frame["loan_amount"], _num, np.float64)
    out["income_num"] = cat_apply(frame["income"], _num, np.float64)
    out["dti_num"] = cat_apply(frame["debt_to_income_ratio"],
                               lambda t: float(DTI_CODES[t]) if t in DTI_CODES else _num(t), np.float64)
    out["cltv_num"] = cat_apply(frame["combined_loan_to_value_ratio"], _num, np.float64)
    sources = {"amount_bin": "loan_amount_num", "income_bin": "income_num", "dti_bin": "dti_num",
               "cltv_bin": "cltv_num"}
    for name, (_, edges, _) in CORE.items():
        out[name] = pd.Categorical(band(out[sources[name]].to_numpy(), edges))
    def term(t):
        v = _num(t)
        return "Unknown" if np.isnan(v) else "360" if v == 360 else "180" if v == 180 else "other"
    def arm(t):
        if t == "NA":
            return "fixed"
        v = _num(t)
        return "Unknown" if np.isnan(v) else "ARM" if v > 0 else "fixed"
    out["term_cat"] = pd.Categorical(cat_apply(frame["loan_term"], term))
    out["arm_cat"] = pd.Categorical(cat_apply(frame["intro_rate_period"], arm))
    out["conforming_cat"] = pd.Categorical(cat_apply(frame["conforming_loan_limit"],
                                                     lambda t: t if t in ("C", "NC") else "other"))
    yes = np.zeros(len(frame), dtype=bool)
    no = np.ones(len(frame), dtype=bool)
    for c in NON_AMORTIZING:
        yes |= cat_apply(frame[c], lambda t: t == "1", bool)
        no &= cat_apply(frame[c], lambda t: t == "2", bool)
    out["nonamort_cat"] = pd.Categorical(np.where(yes, "yes", np.where(no, "no", "Unknown")))
    out["spread"] = cat_apply(frame["rate_spread"], _num, np.float64)
    return out


def pandas_design(feats, columns):
    """Dummy matrix (uint8) with the given columns; converted to float64 in chunks when summed."""
    x = np.zeros((len(feats), len(columns)), dtype=np.uint8)
    for j, col in enumerate(columns):
        name, cat = col.split("=", 1)
        series = feats[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            cats = list(series.cat.categories)
            x[:, j] = (series.cat.codes.to_numpy() == cats.index(cat)) if cat in cats else 0
        else:
            x[:, j] = (series.to_numpy() == cat)
    return x


def solve(gram, cross):
    """Least squares from within-segment cross-products; minimum-norm if a column has no
    within-segment variation in the training folds. Shared by both engines."""
    return np.linalg.lstsq(gram, cross, rcond=None)[0]


def shrink(sum_resid, n, prior):
    """Shrunken mean: (sum + SHRINK * prior) / (n + SHRINK). With n = 0 it returns the prior."""
    return (sum_resid + SHRINK * prior) / (n + SHRINK)


def market_ok(series):
    """Five-digit code other than 99999 (same rule as hmda_defs.market_code)."""
    import re
    pattern = re.compile(r"[0-9]{5}")
    return cat_apply(series, lambda t: bool(pattern.fullmatch(t)) and t != "99999", bool)

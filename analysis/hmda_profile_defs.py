"""Variables, percentiles and screens shared by the Spark and Pandas profiles."""
import operator

# variable: (histogram lower bound, upper bound, bin width); units are those of the HMDA field
NUMERIC = {
    "loan_amount": (0, 1_500_000, 50_000),  # HMDA reports $10,000-interval midpoints; 5 per bin
    "income": (0, 500, 10),
    "interest_rate": (2, 10, 0.125),
    "rate_spread": (-2, 4, 0.05),
    "loan_term": (0, 480, 12),
    "combined_loan_to_value_ratio": (0, 120, 2.5),
}
PCTS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
PCT_NAMES = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]
# Plausibility screens: counted and reported, never used to drop records.
CHECKS = {
    "loan_amount": [("nonpositive", "<=", 0), ("above_5m", ">", 5_000_000)],
    "income": [("negative", "<", 0), ("zero", "==", 0), ("above_10m", ">", 10_000)],
    "interest_rate": [("nonpositive", "<=", 0), ("above_15", ">", 15)],
    "rate_spread": [("below_minus_5", "<", -5), ("above_10", ">", 10)],
    "loan_term": [("nonpositive", "<=", 0), ("above_480", ">", 480)],
    "combined_loan_to_value_ratio": [("nonpositive", "<=", 0), ("above_150", ">", 150)],
}
# operator functions work on Spark Columns and NumPy arrays alike
OPS = {"<=": operator.le, "<": operator.lt, ">": operator.gt, "==": operator.eq}
STATUS_COLUMNS = list(NUMERIC) + ["debt_to_income_ratio", "county_code", "derived_msa_md"]
DTI_CATEGORIES = ["<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%"]

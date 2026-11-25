#!/usr/bin/env bash
set -eo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash reproduce.sh PATH_TO_HMDA_CSV [OUTPUT_PARENT]" >&2
    exit 2
fi
scripts="$(cd "$(dirname "$0")" && pwd)"
raw="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_parent="$scripts"
if [[ $# -eq 2 ]]; then output_parent="$2"; fi
mkdir -p "$output_parent"
run="$(mktemp -d "$(cd "$output_parent" && pwd)/hmda-rerun.XXXXXX")"

# Python with pyspark, pandas and pyarrow; Python with pandas and matplotlib for figures.
# Defaults point to the environment used for the archived run; override when needed.
if [[ -z "$SPARK_PYTHON" ]]; then SPARK_PYTHON=/opt/anaconda3/envs/hmda-spark/bin/python; fi
if [[ -z "$PLOT_PYTHON" ]]; then PLOT_PYTHON=/opt/anaconda3/bin/python; fi
if [[ -z "$JAVA_HOME" ]]; then JAVA_HOME=/opt/anaconda3/envs/hmda-spark/lib/jvm; fi
export JAVA_HOME
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_DIRS="$run/spark-tmp"
export MPLCONFIGDIR="$run/matplotlib"
mkdir -p "$SPARK_LOCAL_DIRS"
tables="$run/tables"; checks="$run/pandas"

# 1. Spark: staging, cohort tables, duplicate audit, benchmark
for phase in stage run; do
    "$SPARK_PYTHON" "$scripts/hmda_spark.py" --raw "$raw" --staged "$run/staged" --out "$tables" --phase "$phase"
done
# 2. Pandas: benchmark and grouped check, then the original agreement test
"$SPARK_PYTHON" "$scripts/hmda_pandas.py" --staged "$run/staged" --out "$checks" --benchmark-only
"$SPARK_PYTHON" "$scripts/hmda_pandas.py" --staged "$run/staged" --out "$checks" --benchmark-repeats 0
"$SPARK_PYTHON" "$scripts/hmda_verify.py" --spark "$tables" --pandas "$checks"
# 3. Descriptive profile in both engines
"$SPARK_PYTHON" "$scripts/hmda_profile_spark.py" --raw "$raw" --staged "$run/staged" --out "$tables"
"$SPARK_PYTHON" "$scripts/hmda_profile_pandas.py" --staged "$run/staged" --out "$checks"
# 4. Diagnostics in both engines
"$SPARK_PYTHON" "$scripts/hmda_diag_spark.py" --staged "$run/staged" --out "$tables"
"$SPARK_PYTHON" "$scripts/hmda_diag_pandas.py" --staged "$run/staged" --out "$checks"
"$SPARK_PYTHON" "$scripts/hmda_price_model.py" --staged "$run/staged" --out "$tables"
"$SPARK_PYTHON" "$scripts/hmda_price_model_pandas.py" --staged "$run/staged" --out "$checks"
# 5. Raw CSV read with PyArrow, independent of Spark staging
"$SPARK_PYTHON" "$scripts/hmda_raw_audit.py" --raw "$raw" --out "$checks" --part all
# 6. Original Pandas diagnostics on Spark tables, correctness table, follow-up checks
"$PLOT_PYTHON" "$scripts/hmda_results.py" --tables "$tables" --figures "$run/figures"
"$SPARK_PYTHON" "$scripts/hmda_verify_ext.py" --spark "$tables" --pandas "$checks"
"$SPARK_PYTHON" "$scripts/hmda_sensitivity.py" --tables "$tables"
# 7. Figures
"$PLOT_PYTHON" "$scripts/make_paper_figures.py" --tables "$tables" --out "$run/paper_figures"
"$PLOT_PYTHON" "$scripts/make_profile_figures.py" --tables "$tables" --out "$run/paper_figures"
# 8. Records removed by the exempt code 1111 (partial exemption)
"$SPARK_PYTHON" "$scripts/hmda_exempt_audit.py" --staged "$run/staged" --out "$tables"
# 9. Optional predictive extension, specified in predictive_plan.md before fitting
"$SPARK_PYTHON" "$scripts/hmda_predict_pandas.py" --raw "$raw" --out "$run/predict/pandas"
"$SPARK_PYTHON" "$scripts/hmda_predict_spark.py" --raw "$raw" --out "$run/predict/spark" --phase all
"$SPARK_PYTHON" "$scripts/hmda_predict_summary.py" --spark "$run/predict/spark" --pandas "$run/predict/pandas" \
    --tables "$tables" --out "$tables"
"$PLOT_PYTHON" "$scripts/make_predict_figures.py" --summary "$tables" --out "$run/paper_figures"
# 10. Pure pandas read_csv of the raw file (no PyArrow, no Spark staging) and state geography
"$SPARK_PYTHON" "$scripts/hmda_pandas_csv.py" --raw "$raw" --out "$checks"
"$PLOT_PYTHON" "$scripts/hmda_geography.py" --spark "$tables/segments.csv" \
    --pandas "$checks/segments_county_pandas_csv.parquet" \
    --pandas-waterfall "$checks/waterfall_pandas_csv.csv" --spark-waterfall "$tables/waterfall.csv" \
    --tables "$tables" --figures "$run/paper_figures" --slides "$run/slide_figures"
printf 'Results saved in %s\n' "$run"

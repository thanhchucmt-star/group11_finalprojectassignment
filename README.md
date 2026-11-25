# HMDA 2024

**Báo cáo** nằm tại **report/report.pdf**; thư mục **report** có nguồn LaTeX, hình và tài liệu tham khảo. **Mã nguồn và kết quả** nằm tại **analysis**.

## Dữ liệu

- Nguồn: FFIEC/CFPB, One-Year National Loan-Level Dataset, năm báo cáo 2024. Tải tại [ffiec.cfpb.gov](https://ffiec.cfpb.gov/data-publication/one-year-national-loan-level-dataset/2024), chọn bản CSV.
- Tệp dùng trong bài: `2024_public_lar_one_year_csv.csv`, 4.636.628.799 byte, 12.259.199 bản ghi, 99 cột. Mã đặc biệt của từng cột ghi trong `pandas/raw_schema_profile.csv` sau khi chạy.
- Kho mã không chứa tệp gốc. Kiểm tra tệp tải về bằng `shasum -a 256 2024_public_lar_one_year_csv.csv`; tệp đã dùng có SHA-256 `223da0f8d882436e95e03b3af3e41efa9cc1538683d4a912ab0b667d2ffa10aa`.

## Cài đặt

Cần Java 17 hoặc 21 (Spark 4 không chạy với Java 11) và Python 3.10 trở lên.

    conda create -n hmda-spark python=3.12 openjdk=17 -c conda-forge
    conda activate hmda-spark
    pip install -r analysis/requirements.txt

Hoặc dùng `venv` và một JDK cài sẵn; khi đó đặt `JAVA_HOME` trỏ tới JDK.

## Chạy lại

Từ thư mục **analysis**:

    SPARK_PYTHON=$(which python) PLOT_PYTHON=$(which python) JAVA_HOME=$JAVA_HOME \
        bash reproduce.sh /đường/dẫn/2024_public_lar_one_year_csv.csv

Chỉ chạy lại benchmark:

    python hmda_spark.py --raw RAW.csv --staged STAGED_DIR --out OUT/tables --phase benchmark
    python hmda_pandas.py --staged STAGED_DIR --out OUT/pandas --benchmark-only

trong đó `STAGED_DIR` là thư mục Parquet do bước `--phase stage` tạo. Benchmark của Pandas và Spark phải chạy trên cùng một máy. Trên Mac, `run_benchmark.command` (đặt cạnh thư mục dự án) làm đúng hai lệnh này, ghi trạng thái máy (nguồn điện, tải CPU) trước và sau khi đo, rồi tính `benchmark_table.csv`.

Chỉ chạy lại phần dự đoán (mục 9 của báo cáo):

    python hmda_predict_pandas.py --raw RAW.csv --out OUT/predict/pandas
    python hmda_predict_spark.py --raw RAW.csv --out OUT/predict/spark --phase all
    python hmda_predict_summary.py --spark OUT/predict/spark --pandas OUT/predict/pandas --tables OUT/tables --out OUT/tables

Kế hoạch phân tích được viết và băm trước khi chạy mô hình: `predictive_plan.md`, SHA-256 `f6352695a2632101ee1189b30f9d7b9dbb943c5045ad4f6a7dac369903b31285`. Mỗi lần chạy in lại mã băm này; `predictive_plan_log.md` ghi thứ tự các lần chạy và cách hiểu một số điểm của kế hoạch. Trên Mac, `run_predict.command` chạy ba lệnh trên.

Mỗi lần chạy `reproduce.sh` tạo một thư mục `hmda-rerun.*` mới gồm `tables` (Spark), `pandas` (Pandas và PyArrow), `figures`, `paper_figures`. Trên máy ít bộ nhớ, đặt thêm `HMDA_SPARK_MASTER="local[2]"`, `HMDA_DRIVER_MEMORY=2g`; mặc định là `local[6]` và `8g` như lượt chạy đã lưu. `HMDA_BENCH_REPEATS` đặt số lượt đo (mặc định 5, sau một lượt khởi động bị loại).

## Các bước và cặp triển khai

| Bước | Spark | Pandas / PyArrow | Đối chiếu |
|---|---|---|---|
| Đọc CSV, lọc cohort, đếm dòng lặp | `hmda_spark.py` (stage, analysis) | `hmda_raw_audit.py` (PyArrow đọc thẳng CSV) | `hmda_verify_ext.py` |
| Thống kê theo lender × county × sản phẩm | `hmda_spark.py` | `hmda_pandas.py` | `hmda_verify.py` |
| Giá trị thiếu, mã đặc biệt, thống kê mô tả, histogram | `hmda_profile_spark.py` | `hmda_profile_pandas.py` | `hmda_verify_ext.py` |
| So sánh giá trong segment, chuẩn hóa theo peers, bộ lọc, Spearman | `hmda_diag_spark.py` | `hmda_diag_pandas.py` (dùng hàm của `hmda_results.py`) | `hmda_verify_ext.py` |
| Mô hình xác suất tuyến tính có hiệu ứng cố định | `hmda_price_model.py` | `hmda_price_model_pandas.py` | `hmda_verify_ext.py` |
| Benchmark | `hmda_spark.py` (benchmark) | `hmda_pandas.py` | `hmda_verify.py` |
| Hồ sơ mã 1111 bị loại ở hai bộ lọc cuối | | `hmda_exempt_audit.py` | |
| Dự đoán rate spread (M0 đến M2), chênh phần dư | `hmda_predict_spark.py` | `hmda_predict_pandas.py` | `hmda_predict_summary.py` |
| Mô hình cây GBT (M3, kiểm tra phi tuyến) | `hmda_predict_spark.py` (chỉ Spark) | | |
| Đọc CSV gốc bằng `pandas.read_csv` thuần (không PyArrow), lọc cohort, segment theo quận | `hmda_spark.py` | `hmda_pandas_csv.py` | `hmda_geography.py` |
| Địa lý theo bang: tỷ lệ không nhận, tỷ trọng lender lớn nhất, chênh giá theo bang, bản đồ | `segments.csv` của Spark | segment của `hmda_pandas_csv.py` | `hmda_geography.py` (`geography_agreement.csv`) |

`hmda_results.py` tạo bảng kết quả chính từ bảng Spark; `hmda_sensitivity.py` chạy các kiểm tra bổ sung (mức tập trung dữ liệu thiếu theo lender, loại lender báo cáo bất thường, strata Unknown, phân phối hoán vị của số flag, cơ cấu case-mix trong các segment được ghép). Định nghĩa dùng chung cho phía Pandas nằm trong `hmda_defs.py` và `hmda_profile_defs.py`.

## Kết quả chính

- `tables/correctness_table.csv`: từng đại lượng, giá trị Pandas, giá trị Spark, mức khớp, ghi chú.
- `tables/engine_agreement.json`, `tables/benchmark_summary.csv`: đối chiếu và thời gian benchmark.
- `tables/profile_*`, `tables/schema_spark.json`, `pandas/raw_schema_profile.csv`: schema, giá trị thiếu, thống kê mô tả.
- `tables/exempt_audit.csv`: số hồ sơ, số lender và tỷ lệ không giải ngân của nhóm mang mã 1111 so với nhóm được giữ.
- `tables/predict_matched.csv`, `predict_metrics.csv`, `predict_agreement.csv`, `predict_decision.json`, `predict_coef_main.csv`: phần dự đoán (chênh thô và chênh phần dư theo từng phiên bản, chỉ số dự đoán ngoài fold, đối chiếu hai engine, kết luận theo quy tắc đặt trước, hệ số trung bình qua 05 fold).
- `tables/geography_state.csv`, `geography_summary.json`, `geography_agreement.csv`, `hmda_figures/geography_states.png`: tỷ lệ không nhận, tỷ trọng lender lớn nhất và chênh giá theo bang; đối chiếu pandas `read_csv` với Spark (khớp toàn bộ). `pandas/pandas_csv_meta.json`, `waterfall_pandas_csv.csv`: lượt đọc CSV bằng pandas thuần.
- `tables/sensitivity_summary.csv`, `tables/flag_null.csv`, `tables/lender_reporting_audit.csv`: kiểm tra bổ sung. `lender_reporting_lenders.csv` liệt kê LEI để kiểm tra nội bộ, không dùng để nêu tên tổ chức trong báo cáo.

Lượt chạy đã lưu dùng Python 3.12.16, Spark 4.3.0, Java 17, pandas 2.3.3 và PyArrow 26.0.1 trên Mac 8 lõi, 16 GiB RAM.

## Các lượt chạy

- `analysis/hmda_tables`, `analysis/hmda_pandas_checks`, `analysis/hmda_figures`: lượt chạy đầy đủ `reproduce.sh` trên Mac ngày 18/11/2026 (Python 3.12.16, Spark 4.3.0, Java 17, pandas 2.3.3, PyArrow 26.0.1, `local[6]`, heap 8 GB; tổng thời gian 11 phút 32 giây). Bảng đối chiếu `correctness_table.csv`: 104 đầu ra tính bằng cả hai engine, 90 khớp tuyệt đối, 14 khớp trong sai số tương đối 1e-9.
- Một lượt chạy lại khác trong máy ảo Linux (4 lõi, 3,9 GiB RAM, Spark 4.2.1, Java 21, `local[2]`, heap 2,2 GB) cho các bảng waterfall, missingness, coverage, cells trùng từng byte với lượt đã nộp ngày 09/11/2026.
- Benchmark được đo lại ngày 18/11/2026 bằng `run_benchmark.command` khi máy cắm sạc và không có người dùng (Spark nhanh hơn 1,36, 3,38 và 4,04 lần ở mức 25, 50 và 100%). Lượt đo trong lần chạy đầy đủ cho 1,55, 2,87 và 3,76 lần; báo cáo dùng lượt đo lại và nêu cả hai.
- Phần dự đoán: pandas chạy đủ trong máy ảo 3,9 GiB (20 giây cho bước ước lượng); Spark và M3 chạy trên Mac bằng `run_predict.command`.
- `cells.csv.gz` là bản nén của `cells.csv`; `hmda_results.py` đọc được cả hai.

## Hình cho slide

`make_slide_figures.py` (nhãn tiếng Việt) và `make_slide_figures_en.py` (nhãn tiếng Anh, dùng cho deck bảo vệ) vẽ lại các biểu đồ slide từ cùng bảng với báo cáo:

    python make_slide_figures_en.py --tables hmda_tables --bench hmda_tables --predict hmda_tables --out hmda_figures/slides_en

Bản đồ bang của slide 3.5 (`s_geo_map.png`) do `hmda_geography.py --slides hmda_figures/slides_en` vẽ.

Kết quả chạy dự đoán cuối cùng trên Mac nằm trong `predict_run/20261118-183723` (cạnh thư mục dự án), các bảng tóm tắt đã được chép vào `hmda_tables`.

## Nội dung bài nộp

| Thư mục / tệp | Nội dung |
|---|---|
| `report/report.pdf` | Báo cáo cuối kỳ (tiếng Anh, 8 trang nội dung, tài liệu tham khảo và phụ lục sau trang 8); `report.tex`, `references.bib`, `figures/` là mã nguồn LaTeX |
| `slides/HMDA_2024_defense_slides.pdf` | Slide bảo vệ (tiếng Anh, 17 slide chính và 02 slide dự phòng) |
| `slides/HMDA_2024_speaker_notes.md` | Lời thuyết trình theo từng slide, tiếng Anh kèm tóm tắt tiếng Việt |
| `analysis/` | Toàn bộ mã pandas và PySpark, `reproduce.sh`, notebook `HMDA_2024.ipynb` đã chạy sẵn kết quả, kế hoạch dự đoán `predictive_plan.md` và nhật ký `predictive_plan_log.md` |
| `analysis/hmda_tables/`, `analysis/hmda_pandas_checks/` | Bảng kết quả của Spark và của pandas dùng trong báo cáo và slide |
| `analysis/us_states.geojson` | Ranh giới bang (GeoJSON công khai, PublicaMundi MappingAPI) dùng cho bản đồ |
| `analysis/hmda_figures/` | Hình của báo cáo; `slides_en/` là biểu đồ của slide |
| `run_predict.command`, `run_benchmark.command` | Chạy phần dự đoán và benchmark trên Mac bằng cách nhấp đúp |

Dữ liệu gốc (`2024_public_lar_one_year_csv.csv`, 4,6 GB) không nằm trong gói; cách tải và mã SHA-256 để kiểm tra tệp ghi ở mục đầu của README này.


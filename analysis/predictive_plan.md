# Pre-specified plan: optional predictive extension (rate-spread model)

Written on 18 November 2026, before any model was fitted and before any residual was computed.
The SHA-256 of this file is printed at the start of every predictive run and stored with the
outputs, so the plan can be checked against the results. Changes after the first run must be
listed at the end of this file under "Deviations".

## 1. Purpose

Predict the rate spread that an origination would carry, given its lender, county and loan
product and the borrower and loan characteristics known at approval. Out-of-fold predictions
serve two uses:

1. Prediction quality: how well public HMDA fields predict the price of an origination.
2. Price beyond observables: whether approvals not accepted were priced above what their
   characteristics predict, inside the same lender, county and product. This measures how much
   of the within-segment gap reported in the main analysis (10.9 bp) observable characteristics
   account for. It also flags approvals priced above expectation, which fits the assignment's
   option of identifying unusual lending patterns.

## 2. Data and sample

- Input: the raw 2024 public LAR CSV, read independently by Spark and by PyArrow/pandas, with the
  eight cohort filters of the main analysis (2,867,757 approvals).
- Segment: lender (LEI) x county x loan type. Records without a valid five-digit county are
  excluded, as in the county view of the main analysis.
- Price: rate spread in percentage points, finite values only.
- Training records: originations (action 1) with a valid spread inside the 1st to 99th percentile
  of origination spreads (exact `percentile_disc`), in the training folds.
- Main comparison sample: complete cases, records whose core features (loan amount, income, DTI,
  CLTV) are all known.

## 3. Features

Core features, always used, entered as bins (dummy variables) so the effect of each variable
can be non-linear:

| Feature | Bins (edges) | Reference |
|---|---|---|
| Loan amount (USD) | 100k, 150k, 200k, 250k, 300k, 400k, 500k, 650k, 800k, 1M | 300000-<400000 |
| Income (USD 000) | 40, 60, 80, 100, 150, 200, 300 | 100-<150 |
| DTI (%) | 20, 30, 36, 40, 45, 50, 60 (HMDA range codes mapped as in the main analysis) | 36-<40 |
| CLTV (%) | 60, 70, 75, 80, 85, 90, 95, 97, 100 | 95-<97 |

Loan-term features, used only if they pass the reporting rule below:

| Feature | Categories | Reference |
|---|---|---|
| Loan term | 360, 180, other, Unknown | 360 |
| Introductory rate period | fixed (NA), ARM (> 0), Unknown | fixed |
| Conforming loan limit | C, NC, other | C |
| Non-amortizing features | no, yes, Unknown (any of interest-only, balloon, negative amortization, other = 1) | no |

Reporting rule: a loan-term feature is used only if the share of records with a missing,
NA or Exempt value differs by no more than 5 percentage points between originations and
approvals not accepted. NA for the introductory rate period means a fixed-rate loan and is not
counted as missing.

Excluded by design: fields that are part of the price package or set at closing (discount
points, lender credits, origination charges, total loan costs, total points and fees), the
interest rate, fields reported as NA for applications that did not close (initially payable to
the institution), and demographic fields and tract demographics.

## 4. Folds

Five folds. Each record's fold is CRC32 of its extracted raw fields, trimmed and joined with "|",
modulo 5. Every record, origination or not, belongs to one fold. A model trained without fold k
predicts every record in fold k.

## 5. Models

- M0: mean spread by loan type.
- M1: segment effect only. The segment effect is the training mean, shrunk toward the
  county x loan-type effect with weight n/(n + 20), which is in turn shrunk toward the loan-type
  effect with the same rule.
- M2: M1 plus the features of section 3. Coefficients are estimated within segments (fixed
  effects) by least squares from training records; segment effects are then computed and shrunk
  as in M1.
- M3 (sensitivity, Spark only): gradient-boosted trees on the training spread minus the M1
  segment effect, with the core features as numbers and the loan-term features as indicators;
  50 trees, depth 5, step size 0.1, 64 bins, seed 2024, no subsampling.

## 6. Prediction metrics

Out-of-fold, on originations inside the training trim range: root mean squared error and mean
absolute error in basis points, and R-squared against M0. Within-segment R-squared is reported
against M1.

## 7. Price beyond observables

For M2 the residual is the spread minus x'b of the record's fold model. Segment effects are
common to both outcomes in a segment, so they cancel in a within-segment comparison of medians.
For each segment with at least 10 valid residuals per outcome: gap = median residual of approvals
not accepted minus median residual of originations; segments are combined by a weighted mean,
with the smaller of the two counts as weight, as in the main analysis.

Reported numbers:

1. The main analysis gap (10.9 bp, all records).
2. The raw gap (spread itself) on the complete-case sample.
3. The residual gap on the same segments and records as (2).
4. Share accounted for = 1 - (3)/(2).

Uncertainty: 2,000 bootstrap resamples of segments, seed 2024, percentile 95% intervals for (2),
(3) and (4).

## 8. Decision rule for the conclusion

- The 95% interval of (3) excludes zero and (4) is below 50%: the gap persists after observable
  adjustment; Finding 1 stands as written.
- The 95% interval of (3) includes zero: the gap cannot be distinguished from differences in
  observable characteristics; Finding 1 is restated accordingly.
- Otherwise (interval excludes zero, (4) at least 50%): observable characteristics account for
  part of the gap; Finding 1 is qualified.

## 9. Sensitivity analyses

1. Unknown core features kept as their own category (no complete-case restriction).
2. Central 98% of spreads for all records.
3. Support: drop approvals not accepted in bins with fewer than 100 training originations within
   the loan type, or in segments with fewer than 20 training originations.
4. Residual gap by loan-amount band of the main analysis.
5. M3 in place of M2.

## 10. Engines and agreement

Extraction, folds, cross-products, coefficients, predictions, metrics and segment medians are
computed separately in Spark and in pandas from the raw CSV. Outputs are compared at relative
tolerance 1e-9. M3 runs in Spark only; its determinism is checked by a second run with the same
seed.

## Deviations

None so far.

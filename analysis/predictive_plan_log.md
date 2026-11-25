# Log for predictive_plan.md

The plan file is left unchanged so that its SHA-256 still matches the value printed in every run log:
`f6352695a2632101ee1189b30f9d7b9dbb943c5045ad4f6a7dac369903b31285` (written 18 November 2026, 10:04 UTC).

## Order of events

1. 10:04 UTC: plan written and hashed. No model had been fitted.
2. Code tests on synthetic data, then on the first 400,000 records of the raw CSV (pandas and Spark). The sample metrics were seen; no setting in the plan was changed.
3. The pandas script was rewritten to use categorical codes, because the first full run ran out of memory in a 3.9 GB virtual machine. Results on the synthetic test data were identical before and after the rewrite.
4. Full pandas run in the virtual machine (used only for checking code paths).
5. Final run on the Mac, 18 November 2026, 18:37 to 18:51 (UTC+7), folder `predict_run/20261118-183723`: pandas, then Spark (extract, M0 to M2, then M3), then the summary. The plan hash printed at the start matches. pandas metrics equal the virtual-machine run to 1e-12; all 37 compared outputs agree between engines; a refit of M3 fold 0 with the same seed gave identical trees (largest prediction difference 4.4e-16). The decision under section 8 is "persists".

## Interpretations of the plan

- Section 9, item 2 ("central 98% of spreads for all records"): the 1st and 99th percentiles of origination spreads, the training bounds of section 2, are applied to records of both outcomes.
- Section 9, item 3 (support): counts of training originations are taken over all five folds.
- Figures: the residual gap below USD 150,000 (15 segments) is computed and stored in `predict_matched.csv` but not drawn, as in the main analysis, because fixed fees dominate APR-based spreads of small loans.

## Deviations

None.

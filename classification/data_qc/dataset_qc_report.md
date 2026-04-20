# Liver + Spleen Dataset QC Report

- Dataset CSV: `artifacts/liver_spleen_dataset.csv`
- Total samples checked: `603`
- Missing/broken files: `0`
- Liver-spleen mask overlap cases: `0`
- Liver touching image border: `0`
- Spleen touching image border: `0`
- Left-right geometry violations: `0`

## Review Priority Counts

- `clean`: `531`
- `watch`: `53`
- `review`: `12`
- `urgent`: `7`

## Most Common QC Flags

- `flag_small_liver`: `7` cases
- `flag_large_liver`: `7` cases
- `flag_small_spleen`: `7` cases
- `flag_large_spleen`: `7` cases
- `flag_low_img_mean`: `7` cases
- `flag_high_img_mean`: `7` cases
- `flag_low_img_std`: `7` cases
- `flag_high_img_std`: `7` cases
- `flag_low_liver_mean`: `7` cases
- `flag_high_liver_mean`: `7` cases
- `flag_low_spleen_mean`: `7` cases
- `flag_high_spleen_mean`: `7` cases
- `flag_low_mean_ratio`: `7` cases
- `flag_high_mean_ratio`: `7` cases

## Interpretation

- No catastrophic dataset problems were found: all 603 image-mask pairs loaded successfully.
- Organ geometry is internally consistent across the entire dataset, which supports the current pairing and segmentation pipeline.
- The main QC risk is not broken data but tail cases: unusually small organ masks, unusually dark/bright images, and extreme liver-to-spleen attenuation ratios.
- These flagged cases should be reviewed manually before any further model optimization, because a relatively small number of outliers can destabilize training in a dataset of this size.

## Files Produced

- `dataset_qc_metrics.csv`: per-image QC metrics and flags
- `class_qc_summary.csv`: per-class descriptive summary
- `top_flagged_cases.csv`: highest-priority manual review list
- `flagged_case_previews.png`: preview grid for rapid visual screening
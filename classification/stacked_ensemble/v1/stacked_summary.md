# Stacked Ensemble Summary

- Masked OOF source: `classification/masked_efficientnetb0/v4_stabilized/oof_predictions.csv`
- Hybrid OOF source: `classification/hybrid_densenet_late_fusion/v2_class/oof_predictions.csv`
- Samples merged: `603`

## Best By AUC

- `simple_logreg_stack_prob`: AUC `0.9384`, Accuracy `0.8723`, F1 `0.8715`, fold mean AUC `0.9426 ± 0.0119`

## Best By Calibration

- `fold_weight_search_prob`: log loss `0.3201`, Brier `0.0982`, AUC `0.9378`

## Interpretation

- Stacking improves performance only if it preserves ranking between strong base predictors.
- Isotonic calibration improves probability quality but may slightly reduce AUC because calibration is not designed to improve ranking.
- For thesis reporting, use the highest-AUC stacked model if discrimination is the priority, and report the best-calibrated model separately if probability quality matters.
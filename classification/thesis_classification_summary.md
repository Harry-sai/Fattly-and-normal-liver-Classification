# Thesis Classification Summary

## Recommended Reporting

Use these four results in the thesis:

1. Primary headline model:
   `classification/hybrid_densenet_late_fusion/v2_class` with `weighted_blend_prob`
   This is the best overall system result and is the strongest final cross-validated classification score.

2. Best standalone deep model:
   `classification/masked_efficientnetb0/v2_audited` with `model_prob`
   This is the best single deep model without relying on a separate classical branch.

3. Best interpretable baseline:
   `classification/masked_efficientnetb0/v4_stabilized` with `logreg_prob`
   This demonstrates that liver-spleen handcrafted attenuation features alone already provide strong diagnostic signal.

4. Best audited/stable final masked pipeline:
   `classification/masked_efficientnetb0/v4_stabilized` with `blend_prob`
   This is the most thesis-safe masked system because it includes epoch-0 sanity checks, split-integrity reports, warmup scheduling, and evaluation TTA.

## Thesis-Ready Result Table

| Experiment Stage | Reported Output | Best Single-Fold AUC | Mean Fold AUC | Fold AUC SD | OOF AUC | Accuracy | Precision | Recall | F1 Score | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Classical baseline | `logreg_prob` | 0.9390 | 0.9293 | 0.0091 | 0.9257 | 0.8541 | 0.8577 | 0.8339 | 0.8456 | Interpretable liver-spleen attenuation-ratio baseline. |
| Early paired EfficientNet | single model | 0.9352 | 0.8835 | 0.0433 | NA | NA | NA | NA | NA | First strong paired deep baseline using liver ROI, spleen ROI, difference map, and limited attenuation stats. |
| Shared-context DenseNet | single model | 0.9427 | 0.8895 | 0.0318 | NA | NA | NA | NA | NA | Improved paired deep model after preserving liver-spleen joint context. |
| Hybrid late-fusion best overall | `weighted_blend_prob` | 0.9579 | 0.9404 | 0.0143 | 0.9360 | 0.8806 | 0.8729 | 0.8789 | 0.8759 | Best overall thesis headline system. |
| Masked EfficientNet best standalone | `model_prob` | 0.9555 | 0.9386 | 0.0107 | 0.9335 | 0.8640 | 0.8848 | 0.8235 | 0.8530 | Best standalone deep classifier with strict mask usage. |
| Masked EfficientNet stabilized | `blend_prob` | 0.9447 | 0.9342 | 0.0077 | 0.9329 | 0.8624 | 0.8652 | 0.8443 | 0.8546 | Best audited and most stable masked pipeline. |

## Gradual Experimental Progression

### Stage 1: Early paired deep baselines

Representative run:
`classification/efficientnetb0/efficientnet_b0_6`

What was done:
- Liver ROI, spleen ROI, and a difference map were used as input.
- Only a limited handcrafted attenuation feature subset was used.
- This stage tested whether paired organ comparison contained enough signal for classification.

What it showed:
- Mean best fold AUC reached about `0.8835`.
- The task was clearly learnable.
- However, fold variance was still relatively high, showing that the model was sensitive to crop design and training setup.

Why it matters:
- This stage established the feasibility of liver-spleen based fatty liver classification.

### Stage 2: Shared-context paired models

Representative run:
`classification/densenet121/4th_run_shared_context`

What was done:
- Instead of treating liver and spleen as fully separate resized crops, the model used a shared crop around both organs.
- This preserved relative anatomical context.

What it showed:
- Mean best fold AUC improved to about `0.8895`.
- This confirmed that preserving joint organ context is better than comparing independently resized crops alone.

Why it matters:
- It demonstrated that data representation was as important as backbone choice.

### Stage 3: Hybrid late-fusion modeling

Representative best run:
`classification/hybrid_densenet_late_fusion/v2_class`

What was done:
- Separate image branches were used for liver, spleen, and context.
- A tabular branch processed attenuation features.
- A late-fusion head combined image and feature representations.
- A weighted blend with the logistic-regression baseline was also evaluated.

What it showed:
- Best overall result:
  `OOF AUC = 0.9360`, `Accuracy = 0.8806`, `F1 = 0.8759`
- This is the best total-system result in the repository.

Why it matters:
- It showed that combining learned image features with explicit liver-spleen attenuation features gives the strongest final classification system.

### Stage 4: Strict segmentation-masked input design

Representative best standalone deep run:
`classification/masked_efficientnetb0/v2_audited`

What was done:
- One aligned union crop was used.
- Non-organ pixels were zeroed after mask resizing.
- The model saw masked context, liver-only pixels, and spleen-only pixels in aligned channels.
- Epoch-0 audits and split-integrity checks were added to verify that early high validation AUC was not due to leakage.

What it showed:
- Best standalone deep model:
  `OOF AUC = 0.9335`, `Accuracy = 0.8640`
- This model is extremely competitive with the hybrid system while remaining architecturally simpler.

Why it matters:
- It validated that segmentation was genuinely useful.
- It also produced the cleanest standalone deep-learning result for thesis discussion.

### Stage 5: Stabilized audited masked pipeline

Representative audited/stable run:
`classification/masked_efficientnetb0/v4_stabilized`

What was done:
- Deterministic seeding and worker initialization
- Epoch-0 sanity metrics
- Split-integrity reporting
- Short backbone warmup
- Evaluation TTA

What it showed:
- `blend_prob OOF AUC = 0.9329`
- `fold_mean_auc = 0.9342`
- `fold_std_auc = 0.0077`

Why it matters:
- Although its headline AUC is slightly below the best hybrid run, it is one of the most stable and auditable final pipelines in the project.
- This makes it very valuable for methodological discussion and for defending the robustness of the workflow.

## What The Main Values Signify

- **Best Single-Fold AUC**:
  The highest AUC achieved by one fold at its best epoch.
  Useful for understanding peak fold behavior, but not the main thesis metric.

- **Mean Fold AUC**:
  The average best-fold AUC across folds.
  Useful for showing how well the approach performs across repeated train/validation splits.

- **Fold AUC SD**:
  The standard deviation of fold AUCs.
  Lower values indicate greater stability and consistency.

- **OOF AUC**:
  The most important final metric for the thesis.
  This is computed from out-of-fold predictions on held-out cases across all folds.
  It is the most honest estimate of generalization.

- **Accuracy**:
  The percentage of correctly classified cases at the chosen threshold.
  Useful for intuitive interpretation, but threshold-dependent.

- **Precision**:
  Among predicted fatty-liver cases, the proportion that were truly fatty.
  Useful when false positives matter.

- **Recall**:
  Among actual fatty-liver cases, the proportion correctly detected.
  Useful when false negatives matter.

- **F1 Score**:
  Harmonic mean of precision and recall.
  Useful when both false positives and false negatives matter.

## How These Results Support The Project

- The strong **classical baseline** shows that liver-spleen attenuation relationships are diagnostically meaningful.
- The **shared-context deep models** show that spatially aligned paired-organ representation improves learning.
- The **hybrid model** proves that deep features and handcrafted attenuation features complement each other.
- The **masked EfficientNet** proves that segmentation-guided organ masking is useful rather than cosmetic.
- The **stabilized audited pipeline** strengthens the trustworthiness of the final methodology.

## Final Thesis Recommendation

Use this reporting structure:

- **Primary final result**:
  `classification/hybrid_densenet_late_fusion/v2_class` with `weighted_blend_prob`

- **Primary standalone deep-learning result**:
  `classification/masked_efficientnetb0/v2_audited` with `model_prob`

- **Primary interpretable baseline**:
  `logreg_prob`

- **Primary robustness/audit result**:
  `classification/masked_efficientnetb0/v4_stabilized` with `blend_prob`

This combination gives a strong thesis narrative:
baseline signal -> improved spatial representation -> multimodal fusion -> segmentation-guided masking -> audited stable final pipeline.

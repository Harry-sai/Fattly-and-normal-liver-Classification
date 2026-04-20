# Fatty Liver Classification from Abdominal CT: Thesis Methodology and Results

This repository archives the experimental development of an MTech thesis project on fatty liver classification using abdominal CT slices. The project evolved in two major phases:

1. A liver-only pipeline built on a manually curated 665-image AIIMS Delhi dataset with 100 manually fixed liver masks for segmentation development.
2. A later liver+spleen pipeline that added paired-organ masking, attenuation-ratio features, hybrid fusion models, and stronger evaluation controls.

The README below is reconstructed directly from the repository code, run configurations, metric files, QC artifacts, and file timestamps. It intentionally includes failed runs, unstable iterations, and artifact-version drift where those details are important for interpreting the final results.

## Chronological Overview

| Approx. date from file timestamps | Project stage | Evidence in repository |
|---|---|---|
| 2026-01-24 | Manual liver mask curation on the original liver-only pool | `data/all/manual_label_fixed` |
| 2025-12-17 to 2026-03-30 | Liver-only segmentation progression: Attention U-Net -> ResNet18 U-Net -> ResNet34 U-Net | `Attention/`, `resnet18/`, `resnet34/` |
| 2026-02-13 to 2026-02-27 | Liver-only classification progression: EfficientNet-B3 -> EfficientNet-B0 -> ResNet34 -> DenseNet121 | `results/efficientnet*`, `results/resnet34/*`, `results/densenet/*` |
| 2026-04-07 | Spleen annotation/segmentation phase begins | `Spleen_data/`, `spleen/*` |
| 2026-04-08 | Early paired-organ classification baselines | `classification/efficientnetb0/*`, `classification/densenet121/*` |
| 2026-04-11 | Hybrid late-fusion and audited masked pipelines | `classification/hybrid_densenet_late_fusion/*`, `classification/masked_efficientnetb0/*`, `classification/data_qc/*` |
| 2026-04-13 | Curated-data reruns of the best masked and hybrid systems | `classification/masked_efficientnetb0/v5_curated_data`, `classification/hybrid_densenet_late_fusion/v5_curated_images` |

## 1. Dataset Description

### 1.1 Source and cohort organization

The project brief identifies the source as an AIIMS Delhi abdominal CT dataset. The repository structure is consistent with CT slice-level classification rather than volumetric DICOM processing:

- all modelling code loads grayscale PNG/JPG slices rather than DICOM series;
- later scripts repeatedly describe the task as **abdominal CT** and rely on **liver-spleen attenuation** differences;
- no scanner metadata, acquisition parameters, or patient-level DICOM headers are preserved in the repository.

Therefore, the modality can be treated as **abdominal CT slice images**, but scanner metadata and acquisition protocol details are not recoverable from the archived files.

### 1.2 Original liver-only dataset

The original curated liver-only pool is preserved under `data/all/images`:

- `333` normal images
- `332` fatty liver images
- total `665` images

This is the class distribution requested in the thesis brief, and it is explicitly verifiable from the repository:

- `data/all/images/normal`: `333` files
- `data/all/images/fatty_liver`: `332` files

This 665-image pool is the basis of the early liver segmentation and liver-only classification work.

### 1.3 Later paired liver+spleen subsets

The repository contains multiple paired-organ snapshots, which is important for correctly interpreting later results.

Current image/mask roots:

- `data/images`: `310` normal + `287` fatty = `597`
- `data/masks_liver`: `310` normal + `287` fatty = `597`
- `data/masks_spleen`: `310` normal + `287` fatty = `597`

Current rebuilt paired dataset:

- `artifacts/liver_spleen_dataset.csv`: `597` matched samples
- class split: `310` normal, `287` fatty

Older paired snapshot still referenced by some artifacts:

- `artifacts/liver_spleen_features.csv`: `603` samples
- `classification/data_qc/dataset_qc_report.md`: also reports `603` checked samples

This means the repository contains **artifact-version drift**:

- the current dataset-building script matches `597` cases from the present `data/images` and mask folders;
- some April 2026 feature/QC/classification artifacts were generated from an earlier `603`-case snapshot.

For thesis reporting, this should be stated explicitly rather than hidden.

### 1.4 Manual curation and “best image” selection

The exact GUI used for slice selection is not stored, but the repository strongly indicates a manual curation stage before modelling:

- images are stored as curated slice files rather than full studies;
- `image_relabel.py` shows manual renumbering of curated PNG selections;
- the entire methodology assumes one diagnostically representative slice per case;
- later paired-organ pipelines rely on both liver and spleen being visible in the same slice.

From the repository evidence, the effective curation criteria were:

- select slices where the liver is clearly visible for liver-only segmentation/classification;
- later, select slices where **both liver and spleen are visible in the same CT slice** for paired-organ analysis;
- avoid grossly truncated organs, obvious failed masks, and unusable contrast/noise cases;
- preserve a near-balanced class distribution.

The exact selection protocol is not encoded in the codebase, so the above should be reported as a repository-supported inference rather than a directly logged annotation manual.

### 1.5 Annotation assets

The strongest explicit annotation evidence is:

- `data/all/manual_label_fixed`: `100` manually fixed liver masks (`50` normal, `50` fatty)

For spleen, the project brief states `100` manually generated spleen masks. The repository, however, now preserves a larger archived spleen segmentation set:

- `Spleen_data/images`: `71` normal + `71` fatty = `142`
- `Spleen_data/masks`: `71` normal + `71` fatty = `142`

Accordingly, the codebase appears to reflect an **expanded spleen annotation phase** beyond the initial 100-mask statement in the project description.

### 1.6 Dataset quality challenges

The code and QC reports show several practical issues:

- small effective dataset size for deep learning, especially once manual segmentation subsets are isolated;
- slice-selection noise: only one slice per case is used, making representativeness critical;
- heterogeneous file formats and naming (`.PNG`, `.JPG`, duplicate-style names such as `1 (2).PNG`);
- grayscale spleen masks mixed with binary pseudo masks, requiring safer thresholding;
- tail cases with unusually small/large organs, very dark/bright images, and extreme liver-to-spleen attenuation ratios.

The paired-organ QC script reported no catastrophic failures in the older 603-case snapshot, but it flagged outlier tails rather than structural corruption:

- `531` clean
- `53` watch
- `12` review
- `7` urgent

## 2. Preprocessing Pipeline

### 2.1 Image loading and intensity handling

All major scripts read grayscale PNG/JPG images with PIL or OpenCV. Preprocessing changed over time:

- `normalize.py` shows an early explicit min-max normalization experiment that rescaled each image to `[0,1]` and wrote 8-bit normalized outputs.
- Early liver-only classification scripts commonly normalized images after ROI cropping with `A.Normalize(...)`.
- Later liver+spleen scripts intentionally became more conservative because attenuation difference was treated as the key biomarker. The hybrid script explicitly states that it preserves **raw 8-bit intensity ordering** and avoids strong brightness/contrast augmentation.
- Some paired-organ scripts use robust percentile normalization inside the dataset class, typically clipping around low/high percentiles before resizing.

Rationale:

- aggressive appearance normalization can erase the very liver-vs-spleen attenuation contrast needed for fatty liver discrimination;
- later experiments therefore shifted toward geometry-safe augmentation and attenuation-preserving intensity handling.

### 2.2 Spatial preprocessing and resizing

The project used several task-specific image sizes:

- segmentation: `320`, `384`, and `512`
- liver-only classification: `224` and `512`
- paired-organ classification: `256`, `288`, and `320`

Spatial preprocessing evolved in a clear sequence:

1. **Liver-only classification**
   - crop the liver ROI from the liver mask bounding box;
   - resize the crop to the model input size;
   - classify the cropped liver patch.

2. **Early liver+spleen paired classification**
   - crop liver ROI and spleen ROI separately;
   - resize each independently to the same size;
   - construct a third channel as a signed liver-spleen difference image.

3. **Shared-context and masked paired classification**
   - compute a union box covering both organs;
   - preserve shared anatomical context in one aligned crop;
   - create liver-only, spleen-only, and context-masked views from the same spatial frame.

Rationale:

- separate organ crops let the model compare organs directly, but lose anatomy;
- shared union crops preserve relative spatial context;
- aligned masked channels make fusion more anatomically meaningful and easier to audit.

### 2.3 Data augmentation

Augmentation was extensive in segmentation and more conservative in later classification.

Representative segmentation augmentations:

- horizontal flips
- moderate rotations
- affine translation/scale
- brightness/contrast
- gamma variation
- Gaussian noise

Representative classification augmentations:

- liver-only models: resize, horizontal flip, small affine transforms, occasional mild intensity changes
- paired-organ hybrid/masked models: mostly geometry-safe transforms; brightness/contrast and mixup were intentionally reduced or removed in later runs

The later hybrid README comments are explicit: brightness/contrast augmentation was avoided because **attenuation difference is the core signal**.

### 2.4 Manual mask generation and storage format

Mask handling details are recoverable from code:

- liver manual masks are stored as grayscale PNG masks and binarized with thresholds such as `>127`;
- spleen masks later required more careful binarization because some “manual-like” spleen masks were grayscale organ cutouts rather than clean binaries;
- `Resnet34_hd95_spleen_pseudo_binary.py` and `Resnet34_hd95_spleen_semisupervised.py` introduce `MANUAL_MASK_FG_THRESHOLD = 10` to avoid the failure mode where background noise turns the whole image into spleen foreground.

Therefore, the annotation output format was image-based raster masking rather than polygon JSON or DICOM-RT structures.

The exact annotation software is **not identifiable from the repository**.

### 2.5 Quality control

The paired-organ QC pipeline is explicit and reproducible:

- `build_liver_spleen_dataset.py` enforces exact image/liver-mask/spleen-mask matching by file stem;
- `analyze_liver_spleen_dataset_qc.py` measures image statistics, organ size fractions, border touching, overlap, and left-right geometry;
- `extract_liver_spleen_features.py` computes organ attenuation distributions and ratio/difference features;
- preview grids and flagged-case lists were generated for manual review.

QC checks included:

- missing or broken files
- liver-spleen mask overlap
- organ touching image borders
- implausible left-right geometry
- extreme intensity or attenuation-ratio outliers

## 3. Methodology / Approach

### A. Liver-Only Pipeline

#### A.1 Segmentation

##### Stage A1.1 Attention U-Net

Primary archived run:

- `Attention/out_decay`

Configuration from `run_config.csv`:

- input size `320`
- batch size `16`
- `60` epochs
- learning rate `1e-5`
- optimizer: `AdamW`
- loss: `BCE`
- 5-fold cross-validation
- heavy geometric augmentation including random resized crop, rotation, elastic deformation, and shift-scale-rotate

Performance:

- mean validation Dice: `0.8247`
- mean validation IoU: `0.7217`

Interpretation:

- This established that the 100-mask liver subset was learnable.
- Performance was acceptable but not yet strong enough for reliable downstream masking.
- The architecture used explicit attention gating, but still lacked the stronger feature hierarchy later obtained from ResNet encoders.

##### Stage A1.2 ResNet18 U-Net

Key runs:

- unstable first run: `resnet18/firstrun`
- improved tuning run: `resnet18/check_graph`
- later partial final run: `resnet18/final`

Configuration pattern:

- input size `384`
- batch size `8`
- `80` epochs
- optimizer: SGD in the archived final run config
- loss: BCE + scheduled Dice
- decoder dropout around `0.1`

Performance progression:

| Run | Mean val Dice | Mean val IoU | Interpretation |
|---|---:|---:|---|
| `resnet18/firstrun` | `0.6753` | `0.5218` | early failure mode; folds 3-4 nearly collapsed despite high recall |
| `resnet18/check_graph` | `0.9073` | `0.8386` | strong recovery after tuning |
| `resnet18/final` | `0.9247` | `0.8640` | best archived ResNet18 metrics, but only 3 fold summaries are present |

Why earlier ResNet18 runs underperformed:

- the first run was overly recall-heavy and produced poor precision/IoU in some folds;
- the scheduled Dice formulation and tuning substantially stabilized overlap;
- nevertheless, the family remained more sensitive to setup than the later ResNet34 pipeline.

##### Stage A1.3 ResNet34 U-Net

This family represents the final liver segmentation direction.

Important archived runs:

- `resnet34/results9`: early weak baseline
- `resnet34/results14`: clear improvement on the manually curated setting
- `resnet34/out_final_run` and `resnet34/out_final_dr`: additional manual-label tuning
- `resnet34/hd95_higherepoch` and `resnet34/hd95_highepco_check`: boundary-aware refinement with HD95/ASD
- `resnet34/hd95_more_imgs`: later pseudo-label-expanded run using `predictions_fixed`

Observed progression:

| Run | Mean val Dice | Mean val IoU | Mean HD95 | Mean ASD | Interpretation |
|---|---:|---:|---:|---:|---|
| `resnet34/results9` | `0.7349` | `0.5939` | NA | NA | early ResNet34 setup still weak |
| `resnet34/results14` | `0.8644` | `0.7851` | NA | NA | strong improvement over early ResNet34 |
| `resnet34/out_final_run` | `0.8611` | `0.7720` | NA | NA | comparable late manual-label run |
| `resnet34/out_final_dr` | `0.8642` | `0.7765` | NA | NA | similar performance with dropout-related tuning |
| `resnet34/hd95_higherepoch` | `0.7902` | `0.7381` | `70.09` | `21.97` | unstable; one fold collapsed badly |
| `resnet34/hd95_highepco_check` | `0.9294` | `0.8746` | `44.40` | `11.15` | much stronger boundary-aware run |
| `resnet34/hd95_more_imgs` | `0.9761` | `0.9540` | `15.33` | `2.80` | best numeric result, but trained/evaluated on `predictions_fixed` pseudo labels rather than the 100 manual masks |

Architecture and training changes that mattered:

- deeper ResNet34 encoder than ResNet18
- decoder dropout
- scheduled Dice weighting in earlier runs
- later adoption of `DiceFocalLoss(sigmoid=True)`
- tracking of HD95 and ASD
- largest connected component post-processing in boundary-aware variants

Why ResNet34 was treated as the best liver segmentation family:

- it consistently outperformed Attention U-Net and the earliest ResNet18 runs;
- it supported boundary-aware evaluation, not just overlap;
- it scaled better to later pseudo-label/self-training style experiments.

Important caveat:

- the spectacular `hd95_more_imgs` scores are **not directly comparable** with the original 100 manually fixed liver-mask experiments, because the script points to `predictions_fixed` as labels. This is best interpreted as self-training consistency on pseudo labels, not as the core gold-standard thesis number.

For thesis discussion, the fairest narrative is:

- Attention U-Net proved feasibility,
- ResNet18 improved overlap,
- ResNet34 became the preferred segmentation backbone,
- later HD95/pseudo-label refinements explored boundary quality and scaling.

#### A.2 Liver-only classification

All liver-only classification scripts use transfer learning with small-sample safeguards. In the early scripts, the image backbone was usually frozen or mostly frozen, and the classifier head was retrained.

##### Stage A2.1 EfficientNet-B3

Representative runs:

- `results/efficientnet/lr`
- `results/efficientnet/best_thr`
- `results/efficientnet/more_imgbetter`

Design:

- liver ROI cropped from the liver mask
- resized to fixed size, usually `512`
- grayscale input
- BCE-based binary classification
- 5-fold stratified CV

Best archived result in this family:

- mean best-fold AUC: `0.9265`
- fold AUC SD: `0.0227`
- best fold AUC: `0.9658`

Why it mattered:

- this was the first strong liver-only deep baseline;
- it showed the task was learnable from liver appearance alone.

Why it was not the final liver-only choice:

- performance varied noticeably across retuned runs;
- the family was sensitive to image size, thresholding, and regularization;
- later ResNet34 and DenseNet121 runs matched or surpassed it more consistently.

##### Stage A2.2 EfficientNet-B0

Representative runs:

- `results/efficientnetB0/prev`
- `results/efficientnetB0/2nd`

Best archived result in this family:

- mean best-fold AUC: `0.8793`
- fold AUC SD: `0.0197`

Interpretation:

- EfficientNet-B0 reduced model size and training cost;
- it was useful as a lighter baseline;
- in this repository it did **not** outperform the best EfficientNet-B3 run.

This is an important negative result: simplifying the backbone improved efficiency, but not headline liver-only discrimination.

##### Stage A2.3 ResNet34

Representative runs:

- `results/resnet34/1st`
- `results/resnet34/backbone`
- `results/resnet34/decoder`

Best archived result:

- `results/resnet34/backbone`
- mean best-fold AUC: `0.9457`
- fold AUC SD: `0.0076`
- mean best-fold accuracy: `0.8812`

Why this stage was important:

- the ResNet34 liver-only classifier was noticeably stronger and more stable than the EfficientNet-B0 family;
- backbone fine-tuning strategy mattered, with the `backbone` run slightly outperforming `decoder`.

##### Stage A2.4 DenseNet121

Representative runs:

- `results/densenet/img512`
- `results/densenet/new_arci`
- `results/densenet/Final_run`

Best archived result in this family:

- `results/densenet/new_arci`
- mean best-fold AUC: `0.9577`
- fold AUC SD: `0.0265`
- best fold AUC: `0.9822`

However, later DenseNet runs were not uniformly better:

- `results/densenet/Final_run` dropped to mean best-fold AUC `0.9294`

Interpretation:

- DenseNet121 became the **final liver-only classification architecture family**, because its best run delivered the strongest liver-only AUC in the repository;
- at the same time, the later regression in `Final_run` shows the liver-only setting remained hyperparameter-sensitive.

This should be reported honestly: the final chosen family was DenseNet121, but not every later DenseNet retune improved on the earlier `new_arci` run.

### B. Liver + Spleen Pipeline

### B.1 Spleen segmentation and integration with liver masks

The spleen phase began after a new archived dataset appeared in `Spleen_data`:

- `71` normal images
- `71` fatty-liver images
- total `142`

This stage reused the ResNet34 U-Net style and progressively introduced better loss functions, safer mask binarization, pseudo-label handling, and semi-supervised training.

#### Stage B1.1 First spleen run: complete failure

Run:

- `spleen/1st_run`

Result:

- mean Dice `0.0000`

Interpretation:

- the model effectively predicted empty masks;
- this is the clearest failed segmentation experiment in the repository.

#### Stage B1.2 Manual spleen segmentation recovery

Run:

- `spleen/2nd_run`

Key changes:

- ResNet34 U-Net
- `DiceFocalLoss`
- HD95 and ASD tracking
- validation previews

Result:

- mean Dice `0.9421`
- mean IoU `0.8961`
- mean HD95 `7.83`
- mean ASD `2.09`

This is the first successful spleen segmentation baseline.

#### Stage B1.3 Pseudo-label-aware and semi-supervised refinement

Important runs:

- `spleen/3rd_run_peseudo`
- `spleen/4th_run_pseudo_binary`
- `spleen/5th_run_pseudo_tuned`
- `spleen/6th_run_semisupervised`
- `spleen/7th_run_semisupervised_tuned`

Key methodological refinements visible in code and run configs:

- mixed manual grayscale masks and binary pseudo masks required robust thresholding (`MANUAL_MASK_FG_THRESHOLD = 10`);
- pseudo masks were never allowed into validation in the semi-supervised script;
- pseudo samples were down-weighted in training via `PSEUDO_SAMPLE_WEIGHT` and `PSEUDO_LOSS_WEIGHT`;
- weighted sampling was used so pseudo labels acted as support, not as gold standard.

Performance summary:

| Run | Mean Dice | Mean IoU | Mean HD95 | Mean ASD | Interpretation |
|---|---:|---:|---:|---:|---|
| `spleen/1st_run` | `0.0000` | `0.0000` | NA | NA | collapse to empty predictions |
| `spleen/2nd_run` | `0.9421` | `0.8961` | `7.83` | `2.09` | first successful spleen baseline |
| `spleen/3rd_run_peseudo` | `0.9598` | `0.9247` | `11.09` | `2.75` | strongest overlap result |
| `spleen/4th_run_pseudo_binary` | `0.9223` | `0.8802` | `32.36` | `7.35` | mixed-mask handling still unstable |
| `spleen/5th_run_pseudo_tuned` | `0.9320` | `0.8854` | `14.34` | `4.66` | safer grayscale/binary handling |
| `spleen/6th_run_semisupervised` | `0.9460` | `0.9033` | `10.50` | `2.18` | manual-only validation with weighted pseudo support |
| `spleen/7th_run_semisupervised_tuned` | `0.9455` | `0.9047` | `16.98` | `4.08` | stronger pseudo weighting, not better than 6th on boundary metrics |

Best interpretation:

- numerically, `3rd_run_peseudo` has the highest Dice;
- methodologically, `6th_run_semisupervised` is the most defensible spleen result because pseudo masks were explicitly restricted to training and down-weighted.

### B.2 Paired-organ classification

This phase moved from simple organ comparison to multimodal fusion.

#### Stage B2.1 Early paired EfficientNet-B0

Representative run:

- `classification/efficientnetb0/efficientnet_b0_6`

Input design:

- liver ROI
- spleen ROI
- signed liver-spleen difference map
- limited attenuation-stat feature vector concatenated before classification

Key config:

- input size `320`
- batch size `6`
- `45` epochs
- AdamW + ReduceLROnPlateau
- label smoothing and occasional mixup

Performance:

- mean best-fold AUC: `0.8835`
- fold AUC SD: `0.0433`
- best fold AUC: `0.9352`

Interpretation:

- this established that paired-organ comparison carried real signal;
- however, variance across folds remained high.

#### Stage B2.2 Shared-context DenseNet121

Representative run:

- `classification/densenet121/4th_run_shared_context`

Key design change from the run config:

- replaced independently resized liver/spleen crops with a **shared union crop**
- channels became:
  - full shared crop
  - liver-masked shared crop
  - spleen-masked shared crop

Performance:

- mean best-fold AUC: `0.8895`
- fold AUC SD: `0.0318`
- best fold AUC: `0.9427`

Interpretation:

- performance improved over the early paired EfficientNet;
- preserving shared anatomy was more valuable than simply comparing isolated resized ROIs.

#### Stage B2.3 Hybrid DenseNet late-fusion models

Headline runs:

- `classification/hybrid_densenet_late_fusion/v1`
- `classification/hybrid_densenet_late_fusion/v2_class`
- `classification/hybrid_densenet_late_fusion/v3_masked_organs`
- `classification/hybrid_densenet_late_fusion/v4_auc_accuracy`
- later curated rerun: `classification/hybrid_densenet_late_fusion/v5_curated_images`

Architecture from `train_liver_spleen_classifier_hybrid.py`:

- separate masked liver crop, masked spleen crop, and masked union-context crop
- shared DenseNet121 image encoder
- dedicated tabular branch for attenuation features
- late fusion head
- auxiliary losses for tabular and image branches (`aux_tab_weight`, `aux_image_weight`)
- logistic regression baseline fitted on the same tabular features
- weighted blending between hybrid model and logistic-regression predictions

Best thesis headline run:

- `classification/hybrid_densenet_late_fusion/v2_class`
- weighted blend OOF AUC `0.9360`
- accuracy `0.8806`
- precision `0.8729`
- recall `0.8789`
- F1 `0.8759`
- fold mean AUC `0.9404 +- 0.0143`

Why `v2_class` is the best overall paired-organ system:

- it is explicitly recommended in `classification/thesis_classification_summary.md`;
- it gives the highest overall OOF AUC among the main thesis candidates;
- it demonstrates complementarity between learned image features and attenuation statistics.

Important follow-up observations:

- `v3_masked_organs` kept strong AUC but recall dropped in the weighted blend;
- `v4_auc_accuracy` improved auditability/threshold handling but not headline AUC;
- `v5_curated_images` on the curated snapshot reached OOF AUC `0.9349`, close to `v2_class`, but was not adopted as the thesis headline.

#### Stage B2.4 Masked EfficientNet-B0

This branch is separate from the DenseNet hybrid but becomes central in the final thesis narrative.

Main idea from `train_liver_spleen_classifier_masked_efficientnetb0.py`:

- one aligned union crop
- channel 1: union-masked context
- channel 2: liver-only pixels
- channel 3: spleen-only pixels
- non-organ pixels zeroed after nearest-neighbor mask resizing
- strong fold-wise logistic-regression baseline retained for comparison

Important runs:

- `classification/masked_efficientnetb0/v1`
- `classification/masked_efficientnetb0/v2_audited`
- `classification/masked_efficientnetb0/v3_tuned_blend`
- `classification/masked_efficientnetb0/v4_stabilized`
- curated rerun: `classification/masked_efficientnetb0/v5_curated_data`

Best standalone deep result:

- `classification/masked_efficientnetb0/v2_audited`
- `model_prob` OOF AUC `0.9335`
- accuracy `0.8640`
- precision `0.8848`
- recall `0.8235`
- F1 `0.8530`

Most stable audited pipeline:

- `classification/masked_efficientnetb0/v4_stabilized`
- `blend_prob` OOF AUC `0.9329`
- accuracy `0.8624`
- F1 `0.8546`
- fold mean AUC `0.9342 +- 0.0077`

Late curated rerun:

- `classification/masked_efficientnetb0/v5_curated_data`
- `blend_prob` OOF AUC `0.9360`
- accuracy `0.8760`
- F1 `0.8706`

Interpretation:

- the masked EfficientNet branch demonstrated that segmentation-guided input design itself was useful;
- it also provided the cleanest audited standalone deep-learning result, even when the hybrid model remained the best total system.

## 4. Results and Discussion

### 4.1 Segmentation results

#### Liver segmentation summary

| Family / run | Mean Dice | Mean IoU | Notes |
|---|---:|---:|---|
| Attention U-Net `Attention/out_decay` | `0.8247` | `0.7217` | first workable liver baseline |
| ResNet18 `resnet18/firstrun` | `0.6753` | `0.5218` | unstable early run |
| ResNet18 `resnet18/check_graph` | `0.9073` | `0.8386` | major recovery after tuning |
| ResNet18 `resnet18/final` | `0.9247` | `0.8640` | strongest archived ResNet18 metrics, but only 3 folds logged |
| ResNet34 `resnet34/results14` | `0.8644` | `0.7851` | strongest clearly comparable manual-label ResNet34 run |
| ResNet34 `resnet34/hd95_highepco_check` | `0.9294` | `0.8746` | boundary-aware refinement |
| ResNet34 `resnet34/hd95_more_imgs` | `0.9761` | `0.9540` | pseudo-label-expanded, not directly gold-standard comparable |

#### Spleen segmentation summary

| Run | Mean Dice | Mean IoU | Mean HD95 | Notes |
|---|---:|---:|---:|---|
| `spleen/1st_run` | `0.0000` | `0.0000` | failed all-zero prediction stage |
| `spleen/2nd_run` | `0.9421` | `0.8961` | first strong manual spleen baseline |
| `spleen/3rd_run_peseudo` | `0.9598` | `0.9247` | strongest overlap score |
| `spleen/6th_run_semisupervised` | `0.9460` | `0.9033` | best methodologically controlled semi-supervised run |
| `spleen/7th_run_semisupervised_tuned` | `0.9455` | `0.9047` | stronger pseudo weighting did not improve headline performance |

### 4.2 Classification results

#### Liver-only progression

| Stage | Representative run | Mean best-fold AUC | Fold SD | Comment |
|---|---|---:|---:|---|
| EfficientNet-B3 | `results/efficientnet/lr` | `0.9265` | `0.0227` | first strong liver-only deep baseline |
| EfficientNet-B0 | `results/efficientnetB0/2nd` | `0.8793` | `0.0197` | lighter but weaker |
| ResNet34 | `results/resnet34/backbone` | `0.9457` | `0.0076` | better stability and stronger discrimination |
| DenseNet121 | `results/densenet/new_arci` | `0.9577` | `0.0265` | strongest liver-only archived result |

#### Paired-organ thesis result table

This table aligns with `classification/thesis_classification_summary.md` and `classification/thesis_classification_results.csv`.

| Experiment stage | Run / output | Best single-fold AUC | Mean fold AUC | Fold SD | OOF AUC | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical baseline | `logreg_prob` from masked pipeline | `0.9390` | `0.9293` | `0.0091` | `0.9257` | `0.8541` | `0.8577` | `0.8339` | `0.8456` |
| Early paired EfficientNet | `classification/efficientnetb0/efficientnet_b0_6` | `0.9352` | `0.8835` | `0.0433` | NA | NA | NA | NA | NA |
| Shared-context DenseNet | `classification/densenet121/4th_run_shared_context` | `0.9427` | `0.8895` | `0.0318` | NA | NA | NA | NA | NA |
| Hybrid late-fusion best overall | `classification/hybrid_densenet_late_fusion/v2_class`, `weighted_blend_prob` | `0.9579` | `0.9404` | `0.0143` | `0.9360` | `0.8806` | `0.8729` | `0.8789` | `0.8759` |
| Masked EfficientNet best standalone | `classification/masked_efficientnetb0/v2_audited`, `model_prob` | `0.9555` | `0.9386` | `0.0107` | `0.9335` | `0.8640` | `0.8848` | `0.8235` | `0.8530` |
| Masked EfficientNet stabilized | `classification/masked_efficientnetb0/v4_stabilized`, `blend_prob` | `0.9447` | `0.9342` | `0.0077` | `0.9329` | `0.8624` | `0.8652` | `0.8443` | `0.8546` |

### 4.3 Key findings

1. **The task was learnable from liver appearance alone, but the performance ceiling was higher with liver+spleen comparison.**
   - Liver-only models reached strong AUC values, especially with ResNet34 and DenseNet121.
   - The paired-organ systems were more clinically aligned because they modeled attenuation relative to spleen rather than in isolation.

2. **Segmentation quality mattered, but not all high Dice numbers were equally trustworthy.**
   - Manual-label liver segmentation improved from Attention U-Net to ResNet families.
   - Later pseudo-label-expanded liver runs produced extremely high overlap, but these are not fair substitutes for gold-standard manual-label validation.

3. **The spleen pipeline had a genuine failure-recovery trajectory.**
   - The first spleen run completely failed.
   - Subsequent runs succeeded only after better overlap loss, safer mask binarization, and stricter pseudo-label handling.

4. **Data representation was as important as backbone choice.**
   - Early paired models used separate liver/spleen crops and a difference map.
   - Shared-context crops improved performance by preserving anatomy.
   - Aligned masked union crops improved interpretability and auditability.

5. **Handcrafted attenuation features were genuinely informative.**
   - In the older 603-case feature snapshot, normal cases had mean liver-to-spleen ratio around `0.977`, while fatty-liver cases averaged around `0.892`.
   - The logistic-regression baseline remained competitive, proving that image models were adding to an already strong clinical signal rather than inventing it.

### 4.4 Limitations and failure cases

- **Small-data regime**: all deep models operate close to the overfitting boundary.
- **Artifact-version drift**: some later QC/features/classification results come from a 603-case paired snapshot, whereas the current dataset builder yields 597 matched cases.
- **Slice-selection dependence**: one representative slice per case makes the pipeline sensitive to manual curation.
- **Incomplete metadata**: no DICOM or scanner metadata survive in the repository.
- **Segmentation dependency**: later masked and hybrid pipelines assume organ masks are reliable.
- **Pseudo-label caveat**: some liver and spleen segmentation improvements depend on pseudo masks and should not be presented as pure manual-label generalization.

### 4.5 Final thesis reporting recommendation

The repository itself already contains a clear recommendation in `classification/thesis_classification_summary.md`, and it is appropriate:

- **Primary overall result**: `classification/hybrid_densenet_late_fusion/v2_class` using `weighted_blend_prob`
- **Primary standalone deep result**: `classification/masked_efficientnetb0/v2_audited` using `model_prob`
- **Primary interpretable baseline**: logistic regression on liver+spleen attenuation features
- **Primary robustness result**: `classification/masked_efficientnetb0/v4_stabilized` using `blend_prob`

This gives the cleanest thesis narrative:

- curated CT slice selection
- manual liver masking and liver-only segmentation/classification
- extension to spleen segmentation
- paired-organ deep baselines
- hybrid late fusion
- segmentation-guided masked classification with explicit audit controls

## Repository Notes

- `data/all/*` preserves the original 665-image liver-focused phase.
- `data/images`, `data/masks_liver`, and `data/masks_spleen` preserve the later paired-organ phase.
- `classification/data_qc/*` and `artifacts/*` contain the strongest reproducibility artifacts for the final paired-organ work.
- GAN and diffusion scripts are present in the repository, but thesis notes in `thesis/extra_details.md` explicitly argue against making synthetic augmentation a core methodological contribution because it may distort clinically meaningful liver-spleen attenuation relationships.

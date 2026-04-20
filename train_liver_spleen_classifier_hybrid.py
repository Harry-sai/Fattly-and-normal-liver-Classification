import argparse
import os
import random
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import albumentations as A
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_COMMENT = """Hybrid liver+spleen classifier for small-data abdominal CT.
Key design choices:
- use segmentation-masked liver crop, segmentation-masked spleen crop, and union-masked context crop
- preserve raw 8-bit intensity ordering instead of per-image histogram re-scaling
- avoid brightness/contrast augmentation and mixup because attenuation difference is the core signal
- use a shared DenseNet encoder with late fusion, plus a strong tabular attenuation branch
- run a fold-wise logistic-regression baseline from the same feature table for direct comparison
- use class-aware derived groups when no true patient/study id is available
- plot AUC with ROC/Youden-threshold accuracy instead of loss, including an epoch-0 sanity point
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a hybrid liver+spleen classifier.")
    parser.add_argument("--dataset-csv", default="artifacts/liver_spleen_dataset.csv")
    parser.add_argument("--features-csv", default="artifacts/liver_spleen_features.csv")
    parser.add_argument("--results-dir", default="classification/hybrid_densenet_late_fusion/v1")
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.18)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--aux-tab-weight", type=float, default=0.35)
    parser.add_argument("--aux-image-weight", type=float, default=0.15)
    parser.add_argument("--debug-max", type=int, default=8)
    parser.add_argument("--organ-pad-ratio", type=float, default=0.18)
    parser.add_argument("--context-pad-ratio", type=float, default=0.12)
    parser.add_argument("--model-blend-weight", type=float, default=0.40)
    parser.add_argument(
        "--pretrained-weights",
        choices=("imagenet", "random"),
        default="imagenet",
        help="Use random to sanity-check whether high early validation AUC is coming from pretrained image features.",
    )
    parser.add_argument(
        "--record-epoch-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record deterministic train/validation metrics before any optimization step.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grayscale(path):
    return np.array(Image.open(path).convert("L"))


def load_mask(path, target_shape):
    mask = np.array(Image.open(path).convert("L"))
    if mask.shape != target_shape:
        mask = np.array(
            Image.fromarray(mask).resize(
                (target_shape[1], target_shape[0]),
                resample=Image.NEAREST,
            )
        )
    return (mask > 0).astype(np.uint8)


def safe_square_box(y1, y2, x1, x2, height, width, pad_ratio):
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1
    side = int(round(max(box_h, box_w) * (1.0 + 2.0 * pad_ratio)))
    side = max(32, min(side, min(height, width)))

    cy = 0.5 * (y1 + y2)
    cx = 0.5 * (x1 + x2)
    top = int(round(cy - side / 2))
    left = int(round(cx - side / 2))
    top = max(0, min(top, height - side))
    left = max(0, min(left, width - side))
    return top, top + side, left, left + side


def box_from_mask(mask, pad_ratio):
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        side = min(h, w)
        top = max(0, (h - side) // 2)
        left = max(0, (w - side) // 2)
        return top, top + side, left, left + side
    return safe_square_box(ys.min(), ys.max(), xs.min(), xs.max(), h, w, pad_ratio)


def box_from_union(mask_a, mask_b, pad_ratio):
    union = ((mask_a > 0) | (mask_b > 0)).astype(np.uint8)
    return box_from_mask(union, pad_ratio)


def crop_box(arr, box):
    top, bottom, left, right = box
    return arr[top:bottom, left:right]


def organ_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
        }

    return {
        f"{prefix}_count": float(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def build_feature_table(dataset_df):
    rows = []
    eps = 1e-6
    for row in dataset_df.itertuples(index=False):
        image = load_grayscale(row.image_path).astype(np.float32)
        liver_mask = load_mask(row.liver_mask_path, image.shape)
        spleen_mask = load_mask(row.spleen_mask_path, image.shape)

        liver_values = image[liver_mask > 0]
        spleen_values = image[spleen_mask > 0]

        features = {
            "image_id": row.image_id,
            "class_name": row.class_name,
            "label": int(row.label),
        }
        features.update(organ_stats(liver_values, "liver"))
        features.update(organ_stats(spleen_values, "spleen"))

        liver_mean = features["liver_mean"]
        spleen_mean = features["spleen_mean"]
        liver_median = features["liver_median"]
        spleen_median = features["spleen_median"]

        features.update(
            {
                "mean_ratio": float(liver_mean / (spleen_mean + eps)),
                "median_ratio": float(liver_median / (spleen_median + eps)),
                "mean_diff": float(liver_mean - spleen_mean),
                "median_diff": float(liver_median - spleen_median),
                "std_ratio": float(features["liver_std"] / (features["spleen_std"] + eps)),
                "count_ratio": float(features["liver_count"] / (features["spleen_count"] + eps)),
                "p25_diff": float(features["liver_p25"] - features["spleen_p25"]),
                "p50_diff": float(features["liver_p50"] - features["spleen_p50"]),
                "p75_diff": float(features["liver_p75"] - features["spleen_p75"]),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def load_dataset_table(dataset_csv):
    dataset_df = pd.read_csv(dataset_csv)
    required = {
        "image_id",
        "class_name",
        "label",
        "image_path",
        "liver_mask_path",
        "spleen_mask_path",
    }
    missing = required.difference(dataset_df.columns)
    if missing:
        raise RuntimeError(f"Dataset CSV is missing columns: {sorted(missing)}")
    return dataset_df


def load_or_build_feature_table(dataset_df, features_csv, results_dir):
    feature_path = Path(features_csv)
    feature_df = None
    if feature_path.exists():
        candidate = pd.read_csv(feature_path)
        required = {"image_id", "class_name", "label"}
        if required.issubset(candidate.columns):
            feature_df = candidate

    if feature_df is None:
        feature_df = build_feature_table(dataset_df)
        generated_path = Path(results_dir) / "generated_features.csv"
        feature_df.to_csv(generated_path, index=False)
        print(f"Features were rebuilt and saved to {generated_path}")

    merged = dataset_df.merge(
        feature_df,
        on=["image_id", "class_name", "label"],
        how="left",
        validate="one_to_one",
    )
    feature_cols = []
    meta_cols = {
        "image_id",
        "class_name",
        "label",
        "image_path",
        "liver_mask_path",
        "spleen_mask_path",
        "patient_id",
        "study_id",
        "group_id",
    }
    for col in merged.columns:
        if col in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(merged[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("No numeric tabular feature columns were found.")
    if merged[feature_cols].isnull().any().any():
        missing_rows = merged[merged[feature_cols].isnull().any(axis=1)][["image_id", "class_name"]]
        raise RuntimeError(
            "Feature table does not fully cover the dataset. Missing rows:\n"
            + missing_rows.head(10).to_string(index=False)
        )

    feature_report = (
        merged.groupby("class_name")[["mean_ratio", "mean_diff", "median_ratio", "median_diff"]]
        .agg(["mean", "std"])
        .round(3)
    )
    feature_report.to_csv(Path(results_dir) / "feature_signal_summary.csv")
    return merged, feature_cols


def build_groups(dataset_df):
    for col in ("group_id", "patient_id", "study_id"):
        if col in dataset_df.columns:
            return dataset_df[col].fillna("missing").astype(str), col
    base_id = dataset_df["image_id"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
    derived = dataset_df["class_name"].astype(str) + "::" + base_id
    return derived, "derived_class_image_group"


def make_splits(dataset_df, folds, seed):
    labels = dataset_df["label"].to_numpy()
    groups, group_source = build_groups(dataset_df)
    use_group_split = (
        StratifiedGroupKFold is not None
        and groups.nunique() < len(groups)
        and groups.nunique() >= folds
    )

    if use_group_split:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(dataset_df, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(dataset_df, labels))
    return splits, groups, use_group_split, group_source


def derived_class_image_groups(dataset_df):
    base_id = dataset_df["image_id"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
    return dataset_df["class_name"].astype(str) + "::" + base_id


def class_image_ids(dataset_df):
    return dataset_df["class_name"].astype(str) + "::" + dataset_df["image_id"].astype(str)


def save_split_integrity_report(train_df, val_df, fold, results_dir):
    checks = [
        (
            "image_id_only_overlap_info",
            set(train_df["image_id"].astype(str)).intersection(set(val_df["image_id"].astype(str))),
            False,
        ),
        (
            "class_image_id_overlap",
            set(class_image_ids(train_df)).intersection(set(class_image_ids(val_df))),
            True,
        ),
        (
            "image_path_overlap",
            set(train_df["image_path"].astype(str)).intersection(set(val_df["image_path"].astype(str))),
            True,
        ),
        (
            "derived_class_image_group_overlap",
            set(derived_class_image_groups(train_df)).intersection(set(derived_class_image_groups(val_df))),
            True,
        ),
    ]
    for col in ("group_id", "patient_id", "study_id"):
        if col in train_df.columns and col in val_df.columns:
            checks.append(
                (
                    f"{col}_overlap",
                    set(train_df[col].fillna("missing").astype(str)).intersection(
                        set(val_df[col].fillna("missing").astype(str))
                    ),
                    True,
                )
            )

    report_rows = []
    for check_name, overlap, warns_on_overlap in checks:
        examples = sorted(overlap)[:10]
        report_rows.append(
            {
                "fold": fold,
                "check": check_name,
                "warns_on_overlap": warns_on_overlap,
                "overlap_count": len(overlap),
                "example_values": "; ".join(examples),
            }
        )
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(Path(results_dir) / f"split_integrity_fold_{fold}.csv", index=False)

    nonzero = report_df[(report_df["overlap_count"] > 0) & (report_df["warns_on_overlap"])]
    if nonzero.empty:
        image_id_info = int(
            report_df.loc[report_df["check"] == "image_id_only_overlap_info", "overlap_count"].iloc[0]
        )
        print(
            f"Fold {fold} split integrity | no leakage-style train/validation overlap detected "
            f"(image_id-only repeats={image_id_info})"
        )
    else:
        compact = ", ".join(
            f"{row.check}={row.overlap_count}" for row in nonzero.itertuples(index=False)
        )
        print(f"Fold {fold} split integrity warning | {compact}")


def build_transforms(img_size):
    additional_targets = {"image_liver": "image", "image_spleen": "image"}
    train_tfms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.04,
                scale=(0.96, 1.04),
                rotate=(-6, 6),
                border_mode=0,
                p=0.35,
            ),
            A.GaussNoise(std_range=(0.002, 0.01), mean_range=(0.0, 0.0), p=0.10),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.25, 0.25, 0.25),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )
    val_tfms = A.Compose(
        [
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.25, 0.25, 0.25),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )
    return train_tfms, val_tfms


class LiverSpleenHybridDataset(Dataset):
    def __init__(
        self,
        records,
        stats_matrix,
        stats_mean,
        stats_std,
        transform,
        img_size,
        organ_pad_ratio,
        context_pad_ratio,
        debug_dir=None,
        debug_max=0,
    ):
        self.records = records
        self.stats_matrix = stats_matrix.astype(np.float32)
        self.stats_mean = stats_mean.astype(np.float32)
        self.stats_std = stats_std.astype(np.float32)
        self.transform = transform
        self.img_size = img_size
        self.organ_pad_ratio = organ_pad_ratio
        self.context_pad_ratio = context_pad_ratio
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.debug_max = debug_max
        self.saved = 0
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def _to_rgb(self, crop):
        crop = np.clip(crop.astype(np.float32), 0.0, 1.0)
        return np.repeat(crop[..., None], 3, axis=-1)

    def _resize_gray(self, crop):
        crop = np.clip(crop.astype(np.float32), 0.0, 1.0)
        crop_u8 = (crop * 255.0).astype(np.uint8)
        resized = Image.fromarray(crop_u8).resize(
            (self.img_size, self.img_size),
            resample=Image.BICUBIC,
        )
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _resize_mask(self, mask):
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        resized = Image.fromarray(mask_u8).resize(
            (self.img_size, self.img_size),
            resample=Image.NEAREST,
        )
        return (np.asarray(resized) > 0).astype(np.float32)

    def _save_debug_panel(self, context_crop, liver_crop, spleen_crop, row):
        if self.debug_dir is None or self.saved >= self.debug_max:
            return

        def make_panel(arr):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("L").resize(
                (self.img_size, self.img_size),
                resample=Image.BICUBIC,
            ).convert("RGB")

        items = [
            ("Context", make_panel(context_crop)),
            ("Liver", make_panel(liver_crop)),
            ("Spleen", make_panel(spleen_crop)),
        ]

        canvas = Image.new("RGB", (self.img_size * 3, self.img_size + 34), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, (title, img) in enumerate(items):
            x0 = idx * self.img_size
            canvas.paste(img, (x0, 34))
            draw.text((x0 + 8, 8), title, fill=(20, 20, 20))
        label_name = "fatty_liver" if int(row["label"]) == 1 else "normal"
        out_name = f"sample_{self.saved:02d}_{label_name}_{row['image_id']}.png"
        canvas.save(self.debug_dir / out_name)
        self.saved += 1

    def __getitem__(self, idx):
        row = self.records[idx]
        image = load_grayscale(row["image_path"]).astype(np.float32) / 255.0
        liver_mask = load_mask(row["liver_mask_path"], image.shape)
        spleen_mask = load_mask(row["spleen_mask_path"], image.shape)

        liver_box = box_from_mask(liver_mask, self.organ_pad_ratio)
        spleen_box = box_from_mask(spleen_mask, self.organ_pad_ratio)
        context_box = box_from_union(liver_mask, spleen_mask, self.context_pad_ratio)

        liver_crop = crop_box(image, liver_box)
        spleen_crop = crop_box(image, spleen_box)
        context_crop = crop_box(image, context_box)
        liver_mask_crop = crop_box(liver_mask, liver_box)
        spleen_mask_crop = crop_box(spleen_mask, spleen_box)
        context_mask_crop = crop_box(((liver_mask > 0) | (spleen_mask > 0)).astype(np.uint8), context_box)

        liver_crop = self._resize_gray(liver_crop)
        spleen_crop = self._resize_gray(spleen_crop)
        context_crop = self._resize_gray(context_crop)
        liver_mask_crop = self._resize_mask(liver_mask_crop)
        spleen_mask_crop = self._resize_mask(spleen_mask_crop)
        context_mask_crop = self._resize_mask(context_mask_crop)

        liver_crop = liver_crop * liver_mask_crop
        spleen_crop = spleen_crop * spleen_mask_crop
        context_crop = context_crop * context_mask_crop

        liver_rgb = self._to_rgb(liver_crop)
        spleen_rgb = self._to_rgb(spleen_crop)
        context_rgb = self._to_rgb(context_crop)

        aug = self.transform(
            image=context_rgb,
            image_liver=liver_rgb,
            image_spleen=spleen_rgb,
        )

        stats = (self.stats_matrix[idx] - self.stats_mean) / self.stats_std
        stats = torch.tensor(stats, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        self._save_debug_panel(context_crop, liver_crop, spleen_crop, row)
        return aug["image_liver"], aug["image_spleen"], aug["image"], stats, label


class DenseNet121Encoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained_weights="imagenet"):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained_weights == "imagenet" else None
        backbone = models.densenet121(weights=weights)
        self.features = backbone.features
        in_features = backbone.classifier.in_features

        for param in self.features.parameters():
            param.requires_grad = False
        for module_name in ("transition3", "denseblock3", "denseblock4", "norm5"):
            module = getattr(self.features, module_name)
            for param in module.parameters():
                param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
        )

    def forward(self, x):
        feats = self.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return self.proj(feats)


class HybridLiverSpleenDenseNet(nn.Module):
    def __init__(self, stats_dim, dropout, pretrained_weights="imagenet"):
        super().__init__()
        self.encoder = DenseNet121Encoder(embed_dim=256, pretrained_weights=pretrained_weights)

        self.image_mlp = nn.Sequential(
            nn.Linear(256 * 7, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.50),
        )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(stats_dim, 160),
            nn.LayerNorm(160),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.60),
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.40),
        )

        self.image_head = nn.Linear(128, 1)
        self.tabular_head = nn.Linear(128, 1)

        self.fusion_head = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.40),
            nn.Linear(64, 1),
        )

    def forward(self, liver, spleen, context, stats):
        liver_feat = self.encoder(liver)
        spleen_feat = self.encoder(spleen)
        context_feat = self.encoder(context)

        pair_abs = torch.abs(liver_feat - spleen_feat)
        pair_mul = liver_feat * spleen_feat
        ctx_liver_abs = torch.abs(context_feat - liver_feat)
        ctx_spleen_abs = torch.abs(context_feat - spleen_feat)

        image_stack = torch.cat(
            [
                liver_feat,
                spleen_feat,
                context_feat,
                pair_abs,
                pair_mul,
                ctx_liver_abs,
                ctx_spleen_abs,
            ],
            dim=1,
        )
        image_repr = self.image_mlp(image_stack)
        tabular_repr = self.tabular_mlp(stats)

        image_logit = self.image_head(image_repr).squeeze(1)
        tabular_logit = self.tabular_head(tabular_repr).squeeze(1)

        fused = torch.cat(
            [
                image_repr,
                tabular_repr,
                torch.abs(image_repr - tabular_repr),
                image_repr * tabular_repr,
            ],
            dim=1,
        )
        logits = self.fusion_head(fused).squeeze(1)
        return {
            "logits": logits,
            "image_logit": image_logit,
            "tabular_logit": tabular_logit,
        }


def get_backbone_and_head_params(model):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.features."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params


def compute_binary_metrics(labels, probs, threshold=None):
    labels = np.asarray(labels, dtype=np.uint8)
    probs = np.asarray(probs, dtype=np.float32)

    auc = float(roc_auc_score(labels, probs))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[best_idx])
    used_thr = best_thr if threshold is None else float(threshold)
    preds = (probs >= used_thr).astype(np.uint8)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "acc": float(accuracy_score(labels, preds)),
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_thr": best_thr,
        "used_thr": used_thr,
    }


def make_epoch_row(epoch, train_loss, val_loss, train_metrics, val_metrics, val_at_train_thr, is_pretrain_eval=False):
    return {
        "epoch": epoch,
        "is_pretrain_eval": bool(is_pretrain_eval),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_metrics["acc"],
        "val_acc": val_metrics["acc"],
        "val_acc_train_thr": val_at_train_thr["acc"],
        "train_auc": train_metrics["auc"],
        "val_auc": val_metrics["auc"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "train_best_thr": train_metrics["best_thr"],
        "val_best_thr": val_metrics["best_thr"],
        "val_used_train_thr": val_at_train_thr["used_thr"],
    }


def fit_logistic_baseline(train_df, val_df, feature_cols):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    pipe.fit(train_df[feature_cols], train_df["label"])
    train_probs = pipe.predict_proba(train_df[feature_cols])[:, 1]
    val_probs = pipe.predict_proba(val_df[feature_cols])[:, 1]
    return pipe, train_probs, val_probs


def predict_with_model(model, loader, device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    labels_all = []
    final_probs = []
    image_probs = []
    tab_probs = []

    with torch.no_grad():
        for liver, spleen, context, stats, labels in loader:
            liver = liver.to(device)
            spleen = spleen.to(device)
            context = context.to(device)
            stats = stats.to(device)
            labels = labels.to(device)

            amp_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with amp_context:
                outputs = model(liver, spleen, context, stats)
                logits = outputs["logits"]
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss_sum += loss.item() * liver.size(0)
            labels_all.extend(labels.cpu().numpy())
            final_probs.extend(torch.sigmoid(outputs["logits"]).cpu().numpy())
            image_probs.extend(torch.sigmoid(outputs["image_logit"]).cpu().numpy())
            tab_probs.extend(torch.sigmoid(outputs["tabular_logit"]).cpu().numpy())

    denom = max(len(labels_all), 1)
    return {
        "loss": loss_sum / denom,
        "labels": np.asarray(labels_all, dtype=np.float32),
        "final_probs": np.asarray(final_probs, dtype=np.float32),
        "image_probs": np.asarray(image_probs, dtype=np.float32),
        "tab_probs": np.asarray(tab_probs, dtype=np.float32),
    }


def train_one_fold(
    model,
    train_loader,
    train_eval_loader,
    val_loader,
    class_counts,
    optimizer,
    scheduler,
    device,
    args,
    results_dir,
    fold,
):
    amp_enabled = torch.cuda.is_available() and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if class_counts[1] > 0:
        pos_weight = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    best_auc = -1.0
    best_checkpoint_path = Path(results_dir) / f"best_model_fold_{fold}.pth"
    no_improve = 0
    history = []
    last_val_pred = None

    if args.record_epoch_zero:
        train_pred = predict_with_model(model, train_eval_loader, device, amp_enabled)
        train_metrics = compute_binary_metrics(train_pred["labels"], train_pred["final_probs"])
        val_pred = predict_with_model(model, val_loader, device, amp_enabled)
        last_val_pred = val_pred
        val_metrics = compute_binary_metrics(val_pred["labels"], val_pred["final_probs"])
        val_at_train_thr = compute_binary_metrics(
            val_pred["labels"],
            val_pred["final_probs"],
            threshold=train_metrics["best_thr"],
        )
        history.append(
            make_epoch_row(
                epoch=0,
                train_loss=train_pred["loss"],
                val_loss=val_pred["loss"],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_at_train_thr=val_at_train_thr,
                is_pretrain_eval=True,
            )
        )
        print(
            f"Fold {fold} Epoch 0 pre-train | "
            f"TrainAUC={train_metrics['auc']:.4f} ValAUC={val_metrics['auc']:.4f} | "
            f"ValAcc(ROC-thr)={val_metrics['acc']:.4f} "
            f"ValAcc(train-thr)={val_at_train_thr['acc']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        labels_all = []
        probs_all = []

        for liver, spleen, context, stats, labels in tqdm(
            train_loader,
            desc=f"Fold {fold} Epoch {epoch}",
        ):
            liver = liver.to(device)
            spleen = spleen.to(device)
            context = context.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            targets = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with amp_context:
                outputs = model(liver, spleen, context, stats)
                loss_main = F.binary_cross_entropy_with_logits(
                    outputs["logits"],
                    targets,
                    pos_weight=pos_weight,
                )
                loss_tab = F.binary_cross_entropy_with_logits(
                    outputs["tabular_logit"],
                    targets,
                    pos_weight=pos_weight,
                )
                loss_img = F.binary_cross_entropy_with_logits(
                    outputs["image_logit"],
                    targets,
                    pos_weight=pos_weight,
                )
                loss = (
                    loss_main
                    + args.aux_tab_weight * loss_tab
                    + args.aux_image_weight * loss_img
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * liver.size(0)
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.sigmoid(outputs["logits"]).detach().cpu().numpy())

        train_loss = train_loss_sum / max(len(labels_all), 1)
        train_metrics = compute_binary_metrics(labels_all, probs_all)

        val_pred = predict_with_model(model, val_loader, device, amp_enabled)
        last_val_pred = val_pred
        val_metrics = compute_binary_metrics(val_pred["labels"], val_pred["final_probs"])
        val_at_train_thr = compute_binary_metrics(
            val_pred["labels"],
            val_pred["final_probs"],
            threshold=train_metrics["best_thr"],
        )
        scheduler.step(val_metrics["auc"])

        history.append(
            make_epoch_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_pred["loss"],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_at_train_thr=val_at_train_thr,
            )
        )

        print(
            f"Fold {fold} Epoch {epoch} | "
            f"TrainLoss={train_loss:.4f} ValLoss={val_pred['loss']:.4f} | "
            f"TrainAUC={train_metrics['auc']:.4f} ValAUC={val_metrics['auc']:.4f} | "
            f"ValAcc(ROC-thr)={val_metrics['acc']:.4f} "
            f"ValAcc(train-thr)={val_at_train_thr['acc']:.4f} "
            f"Recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            no_improve = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Fold {fold} best model saved (val_AUC improved to {best_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(results_dir) / f"metrics_full_fold_{fold}.csv", index=False)

    trained_history_df = history_df[history_df["epoch"] > 0]
    if trained_history_df.empty:
        trained_history_df = history_df
    best_row = trained_history_df.loc[trained_history_df["val_auc"].idxmax()]
    summary_df = pd.DataFrame(
        [best_row, trained_history_df.mean(numeric_only=True), trained_history_df.median(numeric_only=True)],
        index=["best_epoch", "mean", "median"],
    )
    summary_df.to_csv(Path(results_dir) / f"metrics_summary_fold_{fold}.csv")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc (train ROC-thr)")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc (val ROC-thr)")
    plt.plot(history_df["epoch"], history_df["val_acc_train_thr"], label="Val Acc (train ROC-thr)")
    plt.grid(True)
    plt.legend()
    plt.title(f"Fold {fold} Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_auc"], label="Train AUC")
    plt.plot(history_df["epoch"], history_df["val_auc"], label="Val AUC")
    plt.grid(True)
    plt.legend()
    plt.title(f"Fold {fold} AUC")
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"fold_{fold}_training_curves.png")
    plt.close()

    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        best_val_pred = predict_with_model(model, val_loader, device, amp_enabled)
    else:
        best_val_pred = last_val_pred
        if best_val_pred is None:
            best_val_pred = predict_with_model(model, val_loader, device, amp_enabled)
    return history_df, best_val_pred


def compute_oof_summary(pred_df, results_dir):
    summary_rows = []
    prob_cols = [
        "hybrid_prob",
        "logreg_prob",
        "average_blend_prob",
        "weighted_blend_prob",
        "image_branch_prob",
        "tabular_branch_prob",
    ]
    labels = pred_df["label"].to_numpy()
    for col in prob_cols:
        metrics = compute_binary_metrics(labels, pred_df[col].to_numpy())
        fold_aucs = [
            roc_auc_score(fold_df["label"].to_numpy(), fold_df[col].to_numpy())
            for _, fold_df in pred_df.groupby("fold")
        ]
        summary_rows.append(
            {
                "prediction": col,
                **metrics,
                "fold_mean_auc": float(np.mean(fold_aucs)),
                "fold_std_auc": float(np.std(fold_aucs)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    summary_df.to_csv(Path(results_dir) / "oof_summary.csv", index=False)
    return summary_df


def save_debug_previews(
    records,
    stats_matrix,
    stats_mean,
    stats_std,
    transform,
    args,
    debug_dir,
):
    if args.debug_max <= 0:
        return
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for old_file in debug_dir.glob("sample_*.png"):
        old_file.unlink()

    class_to_indices = {}
    for idx, row in enumerate(records):
        class_to_indices.setdefault(int(row["label"]), []).append(idx)

    selected_indices = []
    label_order = sorted(class_to_indices)
    while len(selected_indices) < min(args.debug_max, len(records)):
        made_progress = False
        for label in label_order:
            if class_to_indices[label]:
                selected_indices.append(class_to_indices[label].pop(0))
                made_progress = True
                if len(selected_indices) >= min(args.debug_max, len(records)):
                    break
        if not made_progress:
            break

    if not selected_indices:
        return

    debug_records = [records[idx] for idx in selected_indices]
    debug_stats = stats_matrix[selected_indices]
    debug_dataset = LiverSpleenHybridDataset(
        debug_records,
        debug_stats,
        stats_mean,
        stats_std,
        transform=transform,
        img_size=args.img_size,
        organ_pad_ratio=args.organ_pad_ratio,
        context_pad_ratio=args.context_pad_ratio,
        debug_dir=debug_dir,
        debug_max=args.debug_max,
    )
    for idx in range(len(debug_dataset)):
        _ = debug_dataset[idx]


def main():
    args = parse_args()
    seed_everything(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_config = vars(args).copy()
    run_config["device"] = str(device)
    run_config["comment"] = DEFAULT_COMMENT
    pd.DataFrame.from_dict(run_config, orient="index", columns=["value"]).to_csv(
        results_dir / "run_config.csv"
    )

    dataset_df = load_dataset_table(args.dataset_csv)
    merged_df, feature_cols = load_or_build_feature_table(dataset_df, args.features_csv, results_dir)
    splits, groups, use_group_split, group_source = make_splits(merged_df, args.folds, args.seed)
    if args.max_folds is not None:
        splits = splits[: args.max_folds]

    print("Total matched samples:", len(merged_df))
    print("Class counts:", merged_df["label"].value_counts().sort_index().to_dict())
    print(
        "Using grouped split:"
        f" {use_group_split} | group source: {group_source} | unique groups: {groups.nunique()}"
    )
    print("Tabular feature count:", len(feature_cols))

    train_tfms, val_tfms = build_transforms(args.img_size)
    all_fold_predictions = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n========== FOLD {fold}/{len(splits)} ==========")

        train_df = merged_df.iloc[train_idx].reset_index(drop=True)
        val_df = merged_df.iloc[val_idx].reset_index(drop=True)
        save_split_integrity_report(train_df, val_df, fold, results_dir)

        train_records = train_df[
            ["image_id", "class_name", "label", "image_path", "liver_mask_path", "spleen_mask_path"]
        ].to_dict("records")
        val_records = val_df[
            ["image_id", "class_name", "label", "image_path", "liver_mask_path", "spleen_mask_path"]
        ].to_dict("records")

        train_stats = train_df[feature_cols].to_numpy(dtype=np.float32)
        val_stats = val_df[feature_cols].to_numpy(dtype=np.float32)
        stats_mean = train_stats.mean(axis=0)
        stats_std = train_stats.std(axis=0)
        stats_std[stats_std < 1e-6] = 1.0

        fold_debug_dir = results_dir / f"debug_inputs_fold_{fold}"
        save_debug_previews(
            train_records,
            train_stats,
            stats_mean,
            stats_std,
            val_tfms,
            args,
            fold_debug_dir,
        )
        train_loader = DataLoader(
            LiverSpleenHybridDataset(
                train_records,
                train_stats,
                stats_mean,
                stats_std,
                transform=train_tfms,
                img_size=args.img_size,
                organ_pad_ratio=args.organ_pad_ratio,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_eval_loader = DataLoader(
            LiverSpleenHybridDataset(
                train_records,
                train_stats,
                stats_mean,
                stats_std,
                transform=val_tfms,
                img_size=args.img_size,
                organ_pad_ratio=args.organ_pad_ratio,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            LiverSpleenHybridDataset(
                val_records,
                val_stats,
                stats_mean,
                stats_std,
                transform=val_tfms,
                img_size=args.img_size,
                organ_pad_ratio=args.organ_pad_ratio,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        _, _, logreg_val_probs = fit_logistic_baseline(train_df, val_df, feature_cols)
        logreg_metrics = compute_binary_metrics(val_df["label"].to_numpy(), logreg_val_probs)
        print(
            f"Fold {fold} logistic baseline | "
            f"AUC={logreg_metrics['auc']:.4f} Acc={logreg_metrics['acc']:.4f} "
            f"Recall={logreg_metrics['recall']:.4f}"
        )

        model = HybridLiverSpleenDenseNet(
            stats_dim=len(feature_cols),
            dropout=args.dropout,
            pretrained_weights=args.pretrained_weights,
        ).to(device)
        backbone_params, head_params = get_backbone_and_head_params(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        _, best_val_pred = train_one_fold(
            model=model,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            class_counts=np.bincount(train_df["label"].to_numpy(dtype=np.int64), minlength=2),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            results_dir=results_dir,
            fold=fold,
        )

        fold_pred_df = val_df[["image_id", "class_name", "label"]].copy()
        fold_pred_df["fold"] = fold
        fold_pred_df["hybrid_prob"] = best_val_pred["final_probs"]
        fold_pred_df["image_branch_prob"] = best_val_pred["image_probs"]
        fold_pred_df["tabular_branch_prob"] = best_val_pred["tab_probs"]
        fold_pred_df["logreg_prob"] = logreg_val_probs
        fold_pred_df["average_blend_prob"] = 0.5 * (
            fold_pred_df["hybrid_prob"] + fold_pred_df["logreg_prob"]
        )
        fold_pred_df["weighted_blend_prob"] = (
            args.model_blend_weight * fold_pred_df["hybrid_prob"]
            + (1.0 - args.model_blend_weight) * fold_pred_df["logreg_prob"]
        )
        fold_pred_df.to_csv(results_dir / f"predictions_fold_{fold}.csv", index=False)
        all_fold_predictions.append(fold_pred_df)

    oof_df = pd.concat(all_fold_predictions, ignore_index=True)
    oof_df.to_csv(results_dir / "oof_predictions.csv", index=False)
    summary_df = compute_oof_summary(oof_df, results_dir)
    print("\nOOF summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

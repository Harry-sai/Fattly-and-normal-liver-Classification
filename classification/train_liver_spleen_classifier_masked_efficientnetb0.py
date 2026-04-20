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


COMMENT = """Masked EfficientNet-B0 liver+spleen classifier.
This keeps the more stable single-image architecture style from spleen_efficientnetb0.py,
but fixes the main ROI problem:
- one aligned union crop is used for all channels
- channel 1 is union-masked context
- channel 2 is liver-mask-only pixels
- channel 3 is spleen-mask-only pixels
- non-organ pixels are zeroed after resizing masks with nearest-neighbor interpolation

The model is intentionally smaller and more frozen than the DenseNet hybrid to reduce overfitting.
It also reports a fold-wise logistic-regression feature baseline and blend for comparison.
- it records epoch-0 sanity metrics and split-integrity reports so early high validation AUC can be audited
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train masked EfficientNet-B0 liver+spleen classifier.")
    parser.add_argument("--dataset-csv", default="artifacts/liver_spleen_dataset.csv")
    parser.add_argument("--features-csv", default="artifacts/liver_spleen_features.csv")
    parser.add_argument("--results-dir", default="classification/masked_efficientnetb0/v1")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.12)
    parser.add_argument("--weight-decay", type=float, default=2.0e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.48)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--context-pad-ratio", type=float, default=0.10)
    parser.add_argument("--debug-max", type=int, default=8)
    parser.add_argument("--blend-weight", type=float, default=0.70)
    parser.add_argument("--max-pos-weight", type=float, default=1.12)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument(
        "--eval-tta-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average predictions from original and horizontally flipped masked crops during evaluation.",
    )
    parser.add_argument(
        "--pretrained-weights",
        choices=("imagenet", "random"),
        default="imagenet",
        help="Use random to sanity-check whether early high validation AUC is coming from pretrained image features.",
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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def box_from_union(mask_a, mask_b, pad_ratio):
    union = ((mask_a > 0) | (mask_b > 0)).astype(np.uint8)
    ys, xs = np.where(union > 0)
    h, w = union.shape
    if len(xs) == 0:
        side = min(h, w)
        top = max(0, (h - side) // 2)
        left = max(0, (w - side) // 2)
        return top, top + side, left, left + side
    return safe_square_box(ys.min(), ys.max(), xs.min(), xs.max(), h, w, pad_ratio)


def crop_box(arr, box):
    top, bottom, left, right = box
    return arr[top:bottom, left:right]


def resize_gray(crop, img_size):
    crop = np.clip(crop.astype(np.float32), 0.0, 1.0)
    crop_u8 = (crop * 255.0).astype(np.uint8)
    resized = Image.fromarray(crop_u8).resize((img_size, img_size), resample=Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_mask(mask, img_size):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    resized = Image.fromarray(mask_u8).resize((img_size, img_size), resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.float32)


def build_groups(df):
    for col in ("group_id", "patient_id", "study_id"):
        if col in df.columns:
            return df[col].fillna("missing").astype(str), col
    base_id = df["image_id"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
    return df["class_name"].astype(str) + "::" + base_id, "derived_class_image_group"


def make_splits(df, folds, seed):
    labels = df["label"].to_numpy()
    groups, group_source = build_groups(df)
    use_groups = (
        StratifiedGroupKFold is not None
        and groups.nunique() < len(groups)
        and groups.nunique() >= folds
    )
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(df, labels, groups)), groups, group_source
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(df, labels)), groups, "stratified_no_groups"


def derived_class_image_groups(df):
    base_id = df["image_id"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
    return df["class_name"].astype(str) + "::" + base_id


def class_image_ids(df):
    return df["class_name"].astype(str) + "::" + df["image_id"].astype(str)


def save_split_integrity_report(train_df, val_df, fold, out_dir):
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

    rows = []
    for check_name, overlap, warns_on_overlap in checks:
        rows.append(
            {
                "fold": fold,
                "check": check_name,
                "warns_on_overlap": warns_on_overlap,
                "overlap_count": len(overlap),
                "example_values": "; ".join(sorted(overlap)[:10]),
            }
        )
    report_df = pd.DataFrame(rows)
    report_df.to_csv(Path(out_dir) / f"split_integrity_fold_{fold}.csv", index=False)

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


def load_tables(dataset_csv, features_csv):
    dataset_df = pd.read_csv(dataset_csv)
    features_df = pd.read_csv(features_csv)
    merged = dataset_df.merge(
        features_df,
        on=["image_id", "class_name", "label"],
        how="left",
        validate="one_to_one",
    )
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
    feature_cols = [
        col for col in merged.columns
        if col not in meta_cols and pd.api.types.is_numeric_dtype(merged[col])
    ]
    if merged[feature_cols].isnull().any().any():
        raise RuntimeError("Feature table has missing values after merge. Rebuild artifacts/liver_spleen_features.csv.")
    return merged, feature_cols


def build_transforms():
    train_tfms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.03,
                scale=(0.97, 1.03),
                rotate=(-5, 5),
                border_mode=0,
                p=0.30,
            ),
            A.GaussNoise(std_range=(0.002, 0.008), mean_range=(0.0, 0.0), p=0.08),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


class MaskedLiverSpleenDataset(Dataset):
    def __init__(
        self,
        records,
        stats_matrix,
        stats_mean,
        stats_std,
        transform,
        img_size,
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
        self.context_pad_ratio = context_pad_ratio
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.debug_max = debug_max
        self.saved = 0
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def _save_debug_panel(self, context, liver, spleen, row):
        if self.debug_dir is None or self.saved >= self.debug_max:
            return
        panels = [("Context", context), ("Liver", liver), ("Spleen", spleen)]
        canvas = Image.new("RGB", (self.img_size * 3, self.img_size + 34), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, (title, arr) in enumerate(panels):
            arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr_u8).convert("RGB")
            x0 = idx * self.img_size
            canvas.paste(img, (x0, 34))
            draw.text((x0 + 8, 8), title, fill=(20, 20, 20))
        label_name = "fatty_liver" if int(row["label"]) == 1 else "normal"
        canvas.save(self.debug_dir / f"sample_{self.saved:02d}_{label_name}_{row['image_id']}.png")
        self.saved += 1

    def __getitem__(self, idx):
        row = self.records[idx]
        image = load_grayscale(row["image_path"]).astype(np.float32) / 255.0
        liver_mask = load_mask(row["liver_mask_path"], image.shape)
        spleen_mask = load_mask(row["spleen_mask_path"], image.shape)
        union_mask = ((liver_mask > 0) | (spleen_mask > 0)).astype(np.uint8)

        box = box_from_union(liver_mask, spleen_mask, self.context_pad_ratio)
        image_crop = resize_gray(crop_box(image, box), self.img_size)
        liver_mask_crop = resize_mask(crop_box(liver_mask, box), self.img_size)
        spleen_mask_crop = resize_mask(crop_box(spleen_mask, box), self.img_size)
        union_mask_crop = resize_mask(crop_box(union_mask, box), self.img_size)

        context = image_crop * union_mask_crop
        liver = image_crop * liver_mask_crop
        spleen = image_crop * spleen_mask_crop
        stacked = np.stack([context, liver, spleen], axis=-1).astype(np.float32)

        aug = self.transform(image=stacked)
        stats = (self.stats_matrix[idx] - self.stats_mean) / self.stats_std
        stats = torch.tensor(stats, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        self._save_debug_panel(context, liver, spleen, row)
        return aug["image"], stats, label


class MaskedEfficientNetB0(nn.Module):
    def __init__(self, stats_dim, dropout, pretrained_weights="imagenet"):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained_weights == "imagenet" else None
        self.backbone = models.efficientnet_b0(weights=weights)

        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for block in self.backbone.features[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.35),
        )
        self.head = nn.Sequential(
            nn.Linear(in_features + 64, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.70),
            nn.Linear(64, 1),
        )

    def forward(self, x, stats):
        features = self.backbone(x)
        stats_features = self.stats_mlp(stats)
        fused = torch.cat([features, stats_features], dim=1)
        return self.head(fused).squeeze(1)


def compute_metrics(labels, probs, threshold=None):
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


def fit_logreg(train_df, val_df, feature_cols):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    pipe.fit(train_df[feature_cols], train_df["label"])
    return pipe.predict_proba(val_df[feature_cols])[:, 1]


def predict(model, loader, device, amp_enabled, tta_flip=False):
    model.eval()
    losses, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for imgs, stats, labels in loader:
            imgs = imgs.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            amp_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(imgs, stats)
                if tta_flip:
                    flip_logits = model(torch.flip(imgs, dims=[3]), stats)
                    logits = 0.5 * (logits + flip_logits)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(loss.item() * imgs.size(0))
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(torch.sigmoid(logits).cpu().numpy())
    return sum(losses) / max(len(labels_all), 1), np.asarray(labels_all), np.asarray(probs_all)


def train_fold(model, train_loader, train_eval_loader, val_loader, class_counts, optimizer, scheduler, device, args, out_dir, fold):
    amp_enabled = torch.cuda.is_available() and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pos_weight_value = min(class_counts[0] / max(class_counts[1], 1), args.max_pos_weight)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    head_group_idx = 1 if len(optimizer.param_groups) > 1 else 0
    backbone_group_idx = 0

    history = []
    best_auc = -1.0
    best_path = Path(out_dir) / f"best_model_fold_{fold}.pth"
    no_improve = 0
    last_val_pred = None

    if args.record_epoch_zero:
        train_loss0, train_labels0, train_probs0 = predict(
            model,
            train_eval_loader,
            device,
            amp_enabled,
            tta_flip=args.eval_tta_flip,
        )
        train_metrics0 = compute_metrics(train_labels0, train_probs0)
        val_loss0, val_labels0, val_probs0 = predict(
            model,
            val_loader,
            device,
            amp_enabled,
            tta_flip=args.eval_tta_flip,
        )
        last_val_pred = (val_labels0, val_probs0)
        val_metrics0 = compute_metrics(val_labels0, val_probs0)
        val_at_train_thr0 = compute_metrics(val_labels0, val_probs0, threshold=train_metrics0["best_thr"])
        history.append(
            make_epoch_row(
                epoch=0,
                train_loss=train_loss0,
                val_loss=val_loss0,
                train_metrics=train_metrics0,
                val_metrics=val_metrics0,
                val_at_train_thr=val_at_train_thr0,
                is_pretrain_eval=True,
            )
        )
        print(
            f"Fold {fold} Epoch 0 pre-train | "
            f"TrainAUC={train_metrics0['auc']:.4f} ValAUC={val_metrics0['auc']:.4f} | "
            f"ValAcc(ROC-thr)={val_metrics0['acc']:.4f} "
            f"ValAcc(train-thr)={val_at_train_thr0['acc']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        current_head_lr = optimizer.param_groups[head_group_idx]["lr"]
        if epoch <= args.freeze_backbone_epochs:
            optimizer.param_groups[backbone_group_idx]["lr"] = 0.0
        else:
            optimizer.param_groups[backbone_group_idx]["lr"] = current_head_lr * args.backbone_lr_scale

        model.train()
        train_loss_sum = 0.0
        train_labels, train_probs = [], []

        for imgs, stats, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"):
            imgs = imgs.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            targets = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(imgs, stats)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * imgs.size(0)
            train_labels.extend(labels.detach().cpu().numpy())
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss = train_loss_sum / max(len(train_labels), 1)
        train_metrics = compute_metrics(train_labels, train_probs)
        val_loss, val_labels, val_probs = predict(
            model,
            val_loader,
            device,
            amp_enabled,
            tta_flip=args.eval_tta_flip,
        )
        last_val_pred = (val_labels, val_probs)
        val_metrics = compute_metrics(val_labels, val_probs)
        val_at_train_thr = compute_metrics(val_labels, val_probs, threshold=train_metrics["best_thr"])
        scheduler.step(val_metrics["auc"])

        history.append(
            make_epoch_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_at_train_thr=val_at_train_thr,
            )
        )
        print(
            f"Fold {fold} Epoch {epoch} | "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} | "
            f"TrainAUC={train_metrics['auc']:.4f} ValAUC={val_metrics['auc']:.4f} | "
            f"ValAcc(ROC-thr)={val_metrics['acc']:.4f} "
            f"ValAcc(train-thr)={val_at_train_thr['acc']:.4f} "
            f"Recall={val_metrics['recall']:.4f} | "
            f"BackboneLR={optimizer.param_groups[backbone_group_idx]['lr']:.2e} "
            f"HeadLR={optimizer.param_groups[head_group_idx]['lr']:.2e}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Fold {fold} best model saved (val_AUC improved to {best_auc:.4f})")
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(out_dir) / f"metrics_full_fold_{fold}.csv", index=False)
    trained_history_df = history_df[history_df["epoch"] > 0]
    if trained_history_df.empty:
        trained_history_df = history_df
    best_row = trained_history_df.loc[trained_history_df["val_auc"].idxmax()]
    summary = pd.DataFrame(
        [best_row, trained_history_df.mean(numeric_only=True), trained_history_df.median(numeric_only=True)],
        index=["best_epoch", "mean", "median"],
    )
    summary.to_csv(Path(out_dir) / f"metrics_summary_fold_{fold}.csv")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc (val-selected thr)", linestyle=":")
    plt.plot(history_df["epoch"], history_df["val_acc_train_thr"], label="Val Acc (train-selected thr)", linestyle="--")
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
    plt.savefig(Path(out_dir) / f"fold_{fold}_training_curves.png")
    plt.close()

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        _, labels, probs = predict(
            model,
            val_loader,
            device,
            amp_enabled,
            tta_flip=args.eval_tta_flip,
        )
    else:
        if last_val_pred is None:
            _, labels, probs = predict(
                model,
                val_loader,
                device,
                amp_enabled,
                tta_flip=args.eval_tta_flip,
            )
        else:
            labels, probs = last_val_pred
    return history_df, labels, probs


def save_debug(records, stats_matrix, stats_mean, stats_std, transform, args, debug_dir):
    if args.debug_max <= 0:
        return
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for old_file in debug_dir.glob("sample_*.png"):
        old_file.unlink()

    by_label = {}
    for idx, row in enumerate(records):
        by_label.setdefault(int(row["label"]), []).append(idx)
    selected = []
    labels = sorted(by_label)
    while len(selected) < min(args.debug_max, len(records)):
        made_progress = False
        for label in labels:
            if by_label[label]:
                selected.append(by_label[label].pop(0))
                made_progress = True
                if len(selected) >= min(args.debug_max, len(records)):
                    break
        if not made_progress:
            break
    if not selected:
        return
    debug_dataset = MaskedLiverSpleenDataset(
        [records[idx] for idx in selected],
        stats_matrix[selected],
        stats_mean,
        stats_std,
        transform=transform,
        img_size=args.img_size,
        context_pad_ratio=args.context_pad_ratio,
        debug_dir=debug_dir,
        debug_max=args.debug_max,
    )
    for idx in range(len(debug_dataset)):
        _ = debug_dataset[idx]


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def summarize_oof(oof_df, out_dir):
    rows = []
    for col in ["model_prob", "logreg_prob", "blend_prob"]:
        metrics = compute_metrics(oof_df["label"].to_numpy(), oof_df[col].to_numpy())
        fold_aucs = [
            roc_auc_score(fold_df["label"], fold_df[col])
            for _, fold_df in oof_df.groupby("fold")
        ]
        rows.append(
            {
                "prediction": col,
                **metrics,
                "fold_mean_auc": float(np.mean(fold_aucs)),
                "fold_std_auc": float(np.std(fold_aucs)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary.to_csv(Path(out_dir) / "oof_summary.csv", index=False)
    return summary


def save_blend_sweep(oof_df, out_dir):
    labels = oof_df["label"].to_numpy()
    model_probs = oof_df["model_prob"].to_numpy()
    logreg_probs = oof_df["logreg_prob"].to_numpy()
    rows = []
    for model_weight in np.linspace(0.0, 1.0, 101):
        blend = model_weight * model_probs + (1.0 - model_weight) * logreg_probs
        rows.append(
            {
                "model_weight": float(model_weight),
                "logreg_weight": float(1.0 - model_weight),
                "auc": float(roc_auc_score(labels, blend)),
            }
        )
    sweep_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    sweep_df.to_csv(Path(out_dir) / "blend_weight_sweep.csv", index=False)
    best_row = sweep_df.iloc[0]
    pd.DataFrame([best_row]).to_csv(Path(out_dir) / "best_blend_weight.csv", index=False)
    print(
        f"OOF blend sweep | best model_weight={best_row['model_weight']:.2f} "
        f"logreg_weight={best_row['logreg_weight']:.2f} AUC={best_row['auc']:.4f}"
    )
    return float(best_row["model_weight"])


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = vars(args).copy()
    config["device"] = str(device)
    config["comment"] = COMMENT
    pd.DataFrame.from_dict(config, orient="index", columns=["value"]).to_csv(out_dir / "run_config.csv")

    df, feature_cols = load_tables(args.dataset_csv, args.features_csv)
    splits, groups, group_source = make_splits(df, args.folds, args.seed)
    if args.max_folds is not None:
        splits = splits[: args.max_folds]

    print("Total matched samples:", len(df))
    print("Class counts:", df["label"].value_counts().sort_index().to_dict())
    print(f"Using group source: {group_source} | unique groups: {groups.nunique()}")
    print("Tabular feature count:", len(feature_cols))

    train_tfms, val_tfms = build_transforms()
    oof_rows = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n========== FOLD {fold}/{len(splits)} ==========")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        save_split_integrity_report(train_df, val_df, fold, out_dir)
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

        debug_dir = out_dir / f"debug_inputs_fold_{fold}"
        save_debug(train_records, train_stats, stats_mean, stats_std, val_tfms, args, debug_dir)

        train_loader = make_loader(
            MaskedLiverSpleenDataset(
                train_records,
                train_stats,
                stats_mean,
                stats_std,
                transform=train_tfms,
                img_size=args.img_size,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed + fold * 100 + 1,
        )
        train_eval_loader = make_loader(
            MaskedLiverSpleenDataset(
                train_records,
                train_stats,
                stats_mean,
                stats_std,
                transform=val_tfms,
                img_size=args.img_size,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed + fold * 100 + 2,
        )
        val_loader = make_loader(
            MaskedLiverSpleenDataset(
                val_records,
                val_stats,
                stats_mean,
                stats_std,
                transform=val_tfms,
                img_size=args.img_size,
                context_pad_ratio=args.context_pad_ratio,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed + fold * 100 + 3,
        )

        logreg_probs = fit_logreg(train_df, val_df, feature_cols)
        logreg_metrics = compute_metrics(val_df["label"], logreg_probs)
        print(f"Fold {fold} logistic baseline | AUC={logreg_metrics['auc']:.4f} Acc={logreg_metrics['acc']:.4f}")

        model = MaskedEfficientNetB0(
            stats_dim=len(feature_cols),
            dropout=args.dropout,
            pretrained_weights=args.pretrained_weights,
        ).to(device)
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone.features"):
                backbone_params.append(param)
            else:
                head_params.append(param)
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

        _, val_labels, model_probs = train_fold(
            model,
            train_loader,
            train_eval_loader,
            val_loader,
            class_counts=np.bincount(train_df["label"].to_numpy(dtype=np.int64), minlength=2),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            out_dir=out_dir,
            fold=fold,
        )

        fold_df = val_df[["image_id", "class_name", "label"]].copy()
        fold_df["fold"] = fold
        fold_df["model_prob"] = model_probs
        fold_df["logreg_prob"] = logreg_probs
        fold_df["blend_prob"] = args.blend_weight * fold_df["model_prob"] + (1.0 - args.blend_weight) * fold_df["logreg_prob"]
        fold_df.to_csv(out_dir / f"predictions_fold_{fold}.csv", index=False)
        oof_rows.append(fold_df)

    oof_df = pd.concat(oof_rows, ignore_index=True)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    save_blend_sweep(oof_df, out_dir)
    summary = summarize_oof(oof_df, out_dir)
    print("\nOOF summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

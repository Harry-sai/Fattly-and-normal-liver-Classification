import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Combined Liver + Spleen Classification
# Multi-view EfficientNet-B0
# ============================================================

DATASET_CSV = "artifacts/liver_spleen_dataset.csv"
FEATURES_CSV = "artifacts/liver_spleen_features.csv"
RESULTS_DIR = "classification/multiview_efficientnetb0/1st_run"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 320
BATCH_SIZE = 4
EPOCHS = 36
LR = 1.2e-4
BACKBONE_LR_SCALE = 0.25
WEIGHT_DECAY = 1.2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KFOLDS = 5
SEED = 42
PATIENCE = 7
DROP_OUT = 0.35
STATS_DIM = 8
LABEL_SMOOTHING = 0.01
GRAD_CLIP_NORM = 1.0
MIXUP_ALPHA = 0.10
MIXUP_PROB = 0.12


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


train_tfms = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.04,
            scale=(0.96, 1.04),
            rotate=(-7, 7),
            border_mode=0,
            p=0.35,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.05,
            contrast_limit=0.05,
            p=0.15,
        ),
        A.GaussNoise(std_range=(0.008, 0.02), mean_range=(0.0, 0.0), p=0.08),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0),
        ToTensorV2(),
    ]
)

val_tfms = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0),
        ToTensorV2(),
    ]
)


RUN_CONFIG = {
    "DATASET_CSV": DATASET_CSV,
    "FEATURES_CSV": FEATURES_CSV,
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "BACKBONE_LR_SCALE": BACKBONE_LR_SCALE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "KFOLDS": KFOLDS,
    "DROP_OUT": DROP_OUT,
    "MODEL": "Multi-view EfficientNet-B0 with shared-context crop + attenuation stats",
    "OPTIMIZER": "AdamW + ReduceLROnPlateau",
    "LOSS": "BCEWithLogitsLoss",
    "COMMENT": """Task understanding:
- label is binary: normal vs fatty_liver
- liver and spleen should be analyzed together, not as isolated unrelated crops
- shared anatomical context matters, but organ-specific evidence matters too

This script uses three aligned views from the same union crop:
1) shared full crop around liver+spleen
2) liver-masked crop inside the same shared crop
3) spleen-masked crop inside the same shared crop

Each view is encoded by the same pretrained EfficientNet backbone.
Their features, pairwise differences, and attenuation statistics are fused
in a classification head. Best checkpoint is selected by validation AUC only.
""",
}

pd.DataFrame.from_dict(RUN_CONFIG, orient="index", columns=["value"]).to_csv(
    os.path.join(RESULTS_DIR, "run_config.csv")
)


def robust_normalize(image):
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi - lo < 1e-6:
        image = np.clip(image / 255.0, 0.0, 1.0)
    else:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32)


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


def square_crop_from_box(img, y1, y2, x1, x2, pad_ratio=0.16):
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1
    side = int(max(box_h, box_w) * (1.0 + 2.0 * pad_ratio))
    side = max(32, min(side, max(img.shape[0], img.shape[1])))

    cy = 0.5 * (y1 + y2)
    cx = 0.5 * (x1 + x2)
    y1 = int(round(cy - side / 2))
    x1 = int(round(cx - side / 2))
    y1 = max(0, min(y1, img.shape[0] - side))
    x1 = max(0, min(x1, img.shape[1] - side))
    return img[y1:y1 + side, x1:x1 + side]


def square_crop_from_union(img, liver_mask, spleen_mask, pad_ratio=0.16):
    union_mask = ((liver_mask > 0) | (spleen_mask > 0)).astype(np.uint8)
    ys, xs = np.where(union_mask > 0)
    if len(xs) == 0:
        side = min(img.shape[0], img.shape[1])
        y1 = max(0, (img.shape[0] - side) // 2)
        x1 = max(0, (img.shape[1] - side) // 2)
        return img[y1:y1 + side, x1:x1 + side]

    return square_crop_from_box(img, ys.min(), ys.max(), xs.min(), xs.max(), pad_ratio=pad_ratio)


def build_feature_lookup(features_csv):
    df = pd.read_csv(features_csv)
    lookup = {}
    for row in df.itertuples(index=False):
        stats = np.array(
            [
                row.liver_mean,
                row.spleen_mean,
                row.mean_ratio,
                row.mean_diff,
                row.liver_median,
                row.spleen_median,
                row.median_ratio,
                row.median_diff,
            ],
            dtype=np.float32,
        )
        lookup[row.image_id] = stats
    return lookup


def normalize_stats(train_rows, feature_lookup):
    arr = np.stack([feature_lookup[row["image_id"]] for row in train_rows], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_mixup(ctx, liver, spleen, stats, labels, alpha=MIXUP_ALPHA):
    if ctx.size(0) < 2 or alpha <= 0:
        return ctx, liver, spleen, stats, labels

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(ctx.size(0), device=ctx.device)
    ctx = lam * ctx + (1.0 - lam) * ctx[perm]
    liver = lam * liver + (1.0 - lam) * liver[perm]
    spleen = lam * spleen + (1.0 - lam) * spleen[perm]
    stats = lam * stats + (1.0 - lam) * stats[perm]
    labels = lam * labels + (1.0 - lam) * labels[perm]
    return ctx, liver, spleen, stats, labels


class LiverSpleenMultiViewDataset(Dataset):
    def __init__(
        self,
        rows,
        feature_lookup,
        stats_mean,
        stats_std,
        transform=None,
        debug=False,
        debug_max=8,
        debug_dir=None,
    ):
        self.rows = rows
        self.feature_lookup = feature_lookup
        self.stats_mean = stats_mean
        self.stats_std = stats_std
        self.transform = transform
        self.debug = debug
        self.debug_max = debug_max
        self.saved = 0
        self.debug_dir = Path(debug_dir) if debug_dir is not None else Path(RESULTS_DIR) / "debug_inputs"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = np.array(Image.open(row["image_path"]).convert("L"))
        liver_mask = load_mask(row["liver_mask_path"], image.shape)
        spleen_mask = load_mask(row["spleen_mask_path"], image.shape)

        image_norm = robust_normalize(image)

        shared_crop = square_crop_from_union(image_norm, liver_mask, spleen_mask)
        liver_crop = square_crop_from_union(image_norm * liver_mask, liver_mask, spleen_mask)
        spleen_crop = square_crop_from_union(image_norm * spleen_mask, liver_mask, spleen_mask)

        shared_crop = np.array(
            Image.fromarray((shared_crop * 255.0).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC
            )
        ).astype(np.float32) / 255.0
        liver_crop = np.array(
            Image.fromarray((liver_crop * 255.0).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC
            )
        ).astype(np.float32) / 255.0
        spleen_crop = np.array(
            Image.fromarray((spleen_crop * 255.0).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC
            )
        ).astype(np.float32) / 255.0

        stacked = np.stack([shared_crop, liver_crop, spleen_crop], axis=-1)
        aug = self.transform(image=stacked)
        stacked_tensor = aug["image"]

        context_tensor = stacked_tensor[0:1]
        liver_tensor = stacked_tensor[1:2]
        spleen_tensor = stacked_tensor[2:3]

        if self.debug and self.saved < self.debug_max:
            self._save_debug_panel(shared_crop, liver_crop, spleen_crop)
            self.saved += 1

        stats = (self.feature_lookup[row["image_id"]] - self.stats_mean) / self.stats_std
        stats_tensor = torch.tensor(stats, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return context_tensor, liver_tensor, spleen_tensor, stats_tensor, label

    def _gray_to_pil(self, arr):
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8)).convert("L")

    def _save_debug_panel(self, shared_crop, liver_crop, spleen_crop):
        ctx = self._gray_to_pil(shared_crop).convert("RGB")
        liv = self._gray_to_pil(liver_crop).convert("RGB")
        spl = self._gray_to_pil(spleen_crop).convert("RGB")

        canvas = Image.new("RGB", (IMG_SIZE * 3, IMG_SIZE + 34), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        items = [("Shared crop", ctx), ("Liver view", liv), ("Spleen view", spl)]

        for idx, (title, im) in enumerate(items):
            x0 = idx * IMG_SIZE
            canvas.paste(im, (x0, 34))
            draw.text((x0 + 8, 8), title, fill=(20, 20, 20))

        canvas.save(self.debug_dir / f"sample_{self.saved}.png")


class MultiViewEfficientNetB0(nn.Module):
    def __init__(self, stats_dim=STATS_DIM, dropout=DROP_OUT):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        for param in self.backbone.features[:-2].parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.view_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, 48),
            nn.LayerNorm(48),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.4),
        )

        self.head = nn.Sequential(
            nn.Linear(256 * 6 + 48, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.9),
            nn.Linear(128, 1),
        )

    def encode_view(self, x):
        rgb = x.repeat(1, 3, 1, 1)
        feat = self.backbone(rgb)
        return self.view_proj(feat)

    def forward(self, context, liver, spleen, stats):
        ctx_feat = self.encode_view(context)
        liv_feat = self.encode_view(liver)
        spl_feat = self.encode_view(spleen)

        diff_ls = torch.abs(liv_feat - spl_feat)
        diff_cl = torch.abs(ctx_feat - liv_feat)
        diff_cs = torch.abs(ctx_feat - spl_feat)
        stats_feat = self.stats_mlp(stats)

        fused = torch.cat(
            [ctx_feat, liv_feat, spl_feat, diff_ls, diff_cl, diff_cs, stats_feat],
            dim=1,
        )
        return self.head(fused).squeeze(1)


def compute_metrics(labels, probs):
    labels = np.asarray(labels).astype(np.uint8)
    probs = np.asarray(probs, dtype=np.float32)

    fpr, tpr, thr = roc_curve(labels, probs)
    best_idx = np.argmax(tpr - fpr)
    best_thr = thr[best_idx]
    preds = (probs >= best_thr).astype(np.uint8)

    return {
        "acc": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)),
        "recall": float(tpr[best_idx]),
        "best_thr": float(best_thr),
    }


def collect_rows(dataset_csv):
    df = pd.read_csv(dataset_csv)
    rows = df.to_dict("records")
    print("Total samples:", len(rows))
    print("Class counts:", df["label"].value_counts().sort_index().to_dict())
    return rows


def train_one_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, fold):
    best_auc = -1.0
    no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_probs, tr_labels = [], []

        for context, liver, spleen, stats, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"):
            context = context.to(DEVICE)
            liver = liver.to(DEVICE)
            spleen = spleen.to(DEVICE)
            stats = stats.to(DEVICE)
            labels = labels.to(DEVICE)

            target_labels = labels
            if np.random.rand() < MIXUP_PROB:
                context, liver, spleen, stats, target_labels = apply_mixup(
                    context, liver, spleen, stats, labels
                )

            smooth_labels = target_labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            logits = model(context, liver, spleen, stats)
            loss = criterion(logits, smooth_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            tr_loss += loss.item() * context.size(0)
            tr_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            tr_labels.extend(labels.cpu().numpy())

        tr_loss /= max(len(tr_labels), 1)
        train_metrics = compute_metrics(tr_labels, tr_probs)

        model.eval()
        val_loss = 0.0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for context, liver, spleen, stats, labels in val_loader:
                context = context.to(DEVICE)
                liver = liver.to(DEVICE)
                spleen = spleen.to(DEVICE)
                stats = stats.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(context, liver, spleen, stats)
                loss = criterion(logits, labels)

                val_loss += loss.item() * context.size(0)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= max(len(val_labels), 1)
        val_metrics = compute_metrics(val_labels, val_probs)
        scheduler.step(val_metrics["auc"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc"],
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "Recall": val_metrics["recall"],
                "best_thr": val_metrics["best_thr"],
            }
        )

        print(
            f"Fold {fold} Epoch {epoch} | "
            f"TrainLoss={tr_loss:.4f} ValLoss={val_loss:.4f} | "
            f"TrainAcc={train_metrics['acc']:.4f} ValAcc={val_metrics['acc']:.4f} | "
            f"TrainAUC={train_metrics['auc']:.4f} ValAUC={val_metrics['auc']:.4f} | "
            f"Recall={val_metrics['recall']:.4f} | "
            f"Thr={val_metrics['best_thr']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"best_model_fold_{fold}.pth"))
            no_improve = 0
            print(f"  -> Fold {fold} best model saved (val_AUC improved to {best_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return pd.DataFrame(history)


def save_fold_outputs(df, fold):
    df.to_csv(os.path.join(RESULTS_DIR, f"metrics_full_fold_{fold}.csv"), index=False)

    best_row = df.loc[df["val_auc"].idxmax()]
    summary = pd.DataFrame(
        [best_row, df.mean(numeric_only=True), df.median(numeric_only=True)],
        index=["best_epoch", "mean", "median"],
    )
    summary.to_csv(os.path.join(RESULTS_DIR, f"metrics_summary_fold_{fold}.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(df["train_loss"], label="Train Loss")
    ax[0].plot(df["val_loss"], label="Val Loss")
    ax[0].set_title(f"Fold {fold} - Loss")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(df["train_acc"], label="Train Acc")
    ax[1].plot(df["val_acc"], label="Val Acc")
    ax[1].set_title(f"Fold {fold} - Accuracy")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"fold_{fold}_loss_acc.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_auc"], label="Train AUC")
    plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"Auc_{fold}_plot.png"))
    plt.close()


def main():
    rows = collect_rows(DATASET_CSV)
    feature_lookup = build_feature_lookup(FEATURES_CSV)
    labels_all = [row["label"] for row in rows]

    debug_dir = Path(RESULTS_DIR) / "debug_inputs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for p in debug_dir.glob("sample_*.png"):
        p.unlink()

    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(rows, labels_all), 1):
        print(f"\n========== FOLD {fold}/{KFOLDS} ==========")

        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        stats_mean, stats_std = normalize_stats(train_rows, feature_lookup)

        train_loader = DataLoader(
            LiverSpleenMultiViewDataset(
                train_rows,
                feature_lookup,
                stats_mean,
                stats_std,
                transform=train_tfms,
                debug=True,
                debug_max=8,
                debug_dir=debug_dir,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            LiverSpleenMultiViewDataset(
                val_rows,
                feature_lookup,
                stats_mean,
                stats_std,
                transform=val_tfms,
                debug_dir=debug_dir,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        model = MultiViewEfficientNetB0().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()

        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone.features."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": LR * BACKBONE_LR_SCALE},
                {"params": head_params, "lr": LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        fold_df = train_one_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, fold)
        save_fold_outputs(fold_df, fold)

    print("K-Fold training complete.")


if __name__ == "__main__":
    main()

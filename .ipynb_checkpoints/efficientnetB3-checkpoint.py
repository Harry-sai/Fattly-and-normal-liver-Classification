# ============================================================
# Liver Classification with Stratified K-Fold Cross Validation
# ============================================================

import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

all_folds_metrics = [] 


# ============================================================
# CONFIG
# ============================================================
IMAGES_ROOT = "data/images/"
MASKS_ROOT  = "data/predictions_higherepoch/"
RESULTS_DIR = "results/efficientnet_b3"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS_TO_USE = 2
KFOLDS = 5
SEED = 42
PATIENCE = 10

# ============================================================
# SEED
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ============================================================
# AUGMENTATIONS
# ============================================================
train_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    A.Rotate(limit=5, p=0.4),
    A.Normalize(mean=(0.5,), std=(0.25,)),
    ToTensorV2()
], additional_targets={"mask": "mask"})

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5,), std=(0.25,)),
    ToTensorV2()
], additional_targets={"mask": "mask"})

# ============================================================
# DATASET
# ============================================================
class LiverROICLSDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if mask.shape != img.shape:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (img.shape[1], img.shape[0]),
                    resample=Image.NEAREST
                )
            )

        mask = (mask > 127).astype(np.uint8)
        img = img * mask

        aug = self.transform(image=img, mask=mask)
        img = aug["image"]

        img = img.repeat(3, 1, 1)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

# ============================================================
# DATA COLLECTION
# ============================================================
def collect_labeled_pairs(images_root=IMAGES_ROOT, masks_root=MASKS_ROOT):
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    pairs = []
    mask_lookup = {}

    # Build mask lookup
    for m in masks_root.rglob("*"):
        if not m.is_file():
            continue
        if m.suffix.lower() != ".png":
            continue
    
        cls = m.parent.name.lower()
        if cls not in ["fatty_liver", "normal"]:
            continue
    
        key = (cls, m.stem.lower())
        mask_lookup[key] = m
    
    
    # Match images
    for img in images_root.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower() != ".png":
            continue
    
        cls = img.parent.name.lower()
        if cls not in ["fatty_liver", "normal"]:
            continue
    
        key = (cls, img.stem.lower())
        if key in mask_lookup:
            label = 1 if cls == "fatty_liver" else 0
            pairs.append((str(img), str(mask_lookup[key]), label))


    labels = [p[2] for p in pairs]
    print("Total samples found:", len(pairs))
    print("GLOBAL CLASS COUNTS:", np.bincount(labels))

    return pairs




samples = collect_labeled_pairs(IMAGES_ROOT, MASKS_ROOT)
print("GLOBAL CLASS COUNTS:", np.bincount([s[2] for s in samples]))
labels_all = [s[2] for s in samples]

# ============================================================
# STRATIFIED K-FOLD
# ============================================================
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels_all), 1):
    print(f"\n========== FOLD {fold}/{KFOLDS} ==========")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    history = []   # <<< ADD THIS (per-fold epoch history)

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    print("VAL CLASS COUNTS:", np.bincount([s[2] for s in val_samples]))

    train_labels = [s[2] for s in train_samples]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        LiverROICLSDataset(train_samples, train_tfms),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=1,
        pin_memory=False
    )

    val_loader = DataLoader(
        LiverROICLSDataset(val_samples, val_tfms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    if torch.cuda.device_count() >= NUM_GPUS_TO_USE and NUM_GPUS_TO_USE > 1:
        model = nn.DataParallel(model)

    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=3
    )

    best_auc = -1
    no_improve = 0

    # ====================== EPOCH LOOP ======================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_preds, tr_gt, tr_loss = [], [], 0

        for imgs, lbls in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            logits = model(imgs).squeeze()
            loss = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            tr_gt.extend(lbls.cpu().numpy())

        tr_loss /= len(tr_gt)
        train_acc = accuracy_score(tr_gt, np.array(tr_preds) > 0.5)

        if len(np.unique(tr_gt)) > 1:
            train_auc = roc_auc_score(tr_gt, tr_preds)
        else:
            train_auc = np.nan


        train_losses.append(tr_loss)
        train_accs.append(train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_preds, val_gt, val_loss = [], [], 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

                logits = model(imgs).squeeze()
                loss = criterion(logits, lbls)

                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_gt.extend(lbls.cpu().numpy())

        val_loss /= len(val_gt)
        val_acc = accuracy_score(val_gt, np.array(val_preds) > 0.5)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # -------- SAFE AUC --------
        if len(np.unique(val_gt)) > 1:
            val_auc = roc_auc_score(val_gt, val_preds)
            scheduler.step(val_auc)
        else:
            val_auc = np.nan

        # -------- SAVE EPOCH HISTORY --------
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_auc": train_auc,
            "val_auc": val_auc
        })


        print(
            f"Fold {fold} Epoch {epoch} | "
            f"TrainLoss={tr_loss:.4f} TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} | ValAUC={val_auc:.4f}"
        )

        # -------- EARLY STOP --------
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.module.state_dict() if isinstance(model, nn.DataParallel)
                else model.state_dict(),
                os.path.join(RESULTS_DIR, f"best_model_fold_{fold}.pth")
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping fold {fold} at epoch {epoch}")
            break

    # ====================== AFTER FOLD ======================

    df_fold = pd.DataFrame(history)

    # ---- BEST / MEAN / MEDIAN ----
    best_row = df_fold.loc[df_fold["val_loss"].idxmin()]
    mean_row = df_fold.mean(numeric_only=True)
    median_row = df_fold.median(numeric_only=True)

    summary_df = pd.DataFrame([best_row, mean_row, median_row])
    summary_df.index = ["best_epoch", "mean", "median"]

    summary_csv = os.path.join(
        RESULTS_DIR,
        f"metrics_summary_fold_{fold}.csv"
    )
    summary_df.to_csv(summary_csv)
    print(f"Saved summary metrics → {summary_csv}")

    # ---- FULL HISTORY CSV (optional but recommended) ----
    df_fold.to_csv(
        os.path.join(RESULTS_DIR, f"metrics_full_fold_{fold}.csv"),
        index=False
    )

    # ---- PLOT LOSS & ACC ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(val_losses, label="Val Loss")
    ax[0].set_title(f"Fold {fold} - Loss")
    ax[0].legend()

    ax[1].plot(train_accs, label="Train Acc")
    ax[1].plot(val_accs, label="Val Acc")
    ax[1].set_title(f"Fold {fold} - Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"fold_{fold}_loss_acc.png"))
    plt.close()

    # ---- PLOT TRAIN vs VAL AUC ----
    plt.figure(figsize=(6, 4))
    plt.plot(df_fold["epoch"], df_fold["train_auc"], label="Train AUC")
    plt.plot(df_fold["epoch"], df_fold["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"Fold {fold} - AUC")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"fold_{fold}_auc.png")
    )
    plt.close()


print("K-Fold training complete.")

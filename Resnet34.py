# ============================================================
# Liver Classification 
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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
IMAGES_ROOT = "data/images/"
MASKS_ROOT  = "data/masks/"
RESULTS_DIR = "results/resnet34/Augm"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KFOLDS = 5
SEED = 42
PATIENCE = 7
DROP_OUT=0.3
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
# AUGMENTATIONS (CT-SAFE)

train_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.15),   # low probability

    A.Affine(
        translate_percent=0.05,
        scale=(0.95, 1.05),
        rotate=8,
        border_mode=0,
        p=0.4
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.05,
        contrast_limit=0.05,
        p=0.3
    ),

    A.Normalize(mean=(0.5,), std=(0.25,)),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5,), std=(0.25,)),
    ToTensorV2()
])

# ============================================================
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Run configuration to save parameters
# -------------------------
RUN_CONFIG = {
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "KFOLDS": KFOLDS,
    "AUGMENTATIONS": str(train_tfms),
    "Model": "ResNet34",
    "DROP_OUT":DROP_OUT,
    "OPTIMIZER": "adam with Lr schedular",
    "LOSS": "BCEWithLogitsLoss",
    "COMMENT":""" training full backbone
            """
}

config_df = pd.DataFrame.from_dict(RUN_CONFIG, orient="index", columns=["value"])
config_df.to_csv(os.path.join(RESULTS_DIR, "run_config.csv"))

DEBUG_DIR = os.path.join(RESULTS_DIR, "debug_inputs")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ============================================================
# DATASET (ROI CROP, SINGLE CHANNEL)
# ============================================================
class LiverROICLSDataset(Dataset):
    def __init__(self, samples, transform=None, debug=False, debug_max=10):
        self.samples = samples
        self.transform = transform
        self.debug = debug
        self.debug_max = debug_max
        self.saved = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img  = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if mask.shape != img.shape:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (img.shape[1], img.shape[0]),
                    resample=Image.NEAREST
                )
            )

        mask = (mask > 127).astype(np.uint8)

        # ---------- ROI CROP ----------
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise RuntimeError(f"Empty mask: {mask_path}")

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        img = img[y1:y2, x1:x2]

        aug = self.transform(image=img)
        img_tensor = aug["image"]            # [1, H, W]

        # ---------- DEBUG SAVE ----------
        if self.debug and self.saved < self.debug_max:
            vis = img_tensor.clone()
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            vis = (vis * 255).byte().squeeze(0).cpu().numpy()

            save_path = os.path.join(
                DEBUG_DIR, f"sample_{self.saved}_label_{label}.png"
            )
            Image.fromarray(vis).save(save_path)
            self.saved += 1

        label = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label

# ============================================================
# DATA COLLECTION
# ============================================================
def collect_labeled_pairs(images_root, masks_root):
    images_root = Path(images_root)
    masks_root  = Path(masks_root)

    pairs = []
    mask_lookup = {}

    for m in masks_root.rglob("*.PNG"):
        cls = m.parent.name.lower()
        if cls in ["fatty_liver", "normal"]:
            mask_lookup[(cls, m.stem.lower())] = m

    for img in images_root.rglob("*.PNG"):
        cls = img.parent.name.lower()
        if cls in ["fatty_liver", "normal"]:
            key = (cls, img.stem.lower())
            if key in mask_lookup:
                label = 1 if cls == "fatty_liver" else 0
                pairs.append((str(img), str(mask_lookup[key]), label))

    print("Total samples:", len(pairs))
    print("Class counts:", np.bincount([p[2] for p in pairs]))
    return pairs

samples = collect_labeled_pairs(IMAGES_ROOT, MASKS_ROOT)
labels_all = [s[2] for s in samples]


# ============================================================
# STRATIFIED K-FOLD
# ============================================================
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels_all), 1):
    print(f"\n========== FOLD {fold}/{KFOLDS} ==========")

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

    train_labels = [s[2] for s in train_samples]
    class_counts = np.bincount(train_labels)
    print("Class Count",class_counts)

    pos_weight = torch.tensor(
        class_counts[0] / class_counts[1]
    ).to(DEVICE)

    train_loader = DataLoader(
        LiverROICLSDataset(
            train_samples,
            train_tfms,
            debug=True,      
            debug_max=12     
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

    val_loader = DataLoader(
        LiverROICLSDataset(val_samples, val_tfms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

# ========================================================
# MODEL (SINGLE CHANNEL Densenet121)
# ========================================================
    model = models.resnet34(weights="IMAGENET1K_V1")
    
    # ---- change first conv layer to accept 1 channel ----
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    
    # ---- Freeze backbone ----
    for param in model.parameters():
        param.requires_grad = False
    
    # for param in model.layer2.parameters():
    #     param.requires_grad = True
        
    # for param in model.layer3.parameters():
    #     param.requires_grad = True
        
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    
    # ---- Replace classifier ----
    in_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(DROP_OUT),
        nn.Linear(512, 1)
    )
    
    model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_auc = -1
    no_improve = 0
    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================
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

        scheduler.step()

        tr_loss /= len(tr_gt)
        train_acc = accuracy_score(tr_gt, np.array(tr_preds) > 0.5)
        train_auc = roc_auc_score(tr_gt, tr_preds)

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
        val_auc = roc_auc_score(val_gt, val_preds)
        
        fpr, tpr, thr = roc_curve(val_gt, val_preds)
        
        # Youden’s J statistic
        best_idx = np.argmax(tpr - fpr)
        best_thr = thr[best_idx]
        best_recall = tpr[best_idx]
        val_acc = accuracy_score(val_gt, np.array(val_preds) > best_thr)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "Recall":best_recall
        })

        print(
            f"Fold {fold} Epoch {epoch} | "
            f"TrainLoss={tr_loss:.4f} ValLoss={val_loss:.4f} | "
            f"Train_acc={train_acc:.4f} Val_acc={val_acc:.4f} |"
            f"TrainAUC={train_auc:.4f} ValAUC={val_auc:.4f}|"
            f"Recall={best_recall:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, f"best_model_fold_{fold}.pth")
            )
            no_improve = 0
            print(f"  -> Fold {fold} best model saved (val_AUC improved to {best_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # ========================================================
    # SAVE METRICS & PLOTS (UNCHANGED STRUCTURE)
    # ========================================================
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULTS_DIR, f"metrics_full_fold_{fold}.csv"), index=False)

    best_row = df.loc[df["val_auc"].idxmax()]
    summary = pd.DataFrame([best_row, df.mean(), df.median()],
                           index=["best_epoch", "mean", "median"])
    summary.to_csv(os.path.join(RESULTS_DIR, f"metrics_summary_fold_{fold}.csv"))

    # ---- PLOT LOSS & ACC ----
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
    
    plt.figure(figsize=(6,4))
    plt.plot(df["epoch"], df["train_auc"], label="Train AUC")
    plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"Auc_{fold}_plot.png"))
    plt.close()

print("K-Fold training complete.")

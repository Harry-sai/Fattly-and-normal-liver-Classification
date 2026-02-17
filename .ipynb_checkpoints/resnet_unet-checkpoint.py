# unet_train_eval_aug_kfold_fixed.py
# Requirements: torch, torchvision, pillow, numpy, matplotlib, tqdm, albumentations, opencv-python, scikit-learn, pandas
# Usage: python unet_train_eval_aug_kfold_fixed.py

import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
import pandas as pd

# -------------------------
# CONFIG (edit if needed)
# -------------------------
IMAGES_ROOT      = "data2/images"
LABEL_MASKS_ROOT = "data2/labelled"
RESULTS_DIR      = "data2/results2"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
THRESH = 0.5
NUM_GPUS_TO_USE = 1  # change as needed (set to 1 to avoid DataParallel)
KFOLDS = 5
SEED = 42

# Optional: avoid some NCCL issues on certain clusters (uncomment if needed)
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'

# -------------------------
# Albumentations transforms (paired)
# -------------------------
train_transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.6),
    A.Rotate(limit=12, border_mode=0, p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
    A.Resize(IMG_SIZE, IMG_SIZE, p=1.0),
], additional_targets={"mask": "mask"})

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE)
], additional_targets={"mask": "mask"})

# -------------------------
# Utilities
# -------------------------
def center_crop_to(tensor, target_h, target_w):
    """Center-crop a 4D tensor (B,C,H,W) or 3D tensor (C,H,W) to (target_h, target_w)."""
    if tensor.ndim == 4:
        _, _, h, w = tensor.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return tensor[:, :, top:top+target_h, left:left+target_w]
    elif tensor.ndim == 3:
        _, h, w = tensor.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return tensor[:, top:top+target_h, left:left+target_w]
    else:
        return tensor

# -------------------------
# Dataset using albumentations for paired transforms
# -------------------------
class LiverMaskDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, transform=None):
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get the single pair (img_path, mask_path)
        img_path, mask_path = self.pairs[idx]

        # load as grayscale numpy arrays
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        # ensure same H,W before albumentations (nearest for mask)
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = np.array(Image.fromarray(mask).resize((w, h), resample=Image.NEAREST))

        # apply paired transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = A.Resize(self.img_size, self.img_size)(image=img)["image"]
            mask = A.Resize(self.img_size, self.img_size)(image=mask)["image"]
            mask = (mask > 127).astype(np.uint8)

        # normalize and convert to tensors (C,H,W) and (1,H,W)
        img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor

# -------------------------
# Metrics and Loss
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1 - (2. * inter + eps) / (pred.sum() + target.sum() + eps)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + dice_loss(pred, target)

def compute_metrics_np(pred_bin, gt_bin):
    pred = pred_bin.astype(np.uint8).ravel()
    gt = gt_bin.astype(np.uint8).ravel()
    TP = int(((pred == 1) & (gt == 1)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    eps = 1e-8
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou, "dice": dice}

# -------------------------
# Collect pairs
# -------------------------
def collect_labeled_pairs(images_root=IMAGES_ROOT, masks_root=LABEL_MASKS_ROOT):
    images_root = Path(images_root)
    masks_root = Path(masks_root)
    pairs = []
    missing = []

    # Walk all files under images_root (any depth)
    for img_path in sorted(images_root.rglob('*')):
        if not img_path.is_file():
            continue
        # infer class from immediate parent folder name
        class_name = img_path.parent.name
        # expected mask path (same filename) under masks_root/class_name/
        expected_mask = masks_root / class_name / img_path.name

        if expected_mask.exists():
            pairs.append((str(img_path), str(expected_mask)))
            continue

        # fallback: try to find any mask file with same basename anywhere under masks_root/class_name or masks_root
        basename = img_path.name
        found = None
        search_root = masks_root / class_name if (masks_root / class_name).exists() else masks_root
        for m in search_root.rglob('*'):
            if not m.is_file():
                continue
            if m.name == basename:
                found = m
                break
        if found:
            pairs.append((str(img_path), str(found)))
        else:
            # last resort: search entire masks_root for matching basename (case-insensitive)
            for m in masks_root.rglob('*'):
                if not m.is_file():
                    continue
                if m.name.lower() == basename.lower():
                    found = m
                    break
            if found:
                pairs.append((str(img_path), str(found)))
            else:
                missing.append((str(img_path), str(expected_mask)))

    if len(pairs) == 0:
        print("No pairs found. Example missing pairs (first 20):")
        for i, (img, mask) in enumerate(missing[:20]):
            print(i, img, "->", mask)
    else:
        print(f"Found {len(pairs)} image-mask pairs (missing examples: {len(missing)})")
    return pairs

# -------------------------
# UNet with ResNet34 encoder (pretrained)
# -------------------------
class UnetResNet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        # handle torchvision weights deprecation safely
        resnet_kwargs = {}
        try:
            ResNetWeights = models.ResNet34_Weights  # type: ignore
            resnet = models.resnet34(weights=ResNetWeights.DEFAULT if pretrained else None)
        except Exception:
            # older torchvision fallback
            resnet = models.resnet34(pretrained=pretrained)

        # adapt first conv to accept in_channels (grayscale -> 1)
        if in_channels != 3:
            w = resnet.conv1.weight.data  # shape (64,3,7,7)
            # average across input channel dim to get a single-channel kernel
            w_mean = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_channels == 1:
                new_conv.weight.data = w_mean
            else:
                # repeat averaged kernel across required channels
                new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1)
            resnet.conv1 = new_conv

        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # output stride 2
        self.maxpool = resnet.maxpool  # further downsample
        # encoder layers
        self.encoder1 = resnet.layer1  # out: 64
        self.encoder2 = resnet.layer2  # out: 128
        self.encoder3 = resnet.layer3  # out: 256
        self.encoder4 = resnet.layer4  # out: 512

        # decoder (upsampling + conv blocks)
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # decoder channel sizes tuned for ResNet34 encoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(64 + 64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # x: B,1,H,W
        x0 = self.inc(x)         # B,64,H/2,W/2
        x1 = self.maxpool(x0)    # B,64,H/4,W/4
        e1 = self.encoder1(x1)   # B,64,H/4,W/4
        e2 = self.encoder2(e1)   # B,128,H/8,W/8
        e3 = self.encoder3(e2)   # B,256,H/16,W/16
        e4 = self.encoder4(e3)   # B,512,H/32,W/32

        u4 = self.up4(e4)        # B,256,H/16,W/16
        # center-crop e3 to match u4 if needed
        if u4.shape[2:] != e3.shape[2:]:
            e3_c = center_crop_to(e3, u4.shape[2], u4.shape[3])
        else:
            e3_c = e3
        d4 = self.dec4(torch.cat([u4, e3_c], dim=1))

        u3 = self.up3(d4)        # B,128,H/8,W/8
        if u3.shape[2:] != e2.shape[2:]:
            e2_c = center_crop_to(e2, u3.shape[2], u3.shape[3])
        else:
            e2_c = e2
        d3 = self.dec3(torch.cat([u3, e2_c], dim=1))

        u2 = self.up2(d3)        # B,64,H/4,W/4
        if u2.shape[2:] != e1.shape[2:]:
            e1_c = center_crop_to(e1, u2.shape[2], u2.shape[3])
        else:
            e1_c = e1
        d2 = self.dec2(torch.cat([u2, e1_c], dim=1))

        u1 = self.up1(d2)        # B,64,H/2,W/2
        if u1.shape[2:] != x0.shape[2:]:
            x0_c = center_crop_to(x0, u1.shape[2], u1.shape[3])
        else:
            x0_c = x0
        d1 = self.dec1(torch.cat([u1, x0_c], dim=1))

        out = torch.sigmoid(self.final(d1))
        return out

# -------------------------
# Training & evaluation per fold
# -------------------------
def train_and_evaluate_fold(model, train_loader, val_loader, fold_index, epochs=EPOCHS, save_dir=RESULTS_DIR):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = []
    for epoch in range(1, epochs+1):
        model.train()
        train_running = 0.0
        train_acc_sum = 0.0
        n_train_samples = 0

        for imgs, masks in tqdm(train_loader, desc=f"Fold {fold_index} Train Epoch {epoch}/{epochs}"):
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)

            # forward
            preds = model(imgs)

            # ensure preds and masks spatial sizes match
            if preds.shape[2:] != masks.shape[2:]:
                preds = F.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)

            # compute train accuracy bookkeeping
            pred_bin = (preds.detach() > THRESH).float()
            correct = (pred_bin == masks).float().mean().item()
            train_acc_sum += correct * imgs.size(0)
            n_train_samples += imgs.size(0)

            loss = criterion(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            train_running += loss.item() * imgs.size(0)

        train_loss = train_running / max(len(train_loader.dataset), 1)
        train_accuracy = train_acc_sum / max(n_train_samples, 1)

        # validation
        model.eval()
        val_running = 0.0
        metric_sums = {"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"iou":0.0,"dice":0.0}
        total_samples = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                preds = model(imgs)

                # match sizes
                if preds.shape[2:] != masks.shape[2:]:
                    preds = F.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)

                val_running += criterion(preds, masks).item() * imgs.size(0)
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                bs = preds_np.shape[0]
                total_samples += bs
                for b in range(bs):
                    p = preds_np[b,0]
                    g = masks_np[b,0]
                    p_bin = (p > THRESH).astype(np.uint8)
                    g_bin = (g > 0.5).astype(np.uint8)
                    m = compute_metrics_np(p_bin, g_bin)
                    for k in metric_sums:
                        metric_sums[k] += m[k]

        val_loss = val_running / max(len(val_loader.dataset), 1)
        avg_metrics = {k: (metric_sums[k] / (total_samples + 1e-12)) for k in metric_sums}

        history.append({
            "fold": fold_index,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": avg_metrics["accuracy"],
            "val_precision": avg_metrics["precision"],
            "val_recall": avg_metrics["recall"],
            "val_f1": avg_metrics["f1"],
            "val_iou": avg_metrics["iou"],
            "val_dice": avg_metrics["dice"],
        })

        print(f"Fold {fold_index} Epoch {epoch}: val_acc={avg_metrics['accuracy']:.4f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={avg_metrics['f1']:.4f}, val_dice={avg_metrics['dice']:.4f}")

        # save best model for this fold (handle DataParallel case)
        model_path = os.path.join(save_dir, f"best_unet_fold_{fold_index}.pth")
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state_dict_to_save, model_path)
            epochs_no_improve = 0
            print(f"  -> Fold {fold_index} best model saved (val_loss improved to {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping fold {fold_index} after {epoch} epochs (no improvement for {PATIENCE} epochs).")
            break

    # load best model (map to DEVICE)
    model_path = os.path.join(save_dir, f"best_unet_fold_{fold_index}.pth")
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=DEVICE)
        # if current model is DataParallel, load into module
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(sd)
        else:
            model.load_state_dict(sd)

    return model, history

# -------------------------
# Plotting: two-subplot and loss+accuracy dual-axis
# -------------------------
def plot_two_subplots(history_df, out_path):
    epochs = history_df["epoch"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Subplot 1: Accuracy ----
    ax1.plot(epochs, history_df["train_accuracy"], label="train_accuracy")
    ax1.plot(epochs, history_df["val_accuracy"], label="val_accuracy")
    ax1.set_title("Training vs Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True)
    ax1.legend()

    # ---- Subplot 2: Loss ----
    ax2.plot(epochs, history_df["train_loss"], label="train_loss")
    ax2.plot(epochs, history_df["val_loss"], label="val_loss")
    ax2.set_title("Training vs Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_and_accuracy(history_df, out_path):
    # dual y-axis plot: train/val loss (left) and val accuracy (right)
    x = history_df["epoch"].tolist()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, history_df["train_loss"], label="train_loss")
    ax.plot(x, history_df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(x, history_df["val_accuracy"], label="val_accuracy", linestyle="--")
    ax2.set_ylabel("Accuracy")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved combined loss+accuracy plot to {out_path}")

# -------------------------
# Main: K-Fold orchestration
# -------------------------
def main():
    pairs = collect_labeled_pairs()
    assert len(pairs) > 0, "No labeled pairs found. Check folders."
    print(f"Total labeled pairs: {len(pairs)}")

    # deterministic shuffle
    random.seed(SEED)
    random.shuffle(pairs)

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

    all_history = []
    fold_idx = 0
    for train_idx, val_idx in kf.split(pairs):
        fold_idx += 1
        print(f"\n=== Starting fold {fold_idx}/{KFOLDS} ===")
        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]

        train_ds = LiverMaskDataset(train_pairs, transform=train_transform)
        val_ds = LiverMaskDataset(val_pairs, transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # build model with pretrained encoder
        model = UnetResNet34(in_channels=1, out_channels=1, pretrained=True)

        # user controlled multi-GPU (DataParallel)
        available_gpus = torch.cuda.device_count()
        if available_gpus >= NUM_GPUS_TO_USE and NUM_GPUS_TO_USE > 1:
            gpu_ids = list(range(NUM_GPUS_TO_USE))
            print(f"Using GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            print("Using a single GPU or CPU.")

        model, history = train_and_evaluate_fold(model, train_loader, val_loader, fold_index=fold_idx, epochs=EPOCHS, save_dir=RESULTS_DIR)

        # save per-fold history to csv
        df_fold = pd.DataFrame(history)
        csv_fold = os.path.join(RESULTS_DIR, f"metrics_fold_{fold_idx}.csv")
        df_fold.to_csv(csv_fold, index=False)
        print(f"Saved fold {fold_idx} metrics to {csv_fold}")

        # combined per-fold plot
        plot_two_subplots(df_fold, os.path.join(RESULTS_DIR, f"two_subplots_fold_{fold_idx}.png"))

        all_history.append(df_fold)

    # aggregate all folds
    all_df = pd.concat(all_history, ignore_index=True)
    all_csv = os.path.join(RESULTS_DIR, "results_metrics_all_folds.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved all folds metrics to {all_csv}")

    # overall combined plot (mean per epoch across folds)
    mean_by_epoch = all_df.groupby("epoch").mean().reset_index()
    plot_two_subplots(mean_by_epoch, os.path.join(RESULTS_DIR, "two_subplots_all_folds_mean.png"))
    plot_loss_and_accuracy(mean_by_epoch, os.path.join(RESULTS_DIR, "loss_acc_all_folds_mean.png"))

    print("K-Fold training complete. Best models saved per fold in:", RESULTS_DIR)

if __name__ == "__main__":
    main()


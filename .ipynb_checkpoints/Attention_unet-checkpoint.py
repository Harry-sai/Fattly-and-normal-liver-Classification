import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import albumentations as A

from sklearn.model_selection import KFold
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------------
# CONFIG 
# -------------------------
IMAGES_ROOT      = "data/images"
LABEL_MASKS_ROOT = "data/labelled"
RESULTS_DIR      = "Attention/out_sgd1"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 320
BATCH_SIZE = 16
EPOCHS =60
WEIGHT_DECAY = 5e-5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
THRESH = 0.5
NUM_GPUS_TO_USE = 1
KFOLDS = 5
SEED = 42
# dropout probability applied inside decoder conv blocks
DROPOUT_P = 0.0


# toggle to save a few augmented samples to RESULTS_DIR/aug_debug for visual check
SAVE_AUG_SAMPLES = False
AUG_DEBUG_OUT = Path(RESULTS_DIR) / "aug_debug"
if SAVE_AUG_SAMPLES:
    AUG_DEBUG_OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Albumentations transforms (paired)
# -------------------------
train_transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.8),
    A.Rotate(limit=20, border_mode=0, p=0.6),
    A.ElasticTransform(alpha=0.5, sigma=40, p=0.35),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.4),
    A.Resize(IMG_SIZE, IMG_SIZE, p=1.0),
], additional_targets={"mask": "mask"})

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE)
], additional_targets={"mask": "mask"}) 

# -------------------------
# Run configuration to save parameters
# -------------------------
RUN_CONFIG = {
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE(bs)": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "THRESH": THRESH,
    "DROPOUT_P": DROPOUT_P,
    "KFOLDS": KFOLDS,
    "AUGMENTATIONS": str(train_transform),
    "ENCODER": "Attention Unet ",
    "OPTIMIZER": "AdamW",
    "LOSS": "BCE ",
    "COMMENT":"""
            SGD may work so trying it lr was too high decreasing LR 
            momentum -> 0.7
            """
}

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
# Dataset using albumentations for paired transforms (robust handling)
# -------------------------
class LiverMaskDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, transform=None, save_aug=False, aug_out=None):
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform
        self.save_aug = save_aug
        self.aug_out = Path(aug_out) if aug_out is not None else None

    def __len__(self):
        return len(self.pairs)

    def _ensure_mask_binary(self, mask):
        # mask may be HxW or HxWx1, float or uint8
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.array(mask)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        # threshold
        mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0.5).astype(np.uint8)
        return mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # read grayscale as numpy uint8
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        # ensure same H,W BEFORE albumentations using nearest for mask
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = np.array(Image.fromarray(mask).resize((w, h), resample=Image.NEAREST))

        # apply paired transforms (albumentations returns ndarray by default)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_aug = augmented["image"]
            mask_aug = augmented["mask"]
        else:
            img_aug = A.Resize(self.img_size, self.img_size)(image=img)["image"]
            mask_aug = A.Resize(self.img_size, self.img_size)(image=mask)["image"]

        # albumentations may return float image in 0..255 or 0..1; normalize robustly
        img_aug = np.array(img_aug)
        # if image has channel dim HxWx1 -> squeeze
        if img_aug.ndim == 3 and img_aug.shape[2] == 1:
            img_aug = img_aug[:, :, 0]

        # normalize image to 0..1 float32
        if img_aug.dtype == np.uint8:
            img_aug = img_aug.astype(np.float32) / 255.0
        else:
            img_aug = img_aug.astype(np.float32)
            if img_aug.max() > 1.5:  # likely 0..255 floats
                img_aug = img_aug / 255.0

        # fix mask to be binary HxW
        mask_aug = self._ensure_mask_binary(mask_aug)

        # optional: save a few augmented samples for visual debug
        if self.save_aug and idx < 8:
            img_vis = (img_aug * 255).astype(np.uint8)
            if img_vis.ndim == 2:
                Image.fromarray(img_vis).save(str(self.aug_out / f"img_{idx}.png"))
            else:
                Image.fromarray(img_vis[:, :, 0]).save(str(self.aug_out / f"img_{idx}.png"))
            Image.fromarray((mask_aug * 255).astype(np.uint8)).save(str(self.aug_out / f"mask_{idx}.png"))

        # final resize in case transforms didn't
        if img_aug.shape[0] != self.img_size or img_aug.shape[1] != self.img_size:
            img_aug = np.array(Image.fromarray((img_aug * 255).astype(np.uint8)).resize((self.img_size, self.img_size))).astype(np.float32) / 255.0
        if mask_aug.shape[0] != self.img_size or mask_aug.shape[1] != self.img_size:
            mask_aug = np.array(Image.fromarray((mask_aug * 255).astype(np.uint8)).resize((self.img_size, self.img_size), resample=Image.NEAREST))
            mask_aug = (mask_aug > 127).astype(np.uint8)

        # ensure channel dim
        if img_aug.ndim == 2:
            img_aug = np.expand_dims(img_aug, axis=2)

        # convert to tensors
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()  # C,H,W
        mask_tensor = torch.from_numpy(mask_aug).unsqueeze(0).float()    # 1,H,W

        return img_tensor, mask_tensor

# -------------------------
# Loss & metrics
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
# Collect pairs (robust)
# -------------------------
def collect_labeled_pairs(images_root=IMAGES_ROOT, masks_root=LABEL_MASKS_ROOT):
    images_root = Path(images_root)
    masks_root = Path(masks_root)
    pairs = []
    missing = []
    for img_path in sorted(images_root.rglob('*')):
        if not img_path.is_file():
            continue
        class_name = img_path.parent.name
        expected_mask = masks_root / class_name / img_path.name
        if expected_mask.exists():
            pairs.append((str(img_path), str(expected_mask)))
            continue
        # quick fallback: search same-class folder
        found = None
        search_root = masks_root / class_name if (masks_root / class_name).exists() else masks_root
        for m in search_root.rglob('*'):
            if not m.is_file():
                continue
            if m.name == img_path.name:
                found = m; break
        if found:
            pairs.append((str(img_path), str(found)))
            continue
        # global case-insensitive fallback
        for m in masks_root.rglob('*'):
            if not m.is_file(): continue
            if m.name.lower() == img_path.name.lower():
                found = m; break
        if found:
            pairs.append((str(img_path), str(found)))
        else:
            missing.append((str(img_path), str(expected_mask)))

    if len(pairs) == 0:
        print("No pairs found. Example missing (first 20):")
        for i, (im, ma) in enumerate(missing[:20]):
            print(i, im, "->", ma)
    else:
        print(f"Found {len(pairs)} image-mask pairs (missing {len(missing)}).")
    return pairs

# -------------------------
# AttentionUnet
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_ch, out_ch,dropout_p=DROPOUT_P):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout_p and dropout_p > 0:
                layers.insert(-1, nn.Dropout2d(dropout_p)) 
            return nn.Sequential(*layers)

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.up4(e4)
        e3 = self.att4(d4, e3)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        e2 = self.att3(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        e1 = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.final(d2))


# -------------------------
# Training per-fold
# -------------------------
def train_and_evaluate_fold(model, train_loader, val_loader, fold_index, epochs=EPOCHS, save_dir=RESULTS_DIR):
    torch.cuda.empty_cache()
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.7, weight_decay=WEIGHT_DECAY)
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

            preds = model(imgs)
            if preds.shape[2:] != masks.shape[2:]:
                preds = F.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)

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

                if preds.shape[2:] != masks.shape[2:]:
                    preds = F.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)

                val_running += criterion(preds, masks).item() * imgs.size(0)
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                bs = preds_np.shape[0]
                total_samples += bs
                for b in range(bs):
                    p = preds_np[b,0]; g = masks_np[b,0]
                    p_bin = (p > THRESH).astype(np.uint8)
                    g_bin = (g > 0.5).astype(np.uint8)
                    m = compute_metrics_np(p_bin, g_bin)
                    for k in metric_sums:
                        metric_sums[k] += m[k]

        val_loss = val_running / max(len(val_loader.dataset), 1)
        avg_metrics = {k: (metric_sums[k] / (total_samples + 1e-12)) for k in metric_sums}

        history.append({
            "fold": fold_index, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_accuracy, "val_accuracy": avg_metrics["accuracy"],
            "val_precision": avg_metrics["precision"], "val_recall": avg_metrics["recall"],
            "val_f1": avg_metrics["f1"], "val_iou": avg_metrics["iou"], "val_dice": avg_metrics["dice"],
        })

        print(f"Fold {fold_index} Epoch {epoch}: train_acc={train_accuracy:.4f}, val_acc={avg_metrics['accuracy']:.4f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f},val_dice={avg_metrics['dice']:.4f}")

        # save best model (handle DataParallel)
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
            print(f"Early stopping fold {fold_index} after {epoch} epochs.")
            break

    # load best model
    model_path = os.path.join(save_dir, f"best_unet_fold_{fold_index}.pth")
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(sd)
        else:
            model.load_state_dict(sd)

    return model, history

# -------------------------
# Plot helpers
# -------------------------
def plot_two_subplots(history_df, out_path):
    epochs = history_df["epoch"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history_df["train_accuracy"], label="train_accuracy")
    ax1.plot(epochs, history_df["val_accuracy"], label="val_accuracy")
    ax1.set_title("Training vs Validation Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.grid(True); ax1.legend()
    ax2.plot(epochs, history_df["train_loss"], label="train_loss")
    ax2.plot(epochs, history_df["val_loss"], label="val_loss")
    ax2.set_title("Training vs Validation Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.grid(True); ax2.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_loss_and_accuracy(history_df, out_path):
    x = history_df["epoch"].tolist()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, history_df["train_loss"], label="train_loss"); ax.plot(x, history_df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax2 = ax.twinx(); ax2.plot(x, history_df["val_accuracy"], label="val_accuracy", linestyle="--"); ax2.set_ylabel("Accuracy")
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right"); ax.grid(True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    
def plot_iou_per_fold(history_df, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(history_df["epoch"], history_df["val_iou"], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------
# Main K-Fold orchestration
# -------------------------
def main():
    pairs = collect_labeled_pairs()
    assert len(pairs) > 0, "No labeled pairs found. Check folders."
    print(f"Total labeled pairs: {len(pairs)}")
    random.seed(SEED); random.shuffle(pairs)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    config_df = pd.DataFrame.from_dict(RUN_CONFIG, orient="index", columns=["value"])
    config_df.to_csv(os.path.join(RESULTS_DIR, "run_config.csv"))


    all_history = []
    fold_idx = 0
    for train_idx, val_idx in kf.split(pairs):
        fold_idx += 1
        print(f"\n=== Starting fold {fold_idx}/{KFOLDS} ===")
        train_pairs = [pairs[i] for i in train_idx]; val_pairs = [pairs[i] for i in val_idx]

        train_ds = LiverMaskDataset(train_pairs, transform=train_transform, save_aug=SAVE_AUG_SAMPLES, aug_out=AUG_DEBUG_OUT)
        val_ds = LiverMaskDataset(val_pairs, transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = AttentionUNet(in_channels=1, out_channels=1)

        available_gpus = torch.cuda.device_count()
        print("Visible GPUs:", torch.cuda.device_count())

        if available_gpus >= NUM_GPUS_TO_USE and NUM_GPUS_TO_USE > 1:
            gpu_ids = list(range(NUM_GPUS_TO_USE))
            print(f"Using GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            print("Using a single GPU or CPU.")
        model, history = train_and_evaluate_fold(model, train_loader, val_loader, fold_index=fold_idx, epochs=EPOCHS, save_dir=RESULTS_DIR)

        df_fold = pd.DataFrame(history)
        csv_fold = os.path.join(RESULTS_DIR, f"metrics_fold_{fold_idx}.csv")
        
        best_row = df_fold.loc[df_fold["val_loss"].idxmin()]
        mean_row = df_fold.mean(numeric_only=True)
        median_row = df_fold.median(numeric_only=True)
        summary_df = pd.DataFrame([best_row,mean_row,median_row])
        summary_df.index = ["best_epoch", "mean", "median"]

        csv_fold = os.path.join(RESULTS_DIR, f"metrics_summary_fold_{fold_idx}.csv")
        summary_df.to_csv(csv_fold)
        
        print(f"Saved fold {fold_idx} metrics to {csv_fold}")
        plot_two_subplots(df_fold, os.path.join(RESULTS_DIR, f"plots_fold_{fold_idx}.png"))
        plot_iou_per_fold(df_fold,os.path.join(RESULTS_DIR, f"iou_fold_{fold_idx}.png"))

    
    print("K-Fold training complete. Best models saved per fold in:", RESULTS_DIR)

if __name__ == "__main__":
    main()

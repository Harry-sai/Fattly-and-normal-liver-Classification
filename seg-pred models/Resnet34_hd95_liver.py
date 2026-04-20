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
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from skimage.measure import label
from monai.losses import DiceFocalLoss



# -------------------------
# Main settings
# -------------------------
IMAGES_ROOT      = "images"
LABEL_MASKS_ROOT = "predictions_fixed"
RESULTS_DIR      = "resnet34/hd95_more_imgs"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS =80
WEIGHT_DECAY = 1e-4
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
THRESH = 0.6 
NUM_GPUS_TO_USE = 2
KFOLDS = 7
SEED = 42
# decoder dropout
DROPOUT_P = 0.1
MAX_DICE_WEIGHT=0.15
FREEZE_TILL=None


# save a few augmented samples if needed
SAVE_AUG_SAMPLES = True
AUG_DEBUG_OUT = Path(RESULTS_DIR) / "aug_debug"
if SAVE_AUG_SAMPLES:
    AUG_DEBUG_OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Image and mask transforms
# -------------------------
train_transform = A.Compose([
    A.Rotate(limit=15, border_mode=0, p=0.5),     
    A.Affine(
    translate_percent=0.0625,
    scale=(0.9, 1.1),
    rotate=0,
    border_mode=0,
    p=0.4
),
    A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.7),     
    A.RandomGamma(gamma_limit=(80, 120),p=0.5 ),     
    A.GaussNoise(
    std_range=(0.02, 0.1),
    p=0.4
),

    A.Resize(IMG_SIZE, IMG_SIZE),
    ], additional_targets={"mask": "mask"})

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE)
], additional_targets={"mask": "mask"}) 

# boundary metrics


hd95_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95,
    reduction="mean"
)

asd_metric = SurfaceDistanceMetric(
    include_background=False,
    reduction="mean"
)

# save run settings
RUN_CONFIG = {
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "THRESH": THRESH,
    "DROPOUT_P": DROPOUT_P,
    "MAX_DICE_WEIGHT" : MAX_DICE_WEIGHT, 
    "FREEZE_TILL":FREEZE_TILL,
    "KFOLDS": KFOLDS,
    "AUGMENTATIONS": str(train_transform),
    "ENCODER": "ResNet34",
    "OPTIMIZER": "sgd",
    "LOSS": "BCE+ scheduled dice ",
    "COMMENT":"""added largest area cal , reduced dice effect to .2 and removed horizontal flip , epoch 80 , threshold increase to .6 to check , with 7 folds 
            """
}

# -------------------------
# Helper functions
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

# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = nn.BCELoss()

#     def forward(self, pred, target):
#         return self.bce(pred, target) + 0.3*dice_loss(pred, target)
class BCEDiceLoss(nn.Module):
    def __init__(self, max_dice_weight=MAX_DICE_WEIGHT):
        super().__init__()
        self.bce = nn.BCELoss()
        self.max_dice_weight = max_dice_weight

    def forward(self, pred, target, dice_weight):
        return self.bce(pred, target) + dice_weight * dice_loss(pred, target)

def dice_weight_schedule(epoch, total_epochs, max_weight=MAX_DICE_WEIGHT):
    # linearly increase Dice importance
    return max_weight * min(epoch / (0.4 * total_epochs), 1.0)


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
# # -------------------------
def collect_labeled_pairs(images_root=IMAGES_ROOT, masks_root=LABEL_MASKS_ROOT):
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    pairs = []
    missing = []

    # Build lookup: { (class, stem_lower) : mask_path }
    mask_lookup = {}

    for mask_path in masks_root.rglob("*"):
        if not mask_path.is_file():
            continue
        class_name = mask_path.parent.name
        stem = mask_path.stem.lower()   # <-- ignore extension + case
        mask_lookup[(class_name, stem)] = mask_path

    print(f"Unique masks indexed: {len(mask_lookup)}")

    for img_path in images_root.rglob("*"):
        if not img_path.is_file():
            continue

        class_name = img_path.parent.name
        stem = img_path.stem.lower()    # <-- ignore extension + case

        key = (class_name, stem)

        if key in mask_lookup:
            pairs.append((str(img_path), str(mask_lookup[key])))
        else:
            missing.append((str(img_path), class_name))

    print(f"Total pairs found: {len(pairs)}")
    print(f"Images without mask: {len(missing)}")

    return pairs



# -------------------------
# UNet with ResNet34 encoder + dropout in decoder
# -------------------------
class UnetResNet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True, dropout_p=DROPOUT_P):
        super().__init__()
        try:
            ResNetWeights = models.ResNet34_Weights  # type: ignore
            resnet = models.resnet34(weights=ResNetWeights.DEFAULT if pretrained else None)
        except Exception:
            resnet = models.resnet34(pretrained=pretrained)

        # adapt first conv to accept in_channels
        if in_channels != 3:
            w = resnet.conv1.weight.data
            w_mean = w.mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_channels == 1:
                new_conv.weight.data = w_mean
            else:
                new_conv.weight.data = w_mean.repeat(1, in_channels, 1, 1)
            resnet.conv1 = new_conv

        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        def conv_block(in_ch, out_ch, dropout_p=DROPOUT_P):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout_p and dropout_p > 0:
                layers.insert(-1, nn.Dropout2d(dropout_p))  # spatial dropout before final activation
            return nn.Sequential(*layers)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) #nn.ConvTranspose2d for upsampling
        self.dec4 = conv_block(256 + 256, 256, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(128 + 128, 128, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(64 + 64, 64, dropout_p=dropout_p)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(64 + 64, 32, dropout_p=dropout_p)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)         # B,64,H/2,W/2
        x1 = self.maxpool(x0)    # B,64,H/4,W/4
        e1 = self.encoder1(x1)   # B,64
        e2 = self.encoder2(e1)   # B,128
        e3 = self.encoder3(e2)   # B,256
        e4 = self.encoder4(e3)   # B,512

        u4 = self.up4(e4)
        e3_c = center_crop_to(e3, u4.shape[2], u4.shape[3]) if u4.shape[2:] != e3.shape[2:] else e3
        d4 = self.dec4(torch.cat([u4, e3_c], dim=1))

        u3 = self.up3(d4)
        e2_c = center_crop_to(e2, u3.shape[2], u3.shape[3]) if u3.shape[2:] != e2.shape[2:] else e2
        d3 = self.dec3(torch.cat([u3, e2_c], dim=1))

        u2 = self.up2(d3)
        e1_c = center_crop_to(e1, u2.shape[2], u2.shape[3]) if u2.shape[2:] != e1.shape[2:] else e1
        d2 = self.dec2(torch.cat([u2, e1_c], dim=1))

        u1 = self.up1(d2)
        x0_c = center_crop_to(x0, u1.shape[2], u1.shape[3]) if u1.shape[2:] != x0.shape[2:] else x0
        d1 = self.dec1(torch.cat([u1, x0_c], dim=1))

        out = torch.sigmoid(self.final(d1))
        return out
        
def freeze_resnet_encoder_layers(model, freeze_until="encoder2"):
    """
    freeze_until options:
    - "conv1"
    - "encoder1"
    - "encoder2"
    - "encoder3"
    - "encoder4"
    - None (freeze nothing)
    """

    freeze_map = {
        "conv1": ["inc"],
        "encoder1": ["inc", "encoder1"],
        "encoder2": ["inc", "encoder1", "encoder2"],
        "encoder3": ["inc", "encoder1", "encoder2", "encoder3"],
        "encoder4": ["inc", "encoder1", "encoder2", "encoder3", "encoder4"],
    }

    if freeze_until is None:
        return

    layers_to_freeze = freeze_map[freeze_until]

    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layers_to_freeze):
            param.requires_grad = False

def largest_connected_component(mask: np.ndarray):
    """
    mask: binary HxW or 1xHxW
    """
    if mask.ndim == 3:
        mask = mask[0]
    lab = label(mask)
    if lab.max() == 0:
        return mask
    largest = 1 + np.argmax(np.bincount(lab.flat)[1:])
    return (lab == largest).astype(np.uint8)
# -------------------------
# Training per-fold
# -------------------------
def train_and_evaluate_fold(model, train_loader, val_loader, fold_index, epochs=EPOCHS, save_dir=RESULTS_DIR):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="max",        # because IoU
    factor=0.3,
    patience=3
    )
    criterion = BCEDiceLoss(max_dice_weight=MAX_DICE_WEIGHT)

    best_val_iou = 0.0
    epochs_no_improve = 0

    history = []
    for epoch in range(1, epochs+1):
        model.train()
        train_running = 0.0
        train_acc_sum = 0.0
        n_train_samples = 0

        for imgs, masks in tqdm(train_loader, desc=f"Fold {fold_index} Train Epoch {epoch}/{epochs}"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
        
            preds = model(imgs)
        
            if preds.shape[2:] != masks.shape[2:]:
                preds = F.interpolate(
                    preds,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
        
            # ----- LOSS (uses RAW probabilities) -----
            dice_w = dice_weight_schedule(epoch, epochs, max_weight=MAX_DICE_WEIGHT)
            loss = criterion(preds, masks, dice_w)

            opt.zero_grad()
            loss.backward()
            opt.step()
        
            train_running += loss.item() * imgs.size(0)
        
            # ----- TRAIN ACCURACY (metric only) -----
            with torch.no_grad():
                pred_bin = (preds > THRESH).float()
                correct = (pred_bin == masks).float().mean().item()
                train_acc_sum += correct * imgs.size(0)
                n_train_samples += imgs.size(0)
        
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

                val_running += criterion(preds, masks, dice_weight=MAX_DICE_WEIGHT).item() * imgs.size(0)
                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()

                pred_bin = (preds > THRESH).float()
                gt_bin = (masks > 0.5).float()
                
                # skip batches with empty prediction or GT
                if pred_bin.sum() > 0 and gt_bin.sum() > 0:
                    hd95_metric(pred_bin, gt_bin)
                    asd_metric(pred_bin, gt_bin)
                

                bs = preds_np.shape[0]
                total_samples += bs
                for b in range(bs):
                    p = preds_np[b,0]; g = masks_np[b,0]
                    p_bin = (p > THRESH).astype(np.uint8)
                    p_bin = largest_connected_component(p_bin)
                    g_bin = (g > 0.5).astype(np.uint8)
                    m = compute_metrics_np(p_bin, g_bin)
                    for k in metric_sums:
                        metric_sums[k] += m[k]
                        
        # ---- HD95 / ASD safe aggregation ----
        if hd95_metric.get_buffer() is not None and len(hd95_metric.get_buffer()) > 0:
            avg_hd95 = hd95_metric.aggregate().item()
        else:
            avg_hd95 = float("nan")
        
        if asd_metric.get_buffer() is not None and len(asd_metric.get_buffer()) > 0:
            avg_asd = asd_metric.aggregate().item()
        else:
            avg_asd = float("nan")
        
        hd95_metric.reset()
        asd_metric.reset()



        val_loss = val_running / max(len(val_loader.dataset), 1)
        avg_metrics = {k: (metric_sums[k] / (total_samples + 1e-12)) for k in metric_sums}
        current_iou = avg_metrics["iou"]
        scheduler.step(current_iou)

        history.append({
            "fold": fold_index, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_accuracy, "val_accuracy": avg_metrics["accuracy"],
            "val_precision": avg_metrics["precision"], "val_recall": avg_metrics["recall"],
            "val_f1": avg_metrics["f1"], "val_iou": avg_metrics["iou"], "val_dice": avg_metrics["dice"],
            "val_hd95": avg_hd95,
            "val_asd": avg_asd
        })

        print(f"Fold {fold_index} Epoch {epoch}: val_iou={avg_metrics['iou']:.4f},val_hd95={avg_hd95:.4f},val_asd={avg_asd:.4f},train_acc={train_accuracy:.4f},train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # save best model (handle DataParallel)
        model_path = os.path.join(save_dir, f"best_unet_fold_{fold_index}.pth")
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        if current_iou > best_val_iou:
            best_val_iou = current_iou
            torch.save(state_dict_to_save, model_path)
            epochs_no_improve = 0
            print(f"  -> Fold {fold_index} best model saved (val_iou improved to {best_val_iou:.4f})")
        else:
            epochs_no_improve += 1


        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping fold {fold_index} (IoU stopped improving).")
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
    ax1.plot(epochs, history_df["val_iou"], label="val_iou")
    ax1.plot(epochs, history_df["val_f1"], label="val_f1")
    ax1.set_title("Iou vs F1"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("IoU"); ax1.grid(True); ax1.legend()
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
    
def plot_core_metrics(df, out_path_prefix):
    x = df["epoch"]

    # Boundary metrics
    plt.figure(figsize=(8,4))
    plt.plot(x, df["val_hd95"], label="HD95")
    plt.plot(x, df["val_asd"], label="ASD")
    plt.xlabel("Epoch")
    plt.ylabel("Pixels")
    plt.title("Boundary Metrics (pixel units)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_boundary.png")
    plt.close()



# -------------------------
# Main K-Fold orchestration
# -------------------------
def main():
    pairs = collect_labeled_pairs()
    mask_paths = [m for _, m in pairs]
    print("Unique masks:", len(set(mask_paths)))
    print("Total pairs:", len(mask_paths))
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

        model = UnetResNet34(in_channels=1, out_channels=1, pretrained=True, dropout_p=DROPOUT_P)

        #Freezing encoder layers of Resnet 
        freeze_resnet_encoder_layers(model, freeze_until=FREEZE_TILL)

        
        # # Freeze all BatchNorm layers (important for small batch size)
        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()                 # freeze running mean & var
        #         for param in m.parameters():
        #             param.requires_grad = False


        available_gpus = torch.cuda.device_count()
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
        plot_core_metrics(df_fold,os.path.join(RESULTS_DIR, f"core_matrics_{fold_idx}.png"))

    
    print("K-Fold training complete. Best models saved per fold in:", RESULTS_DIR)

    print(sum(p.requires_grad for p in model.parameters()))

if __name__ == "__main__":
    main()

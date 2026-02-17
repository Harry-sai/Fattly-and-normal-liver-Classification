# unet_train_and_predict.py
# Requirements: torch, torchvision, pillow, numpy, tqdm
# Expects mirrored class subfolders as described above.

import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -------------------------
# Configuration (edit paths if needed)
# -------------------------
IMAGES_ROOT        = "data\images"           # contains class subfolders: fatty/, normal/
LABEL_MASKS_ROOT   = "data\labelled"   # same class subfolders with GT masks (same filenames)
PRED_OUT_DIR       = "data\predicted_masks"  # will be created with same class subfolders
MODEL_PATH         = "unet_liver_mask.pth"

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------

# -------------------------
# Simple U-Net (same as before)
# -------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # build encoder (downs)
        ch = in_channels
        for f in features:
            self.downs.append(conv_block(ch, f))  # conv_block(in_ch=ch, out_ch=f)
            ch = f

        # bottleneck: input channels = last feature size
        self.bottleneck = conv_block(ch, ch * 2)
        ch = ch * 2  # channels leaving bottleneck

        # build decoder (ups) -- transpose conv then conv_block(2*f, f)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))  # upsample to f channels
            self.ups.append(conv_block(f * 2, f))  # after concat skip (f) + up (f) => 2*f in
            ch = f

        self.pool = nn.MaxPool2d(2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        up_idx = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # transpose conv -> has f channels
            skip = skips[up_idx]
            up_idx += 1
            # match spatial size if needed
            if x.shape[2:] != skip.shape[2:]:
                _, _, H, W = x.shape
                skip = T.CenterCrop((H,W))(skip)
            x = torch.cat([skip, x], dim=1)  # channels = 2*f
            x = self.ups[i+1](x)             # conv_block(2*f, f)

        return torch.sigmoid(self.final(x))


# -------------------------
# Dataset that collects labeled image-mask pairs from mirrored subfolders
# -------------------------
class LiverMaskDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE):
        """
        pairs: list of (img_path, mask_path) tuples
        """
        self.pairs = pairs
        self.img_size = img_size
        self.tf_img = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        self.tf_mask = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        img = self.tf_img(img)
        mask = self.tf_mask(mask)
        mask = (mask > 0.5).float()
        return img, mask

# -------------------------
# Loss
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

# -------------------------
# Training loop
# -------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, save_path=MODEL_PATH):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    PATIENCE = 10  # optional early stopping

    for epoch in range(1, epochs+1):
        # ---- train ----
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                preds = model(imgs)
                val_running += criterion(preds, masks).item() * imgs.size(0)
        val_loss = val_running / len(val_loader.dataset)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # ---- save best ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # optional early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping after {epoch} epochs. No improvement in {PATIENCE} epochs.")
            break

    # at end, best model is saved to save_path
    # optionally load best weights into model before returning
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model


# -------------------------
# Prediction: preserve class subfolders when saving masks
# -------------------------
def predict_and_save(model, images_root=IMAGES_ROOT, out_root=PRED_OUT_DIR, img_size=IMG_SIZE, threshold=0.5):
    model = model.to(DEVICE)
    model.eval()
    tf = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
    # walk class subfolders
    for class_dir in sorted(Path(images_root).iterdir()):
        if not class_dir.is_dir():
            continue
        rel_class = class_dir.name
        out_class_dir = Path(out_root) / rel_class
        out_class_dir.mkdir(parents=True, exist_ok=True)
        img_paths = sorted(glob(str(class_dir / "*")))
        with torch.no_grad():
            for p in tqdm(img_paths, desc=f"Predicting [{rel_class}]"):
                im = Image.open(p).convert("L")
                inp = tf(im).unsqueeze(0).to(DEVICE)
                pred = model(inp)[0,0].cpu().numpy()
                orig = Image.open(p)
                orig_w, orig_h = orig.size
                pred_img = Image.fromarray((pred*255).astype(np.uint8)).resize((orig_w, orig_h))
                pred_bin = (np.array(pred_img) > int(threshold*255)).astype(np.uint8) * 255
                out_path = out_class_dir / Path(p).name
                Image.fromarray(pred_bin).save(out_path)

# -------------------------
# Helper to collect labelled pairs from mirrored folders
# -------------------------
def collect_labeled_pairs(images_root=IMAGES_ROOT, masks_root=LABEL_MASKS_ROOT):
    pairs = []
    # iterate classes present in images_root
    for class_dir in sorted(Path(images_root).iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        mask_class_dir = Path(masks_root) / class_name
        if not mask_class_dir.exists():
            continue
        for img_path in sorted(glob(str(class_dir / "*"))):
            fname = Path(img_path).name
            mask_path = mask_class_dir / fname
            if mask_path.exists():
                pairs.append((img_path, str(mask_path)))
            else:
                print(f"Warning: no mask for {img_path} in {mask_class_dir}")
    return pairs

# -------------------------
# Main
# -------------------------
def main():
    pairs = collect_labeled_pairs()
    assert len(pairs) > 0, "No labeled image-mask pairs found. Check folders."
    # simple split
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    train_ds = LiverMaskDataset(train_pairs)
    val_ds = LiverMaskDataset(val_pairs) if val_pairs else None
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) if val_ds else None

    model = UNet(in_channels=1, out_channels=1)
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # Predict for all images in images_root and save mirrored outputs
    # predict_and_save(model, images_root=IMAGES_ROOT, out_root=PRED_OUT_DIR)
    # print("Saved predicted masks to", PRED_OUT_DIR)

if __name__ == "__main__":
    main()

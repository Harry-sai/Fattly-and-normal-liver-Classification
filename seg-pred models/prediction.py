import os
from pathlib import Path
import numpy as np
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.measure import label, regionprops

# -------------------------
# CONFIG (MATCH TRAINING)
# -------------------------
MODEL_PATH = "resnet34/hd95_more_imgs/best_unet_fold_4.pth"
IMAGE_DIR  = "false_img/normal_img"
OUT_DIR    = "false_img/predictions_normal"

IMG_SIZE = 384
THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Utilities (copied)
# -------------------------
def center_crop_to(tensor, target_h, target_w):
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
    return tensor

def clean_mask(mask, min_area_ratio=0.01):
    """
    - Keeps only largest connected component
    - Removes small blobs
    - Does NOT alter true liver boundaries
    """

    labeled = label(mask)

    if labeled.max() == 0:
        return mask

    regions = regionprops(labeled)

    # Keep largest component
    largest = max(regions, key=lambda x: x.area)

    clean = np.zeros_like(mask)

    # dynamic threshold (1% of image area default)
    min_area = mask.shape[0] * mask.shape[1] * min_area_ratio

    for region in regions:
        if region.area >= min_area and region.label == largest.label:
            clean[labeled == region.label] = 1

    return clean


# -------------------------
# Model (COPIED EXACTLY)
# -------------------------
class UnetResNet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False, dropout_p=0.1):
        super().__init__()
        resnet = models.resnet34(pretrained=None)

        if in_channels != 3:
            w = resnet.conv1.weight.data
            w_mean = w.mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            new_conv.weight.data = w_mean
            resnet.conv1 = new_conv

        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = conv_block(128, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.maxpool(x0)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        u4 = self.up4(e4)
        e3 = center_crop_to(e3, u4.shape[2], u4.shape[3])
        d4 = self.dec4(torch.cat([u4, e3], dim=1))

        u3 = self.up3(d4)
        e2 = center_crop_to(e2, u3.shape[2], u3.shape[3])
        d3 = self.dec3(torch.cat([u3, e2], dim=1))

        u2 = self.up2(d3)
        e1 = center_crop_to(e1, u2.shape[2], u2.shape[3])
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        u1 = self.up1(d2)
        x0 = center_crop_to(x0, u1.shape[2], u1.shape[3])
        d1 = self.dec1(torch.cat([u1, x0], dim=1))

        return torch.sigmoid(self.final(d1))

# -------------------------
# Load model
# -------------------------
model = UnetResNet34(in_channels=1, out_channels=1)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# -------------------------
# Prediction loop
# -------------------------
with torch.no_grad():
    for img_path in Path(IMAGE_DIR).rglob("*"):

        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        if not img_path.is_file():
            continue

        # Preserve subfolder (fatty_liver / normal)
        rel_path = img_path.relative_to(IMAGE_DIR)
        out_path = Path(OUT_DIR) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- Load & preprocess image ----
        img = np.array(Image.open(img_path).convert("L"))

        img = np.array(
            Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # ---- Inference ----
        pred = model(img_tensor)[0, 0].cpu().numpy()
        binary_mask = (pred > THRESH).astype(np.uint8)

        # Clean blobs
        binary_mask = clean_mask(binary_mask)

        mask = binary_mask * 255

        # ---- Save mask ----
        Image.fromarray(mask).save(out_path)
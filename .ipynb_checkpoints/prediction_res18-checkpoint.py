import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models

# -------------------------
# CONFIG (MATCH TRAINING)
# -------------------------
MODEL_PATH = "resnet18/final/best_unet_fold_3.pth"
IMAGE_DIR  = "data/images"
OUT_DIR    = "data/predictions_res18"

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

# -------------------------
# UNet with ResNet18 encoder + dropout in decoder
# -------------------------
class UnetResNet18(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True, dropout_p=0.1):
        super().__init__()
        try:
            ResNetWeights = models.ResNet18_Weights  # type: ignore
            resnet = models.resnet18(weights=ResNetWeights.DEFAULT if pretrained else None)
        except Exception:
            resnet = models.resnet18(pretrained=pretrained)

        # adapt first conv to accept grayscale
        if in_channels != 3:
            w = resnet.conv1.weight.data
            w_mean = w.mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv.weight.data = w_mean
            resnet.conv1 = new_conv

        # -------------------------
        # Encoder
        # -------------------------
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # -------------------------
        # Decoder blocks
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

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.maxpool(x0)

        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        u4 = self.up4(e4)
        d4 = self.dec4(torch.cat([u4, center_crop_to(e3, u4.shape[2], u4.shape[3])], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, center_crop_to(e2, u3.shape[2], u3.shape[3])], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, center_crop_to(e1, u2.shape[2], u2.shape[3])], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, center_crop_to(x0, u1.shape[2], u1.shape[3])], dim=1))

        return torch.sigmoid(self.final(d1))

# -------------------------
# Load model
# -------------------------
model = UnetResNet18(in_channels=1, out_channels=1)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# -------------------------
# Prediction loop
# -------------------------
with torch.no_grad():
    for img_path in Path(IMAGE_DIR).rglob("*"):
        if not img_path.is_file():
            continue

        # preserve subfolder (fatty / normal)
        rel_path = img_path.relative_to(IMAGE_DIR)
        out_path = Path(OUT_DIR) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = np.array(Image.open(img_path).convert("L"))
        img = np.array(
            Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

        pred = model(img_tensor)[0, 0].cpu().numpy()
        mask = (pred > THRESH).astype(np.uint8) * 255

        Image.fromarray(mask).save(out_path)

        print(f"Saved: {out_path}")


# =========================================================
# LATENT DIFFUSION TRAINING (MONAI)
# python -m torch.distributed.run --nproc_per_node=2 GANfile.py
# =========================================================

import os
import glob
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from PIL import Image

from monai.data import Dataset
from monai.transforms import (
    Compose,LoadImaged,EnsureChannelFirstd,
    ScaleIntensityd,Resized,ToTensord
)
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# -------------------------
# REPRODUCIBILITY
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------
# RUN SETUP
# -------------------------
DATA_ROOT = "data/images"
MASKS_ROOT = "data/masks"          # path where binary liver masks are stored
NORMAL_DIR = os.path.join(DATA_ROOT, "normal")
FATTY_DIR = os.path.join(DATA_ROOT, "fatty_liver")

RUN_NAME = "liver_gan_5th"
BASE_DIR = os.path.join("GAN", RUN_NAME)
DIRS = {
    "samples": "samples",
    "models": "models"
}
for d in DIRS.values():
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# -------------------------
# DDP INIT
# -------------------------
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = f"cuda:{local_rank}"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------
# HYPERPARAMETERS
# -------------------------
IMG_SIZE = 256
BATCH_SIZE = 2
LATENT_CHANNELS = 8
LATENT_SCALE = None  # computed after AE training from data
AE_LR = 1e-4
DIFF_LR = 2e-4  # higher LR for diffusion (small batch size)
KL_WEIGHT = 1e-6
EPOCHS_AE = 50
EPOCHS_DIFF = 150  # more epochs for diffusion convergence
NUM_CLASSES = 2
VAL_SPLIT = 0.1

# -------------------------
# SAVE HYPERPARAMETERS
# -------------------------
with open(os.path.join(BASE_DIR, "hyperparameters.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    for k, v in sorted(locals().items()):
        if k.isupper():
            writer.writerow([k, v])

# helper for pairing images with their corresponding masks
def collect_labeled_pairs(images_root, masks_root):
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    pairs = []
    mask_lookup = {}
    for m in masks_root.rglob("*.PNG"):
        cls = m.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            mask_lookup[(cls, m.stem.lower())] = m

    for img in images_root.rglob("*.PNG"):
        cls = img.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            key = (cls, img.stem.lower())
            if key in mask_lookup:
                label = 0 if cls == "normal" else 1
                pairs.append((str(img), str(mask_lookup[key]), label))

    print("Total samples:", len(pairs))
    return pairs

# simple dataset that loads image+mask, crops to mask bounding box (ROI), resizes
def make_roi(img, mask, pad=4):
    # both numpy arrays, single channel
    if mask.shape != img.shape:
        mask = np.array(
            Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        )
    mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty mask encountered")
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # minimal padding to preserve signal
    y1 = max(y1 - pad, 0)
    y2 = min(y2 + pad, img.shape[0])
    x1 = max(x1 - pad, 0)
    x2 = min(x2 + pad, img.shape[1])

    roi = img[y1:y2, x1:x2]
    return roi

class LiverROIDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        roi = make_roi(img, mask)
        # resize to fixed input size
        roi = np.array(Image.fromarray(roi).resize((self.img_size, self.img_size)))
        # scale to [-1, 1] for AutoencoderKL stability
        roi = roi.astype(np.float32) / 255.0
        roi = roi * 2.0 - 1.0
        tensor = torch.from_numpy(roi).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return tensor, label

# -------------------------
# DATA LOADING (SUPERVISED VIA FOLDERS WITH ROI CROPPING)
# -------------------------
# pair up every image with its corresponding liver mask and label
pairs = collect_labeled_pairs(DATA_ROOT, MASKS_ROOT)

# monai Dataset not needed any more; use a lightweight PyTorch Dataset that
# performs ROI cropping on-the-fly so it will run in parallel on both GPUs
full_dataset = LiverROIDataset(pairs, IMG_SIZE)


val_len = int(len(full_dataset) * VAL_SPLIT)
train_len = len(full_dataset) - val_len
train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

train_sampler = DistributedSampler(train_ds)
val_sampler = DistributedSampler(val_ds, shuffle=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=2,
    pin_memory=True
)

# -------------------------
# AUTOENCODER (STAGE 1)
# -------------------------
# Use a lighter, more stable AE for small batches
autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 256),
    latent_channels=LATENT_CHANNELS,
    num_res_blocks=2,
    norm_num_groups=32,
).to(DEVICE)

autoencoder = DDP(autoencoder, device_ids=[local_rank])

ae_opt = torch.optim.Adam(autoencoder.parameters(), lr=AE_LR)
l1 = nn.L1Loss()

ae_train_losses, ae_val_losses = [], []

def ae_epoch(loader, train=True):
    autoencoder.train(train)
    total = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        recon, mu, sigma = autoencoder(x)
        rec = l1(recon, x)
        kl = torch.mean(-0.5 * torch.sum(
            1 + torch.log(sigma**2) - mu**2 - sigma**2, dim=[1,2,3]
        ))
        loss = rec + KL_WEIGHT * kl
        if train:
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()

        loss_detached = loss.detach()
        dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
        loss_detached /= dist.get_world_size()
        total += loss_detached.item()

    return total / len(loader)

for e in range(EPOCHS_AE):
    train_sampler.set_epoch(e)
    train_loss = ae_epoch(train_loader, True)
    val_loss = ae_epoch(val_loader, False)
    
    ae_train_losses.append(train_loss)
    ae_val_losses.append(val_loss)

    if dist.get_rank() == 0:
        print(f"[AE] Epoch {e+1}/{EPOCHS_AE} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

# Freeze AE
autoencoder.eval()
for p in autoencoder.module.parameters():
    p.requires_grad = False

# ===== AE SANITY CHECK (ADD THIS) =====
with torch.no_grad():
    x, y = next(iter(val_loader))
    x = x.to(DEVICE)
    recon, _, _ = autoencoder(x)
    
if dist.get_rank() == 0:
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    # map back to [0, 1] for visualization
    plt.imshow(((x[0,0].cpu() + 1.0) / 2.0).clamp(0, 1), cmap="gray")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(((recon[0,0].cpu() + 1.0) / 2.0).clamp(0, 1), cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(BASE_DIR, "Ae_curve.png"))
    plt.close()

# estimate latent scale to normalize z (like Stable Diffusion), but based on this AE
@torch.no_grad()
def estimate_latent_scale(loader, num_batches=20):
    stds = []
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(DEVICE)
        z_mu, _ = autoencoder.module.encode(x)
        stds.append(z_mu.std().item())
    mean_std = float(np.mean(stds)) if stds else 1.0
    return 1.0 / max(mean_std, 1e-6)

LATENT_SCALE = estimate_latent_scale(train_loader)
if dist.get_rank() == 0:
    print(f"Estimated LATENT_SCALE: {LATENT_SCALE:.5f}")

# -------------------------
# DIFFUSION MODEL (STAGE 2)
# -------------------------
diffusion = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=LATENT_CHANNELS,
    out_channels=LATENT_CHANNELS,
    channels=(128, 256, 512, 512),
    attention_levels=(False, True, True, True),
    num_res_blocks=2,
    with_conditioning=True,
    cross_attention_dim=128,
).to(DEVICE)

diffusion = DDP(diffusion, device_ids=[local_rank])

class_embed = nn.Embedding(NUM_CLASSES, 128).to(DEVICE)

ddpm = DDPMScheduler(num_train_timesteps=1000)
ddim = DDIMScheduler(num_train_timesteps=1000)

diff_opt = torch.optim.Adam(
    list(diffusion.parameters()) + list(class_embed.parameters()),
    lr=DIFF_LR
)

mse = nn.MSELoss()

diff_train_losses, diff_val_losses = [], []
best_val_loss = float("inf")

def diff_epoch(loader, train=True):
    diffusion.train(train)
    total = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            z, _ = autoencoder.module.encode(x)
            z = z * LATENT_SCALE
        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (z.size(0),), device=DEVICE)
        zn = ddpm.add_noise(z, noise, t)
        cond = class_embed(y).unsqueeze(1)
        pred = diffusion(zn, t, cond)
        loss = mse(pred, noise)
        if train:
            diff_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            diff_opt.step()

        loss_detached = loss.detach()
        dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
        loss_detached /= dist.get_world_size()
        total += loss_detached.item()

    return total / len(loader)

# -------------------------
# SAMPLE GENERATION HELPERS
# -------------------------
from torchvision.utils import save_image

@torch.no_grad()
def generate(label, idx, filename=None):
    z = torch.randn(1, LATENT_CHANNELS, IMG_SIZE//16, IMG_SIZE//16).to(DEVICE)
    y = torch.tensor([label], device=DEVICE)
    cond = class_embed(y).unsqueeze(1)
    ddim.set_timesteps(100)
    for t in ddim.timesteps:
        t_batch = torch.full((z.size(0),), t, device=DEVICE, dtype=torch.long)
        eps = diffusion.module(z, t_batch, cond)
        z, _ = ddim.step(eps, t, z)
    z = z / LATENT_SCALE
    z = torch.clamp(z, -10.0, 10.0)
    img = autoencoder.module.decode(z)
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    out_name = filename or f"{'normal' if label==0 else 'fatty'}_{idx}.png"
    save_image(img.cpu(), os.path.join(BASE_DIR, "samples", out_name))

@torch.no_grad()
def save_epoch_samples(epoch_idx):
    # fixed seed so you see the same latents improving over epochs
    torch.manual_seed(SEED)
    generate(0, epoch_idx, filename="latest_normal.png")
    generate(1, epoch_idx, filename="latest_fatty.png")

for e in range(EPOCHS_DIFF):
    train_sampler.set_epoch(e)
    tr = diff_epoch(train_loader, True)
    vl = diff_epoch(val_loader, False)
    diff_train_losses.append(tr)
    diff_val_losses.append(vl)
    if dist.get_rank() == 0:
        print(f"[DIFF] Epoch {e+1}/{EPOCHS_DIFF} | Train: {tr:.4f} | Val: {vl:.4f}")
        save_epoch_samples(e + 1)
    if vl < best_val_loss:
        best_val_loss = vl
        if dist.get_rank() == 0:
            torch.save(
                diffusion.module.state_dict(),
                os.path.join(BASE_DIR, "models", "diffusion.pt")
            )
            torch.save(
                class_embed.state_dict(),
                os.path.join(BASE_DIR, "models", "class_embed.pt")
            )
            torch.save(
                autoencoder.module.state_dict(),
                os.path.join(BASE_DIR, "models", "autoencoder.pt")
            )


# -------------------------
# LOSS CURVES (PAPER-FRIENDLY)
# -------------------------
if dist.get_rank() == 0:
    plt.figure(figsize=(8,5))
    plt.plot(ae_train_losses, label="AE Train")
    plt.plot(ae_val_losses, label="AE Val")
    plt.plot(diff_train_losses, label="Diff Train")
    plt.plot(diff_val_losses, label="Diff Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "loss_curves.png"))
    plt.close()

if dist.get_rank() == 0:
    for i in range(5):
        generate(0, i)
        generate(1, i)

dist.destroy_process_group()

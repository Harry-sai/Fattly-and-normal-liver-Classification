"""
Proper conditional GAN training for liver ROI synthesis.

This script replaces the previous latent-diffusion style pipeline with a true
adversarial setup:
- Generator: mask-conditioned image synthesizer with class conditioning
- Discriminator: PatchGAN critic that judges image realism conditioned on mask
  and class
- Reconstruction regularization: masked L1 + edge loss to preserve anatomy

The folder structure is expected to match the current project:
    data/images/normal/*.PNG
    data/images/fatty_liver/*.PNG
    data/masks/normal/*.PNG
    data/masks/fatty_liver/*.PNG

Example:
    python GANfile_proper_gan.py
    python GANfile_proper_gan.py --img-size 256 --epochs 200 --batch-size 8
"""

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image


SEED = 42
DATA_ROOT = "data/images"
MASKS_ROOT = "data/masks"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="liver_cgan_v2")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--noise-ch", type=int, default=8)
    parser.add_argument("--lambda-l1", type=float, default=25.0)
    parser.add_argument("--lambda-edge", type=float, default=5.0)
    parser.add_argument("--lambda-bg", type=float, default=2.0)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--instance-noise", type=float, default=0.0)
    parser.add_argument("--hflip-aug", action="store_true")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_mean(value, device, world_size):
    if world_size == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


def collect_labeled_pairs(images_root, masks_root):
    images_root = Path(images_root)
    masks_root = Path(masks_root)
    pairs = []
    mask_lookup = {}

    for mask_path in list(masks_root.rglob("*.PNG")) + list(masks_root.rglob("*.png")):
        cls = mask_path.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            mask_lookup[(cls, mask_path.stem.lower())] = mask_path

    for img_path in list(images_root.rglob("*.PNG")) + list(images_root.rglob("*.png")):
        cls = img_path.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            key = (cls, img_path.stem.lower())
            if key in mask_lookup:
                label = 0 if cls == "normal" else 1
                pairs.append((str(img_path), str(mask_lookup[key]), label))

    return sorted(pairs)


def robust_normalize(image):
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi - lo < 1e-6:
        image = np.clip(image / 255.0, 0.0, 1.0)
    else:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32)


def square_crop_from_mask(img, mask, pad_ratio=0.18):
    if mask.shape != img.shape:
        mask = np.array(
            Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        )

    mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        side = min(img.shape[0], img.shape[1])
        y1 = max(0, (img.shape[0] - side) // 2)
        x1 = max(0, (img.shape[1] - side) // 2)
        return (
            img[y1:y1 + side, x1:x1 + side],
            mask[y1:y1 + side, x1:x1 + side],
        )

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

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
    y2 = y1 + side
    x2 = x1 + side

    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


class LiverROIGANDataset(Dataset):
    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        roi, roi_mask = square_crop_from_mask(img, mask)
        roi = robust_normalize(roi)
        roi_mask = roi_mask.astype(np.float32)

        roi = np.array(
            Image.fromarray((roi * 255.0).astype(np.uint8)).resize(
                (self.img_size, self.img_size), resample=Image.BICUBIC
            )
        ).astype(np.float32) / 255.0
        roi_mask = np.array(
            Image.fromarray((roi_mask * 255.0).astype(np.uint8)).resize(
                (self.img_size, self.img_size), resample=Image.NEAREST
            )
        ).astype(np.float32) / 255.0
        roi_mask = (roi_mask > 0.5).astype(np.float32)

        liver_only = roi * roi_mask

        return {
            "image": torch.from_numpy(liver_only * 2.0 - 1.0).unsqueeze(0),
            "mask": torch.from_numpy(roi_mask).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def maybe_augment_batch(images, masks, enabled=False):
    if enabled and random.random() < 0.5:
        images = torch.flip(images, dims=[3])
        masks = torch.flip(masks, dims=[3])
    return images, masks


def conv_block(in_ch, out_ch, stride=2, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)


def disc_conv_block(in_ch, out_ch, stride=2, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=True)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, noise_ch=8, base_ch=64, num_classes=2):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 16)
        in_ch = 1 + noise_ch + 16

        self.e1 = conv_block(in_ch, base_ch, norm=False)
        self.e2 = conv_block(base_ch, base_ch * 2)
        self.e3 = conv_block(base_ch * 2, base_ch * 4)
        self.e4 = conv_block(base_ch * 4, base_ch * 8)
        self.e5 = conv_block(base_ch * 8, base_ch * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=False),
        )

        self.d5 = deconv_block(base_ch * 8, base_ch * 8, dropout=0.3)
        self.d4 = deconv_block(base_ch * 16, base_ch * 4, dropout=0.1)
        self.d3 = deconv_block(base_ch * 8, base_ch * 2)
        self.d2 = deconv_block(base_ch * 4, base_ch)
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, mask, labels, noise=None):
        if noise is None:
            noise = torch.randn(
                mask.size(0),
                self.e1[0].in_channels - 1 - 16,
                mask.size(2),
                mask.size(3),
                device=mask.device,
            )

        class_map = self.class_embed(labels).unsqueeze(-1).unsqueeze(-1)
        class_map = class_map.expand(-1, -1, mask.size(2), mask.size(3))
        x = torch.cat([mask, noise, class_map], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        b = self.bottleneck(e5)

        d5 = self.d5(b)
        d4 = self.d4(torch.cat([d5, e4], dim=1))
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        out = self.d1(torch.cat([d2, e1], dim=1))
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, base_ch=64, num_classes=2):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 16)
        in_ch = 1 + 1 + 16

        self.net = nn.Sequential(
            disc_conv_block(in_ch, base_ch, norm=False),
            disc_conv_block(base_ch, base_ch * 2),
            disc_conv_block(base_ch * 2, base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image, mask, labels):
        class_map = self.class_embed(labels).unsqueeze(-1).unsqueeze(-1)
        class_map = class_map.expand(-1, -1, image.size(2), image.size(3))
        x = torch.cat([image, mask, class_map], dim=1)
        return self.net(x)


def sobel_edges(x):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def discriminator_hinge_loss(real_logits, fake_logits):
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def generator_hinge_loss(fake_logits):
    return -fake_logits.mean()


def reconstruction_loss(fake, real, mask, lambda_l1, lambda_edge, lambda_bg):
    mask = mask.float()
    inside = (fake - real).abs() * mask
    outside = (fake - real).abs() * (1.0 - mask)

    inside_l1 = inside.sum() / mask.sum().clamp_min(1.0)
    outside_l1 = outside.sum() / (1.0 - mask).sum().clamp_min(1.0)
    edge_l1 = (sobel_edges(fake) - sobel_edges(real)).abs()
    edge_l1 = (edge_l1 * mask).sum() / mask.sum().clamp_min(1.0)

    return lambda_l1 * inside_l1 + lambda_edge * edge_l1 + lambda_bg * outside_l1


def add_instance_noise(x, std):
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(-1.0, 1.0)


@torch.no_grad()
def save_preview(generator, preview_batch, fixed_noise, epoch, base_dir, device):
    generator_model = generator.module if isinstance(generator, DDP) else generator
    generator_model.eval()
    images = preview_batch["image"].to(device)
    masks = preview_batch["mask"].to(device)
    labels = preview_batch["label"].to(device)

    fake = generator_model(masks, labels, fixed_noise.to(device))
    fake = fake * masks

    grid = make_grid(
        torch.cat(
            [
                (images + 1.0) * 0.5,
                masks.repeat(1, 3, 1, 1)[:, :1],
                (fake + 1.0) * 0.5,
            ],
            dim=0,
        ),
        nrow=images.size(0),
    )
    save_image(grid, os.path.join(base_dir, "samples", f"epoch_{epoch:04d}.png"))
    save_image((fake + 1.0) * 0.5, os.path.join(base_dir, "samples", "latest_generated_batch.png"))
    generator_model.train()


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed()
    seed_everything(SEED + rank)
    base_dir = os.path.join("GAN", args.run_name)
    if is_main_process(rank):
        ensure_dir(base_dir)
        ensure_dir(os.path.join(base_dir, "samples"))
        ensure_dir(os.path.join(base_dir, "models"))

    pairs = collect_labeled_pairs(DATA_ROOT, MASKS_ROOT)
    if len(pairs) == 0:
        cleanup_distributed(distributed)
        raise RuntimeError("No image/mask pairs found in data/images and data/masks.")

    if args.debug:
        pairs = pairs[: min(len(pairs), 48)]

    dataset = LiverROIGANDataset(pairs, args.img_size)
    val_len = int(len(dataset) * args.val_split)
    if len(dataset) > 1:
        val_len = max(1, val_len)
    val_len = min(val_len, max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len

    split_generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=split_generator)
    preview_source = val_ds if len(val_ds) > 0 else train_ds

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed and len(val_ds) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_ds) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    ) if len(val_ds) > 0 else None

    preview_items = [preview_source[i] for i in range(min(args.preview_count, len(preview_source)))]
    preview_batch = {
        "image": torch.stack([item["image"] for item in preview_items], dim=0),
        "mask": torch.stack([item["mask"] for item in preview_items], dim=0),
        "label": torch.stack([item["label"] for item in preview_items], dim=0),
    }
    fixed_noise = torch.randn(
        preview_batch["image"].size(0),
        args.noise_ch,
        args.img_size,
        args.img_size,
    )

    generator = Generator(noise_ch=args.noise_ch, base_ch=args.base_ch).to(device)
    discriminator = PatchDiscriminator(base_ch=args.base_ch).to(device)

    if distributed:
        generator = DDP(generator, device_ids=[local_rank] if device.type == "cuda" else None)
        discriminator = DDP(discriminator, device_ids=[local_rank] if device.type == "cuda" else None)

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.epochs)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=args.epochs)

    history = {
        "g_total": [],
        "g_adv": [],
        "g_recon": [],
        "d_loss": [],
        "val_l1": [],
    }
    best_val = float("inf")
    metrics_path = os.path.join(base_dir, "metrics_per_epoch.csv")

    if is_main_process(rank):
        with open(os.path.join(base_dir, "hyperparameters.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for key, value in sorted(vars(args).items()):
                writer.writerow([key, value])
            writer.writerow(["world_size", world_size])
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "g_total", "g_adv", "g_recon", "d_loss", "val_l1"])

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()

        g_total_epoch = 0.0
        g_adv_epoch = 0.0
        g_recon_epoch = 0.0
        d_loss_epoch = 0.0
        num_steps = 0

        for batch in train_loader:
            real = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            real, mask = maybe_augment_batch(real, mask, enabled=args.hflip_aug)

            noise = torch.randn(real.size(0), args.noise_ch, args.img_size, args.img_size, device=device)
            fake = generator(mask, label, noise)
            fake = fake * mask
            real = real * mask

            for _ in range(args.d_steps):
                d_opt.zero_grad()
                fake_for_d = fake.detach()
                real_logits = discriminator(
                    add_instance_noise(real, args.instance_noise),
                    mask,
                    label,
                )
                fake_logits = discriminator(
                    add_instance_noise(fake_for_d, args.instance_noise),
                    mask,
                    label,
                )
                d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                d_loss.backward()
                d_opt.step()

            g_opt.zero_grad()
            fake_logits = discriminator(add_instance_noise(fake, args.instance_noise), mask, label)
            g_adv = generator_hinge_loss(fake_logits)
            g_recon = reconstruction_loss(
                fake,
                real,
                mask,
                lambda_l1=args.lambda_l1,
                lambda_edge=args.lambda_edge,
                lambda_bg=args.lambda_bg,
            )
            g_total = g_adv + g_recon
            g_total.backward()
            g_opt.step()

            g_total_epoch += g_total.item()
            g_adv_epoch += g_adv.item()
            g_recon_epoch += g_recon.item()
            d_loss_epoch += d_loss.item()
            num_steps += 1

        g_scheduler.step()
        d_scheduler.step()

        history["g_total"].append(g_total_epoch / max(num_steps, 1))
        history["g_adv"].append(g_adv_epoch / max(num_steps, 1))
        history["g_recon"].append(g_recon_epoch / max(num_steps, 1))
        history["d_loss"].append(d_loss_epoch / max(num_steps, 1))

        val_l1 = 0.0
        val_steps = 0
        if val_loader is not None:
            generator.eval()
            with torch.no_grad():
                for batch in val_loader:
                    real = batch["image"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    label = batch["label"].to(device, non_blocking=True)
                    fake = generator(mask, label)
                    fake = fake * mask
                    real = real * mask
                    val_l1 += (torch.abs(fake - real) * mask).sum().item() / mask.sum().clamp_min(1.0).item()
                    val_steps += 1
            val_l1 = val_l1 / max(val_steps, 1)
        history["g_total"][-1] = reduce_mean(history["g_total"][-1], device, world_size)
        history["g_adv"][-1] = reduce_mean(history["g_adv"][-1], device, world_size)
        history["g_recon"][-1] = reduce_mean(history["g_recon"][-1], device, world_size)
        history["d_loss"][-1] = reduce_mean(history["d_loss"][-1], device, world_size)
        val_l1 = reduce_mean(val_l1, device, world_size)
        history["val_l1"].append(val_l1)

        if is_main_process(rank):
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"G_total={history['g_total'][-1]:.4f} "
                f"G_adv={history['g_adv'][-1]:.4f} "
                f"G_recon={history['g_recon'][-1]:.4f} "
                f"D={history['d_loss'][-1]:.4f} "
                f"ValL1={val_l1:.4f}"
            )
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{history['g_total'][-1]:.6f}",
                    f"{history['g_adv'][-1]:.6f}",
                    f"{history['g_recon'][-1]:.6f}",
                    f"{history['d_loss'][-1]:.6f}",
                    f"{val_l1:.6f}",
                ])

        if is_main_process(rank) and (epoch % args.sample_every == 0 or epoch == 1):
            save_preview(generator, preview_batch, fixed_noise, epoch, base_dir, device)

        metric = val_l1 if val_loader is not None else history["g_recon"][-1]
        if is_main_process(rank) and metric < best_val:
            best_val = metric
            gen_to_save = generator.module if isinstance(generator, DDP) else generator
            disc_to_save = discriminator.module if isinstance(discriminator, DDP) else discriminator
            torch.save(gen_to_save.state_dict(), os.path.join(base_dir, "models", "generator.pt"))
            torch.save(disc_to_save.state_dict(), os.path.join(base_dir, "models", "discriminator.pt"))
            torch.save(gen_to_save.state_dict(), os.path.join(base_dir, "models", f"generator_epoch_{epoch:04d}.pt"))

    if is_main_process(rank):
        plt.figure(figsize=(8, 5))
        plt.plot(history["g_total"], label="G Total")
        plt.plot(history["g_adv"], label="G Adv")
        plt.plot(history["g_recon"], label="G Recon")
        plt.plot(history["d_loss"], label="D Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Training Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "loss_curves.png"))
        plt.close()

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

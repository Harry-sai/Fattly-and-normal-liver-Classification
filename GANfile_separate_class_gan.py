"""
Train two separate liver GANs: one for normal and one for fatty_liver.

Why this script exists:
- removes class-conditioning competition inside one shared model
- keeps a consistent square ROI crop from masks
- evaluates checkpoints with a richer validation score than masked L1 alone

Outputs are saved separately for each class:
    GAN/<run_name>/normal/
    GAN/<run_name>/fatty_liver/

Example:
    torchrun --nproc_per_node=2 GANfile_separate_class_gan.py --run-name liver_sep_gan_v1 --batch-size 4 --epochs 140
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
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image


SEED = 42
DATA_ROOT = "data/images"
MASKS_ROOT = "data/masks"
CLASS_NAMES = ["normal", "fatty_liver"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="liver_separate_gan_v1")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--noise-ch", type=int, default=8)
    parser.add_argument("--lambda-bg", type=float, default=2.0)
    parser.add_argument("--normal-lambda-l1", type=float, default=24.0)
    parser.add_argument("--normal-lambda-edge", type=float, default=5.0)
    parser.add_argument("--normal-lambda-fm", type=float, default=1.5)
    parser.add_argument("--normal-lambda-tv", type=float, default=0.25)
    parser.add_argument("--fatty-lambda-l1", type=float, default=18.0)
    parser.add_argument("--fatty-lambda-edge", type=float, default=6.0)
    parser.add_argument("--fatty-lambda-fm", type=float, default=4.0)
    parser.add_argument("--fatty-lambda-tv", type=float, default=0.08)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--hflip-aug", action="store_true")
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


def ddp_barrier(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def reduce_mean(value, device, world_size):
    if world_size == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


def collect_pairs_for_class(images_root, masks_root, class_name):
    images_root = Path(images_root) / class_name
    masks_root = Path(masks_root) / class_name

    pairs = []
    mask_lookup = {}
    for mask_path in list(masks_root.rglob("*.PNG")) + list(masks_root.rglob("*.png")):
        mask_lookup[mask_path.stem.lower()] = mask_path

    for img_path in list(images_root.rglob("*.PNG")) + list(images_root.rglob("*.png")):
        key = img_path.stem.lower()
        if key in mask_lookup:
            pairs.append((str(img_path), str(mask_lookup[key])))

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
        return img[y1:y1 + side, x1:x1 + side], mask[y1:y1 + side, x1:x1 + side]

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


class LiverSingleClassDataset(Dataset):
    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
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
    layers = [spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=True))]
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
    def __init__(self, noise_ch=8, base_ch=64):
        super().__init__()
        in_ch = 1 + noise_ch
        self.noise_ch = noise_ch

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

    def forward(self, mask, noise=None):
        if noise is None:
            noise = torch.randn(mask.size(0), self.noise_ch, mask.size(2), mask.size(3), device=mask.device)

        x = torch.cat([mask, noise], dim=1)
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
        return self.d1(torch.cat([d2, e1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        in_ch = 1 + 1
        self.block1 = disc_conv_block(in_ch, base_ch, norm=False)
        self.block2 = disc_conv_block(base_ch, base_ch * 2)
        self.block3 = disc_conv_block(base_ch * 2, base_ch * 4)
        self.block4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=1, padding=1, bias=True)),
            nn.InstanceNorm2d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.head = spectral_norm(nn.Conv2d(base_ch * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, image, mask, return_features=False):
        x = torch.cat([image, mask], dim=1)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        logits = self.head(f4)
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


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
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits):
    return -fake_logits.mean()


def reconstruction_loss(fake, real, mask, lambda_l1, lambda_edge, lambda_bg):
    inside = (fake - real).abs() * mask
    outside = (fake - real).abs() * (1.0 - mask)
    inside_l1 = inside.sum() / mask.sum().clamp_min(1.0)
    outside_l1 = outside.sum() / (1.0 - mask).sum().clamp_min(1.0)
    edge_l1 = (sobel_edges(fake) - sobel_edges(real)).abs()
    edge_l1 = (edge_l1 * mask).sum() / mask.sum().clamp_min(1.0)
    return lambda_l1 * inside_l1 + lambda_edge * edge_l1 + lambda_bg * outside_l1


def feature_matching_loss(fake_features, real_features):
    loss = 0.0
    for fake_f, real_f in zip(fake_features, real_features):
        loss = loss + F.l1_loss(fake_f, real_f.detach())
    return loss / max(len(fake_features), 1)


def total_variation_loss(fake, mask):
    dx = (fake[:, :, :, 1:] - fake[:, :, :, :-1]).abs()
    dy = (fake[:, :, 1:, :] - fake[:, :, :-1, :]).abs()
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    tv_x = (dx * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    tv_y = (dy * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return tv_x + tv_y


def get_class_config(args, class_name):
    if class_name == "normal":
        return {
            "lambda_l1": args.normal_lambda_l1,
            "lambda_edge": args.normal_lambda_edge,
            "lambda_fm": args.normal_lambda_fm,
            "lambda_tv": args.normal_lambda_tv,
        }
    return {
        "lambda_l1": args.fatty_lambda_l1,
        "lambda_edge": args.fatty_lambda_edge,
        "lambda_fm": args.fatty_lambda_fm,
        "lambda_tv": args.fatty_lambda_tv,
    }


def masked_ssim(fake_01, real_01, mask):
    mask_sum = mask.sum().clamp_min(1.0)
    mu_x = (fake_01 * mask).sum() / mask_sum
    mu_y = (real_01 * mask).sum() / mask_sum
    sigma_x = (((fake_01 - mu_x) ** 2) * mask).sum() / mask_sum
    sigma_y = (((real_01 - mu_y) ** 2) * mask).sum() / mask_sum
    sigma_xy = ((((fake_01 - mu_x) * (real_01 - mu_y)) * mask).sum()) / mask_sum
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-6)).item()


def laplacian_map(x):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def collect_distribution_statistics(generator, eval_ds, args, device):
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    real_pixels = []
    fake_pixels = []
    real_means, fake_means = [], []
    real_stds, fake_stds = [], []
    real_edges, fake_edges = [], []
    real_lap_vars, fake_lap_vars = [], []

    generator.eval()
    torch.manual_seed(SEED)

    with torch.no_grad():
        for batch in loader:
            real = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            fake = generator(mask) * mask

            real_01 = ((real + 1.0) * 0.5).clamp(0.0, 1.0)
            fake_01 = ((fake + 1.0) * 0.5).clamp(0.0, 1.0)
            real_edge = sobel_edges(real_01)
            fake_edge = sobel_edges(fake_01)
            real_lap = laplacian_map(real_01)
            fake_lap = laplacian_map(fake_01)

            for i in range(real.size(0)):
                mask_i = mask[i, 0] > 0.5
                if mask_i.sum().item() == 0:
                    continue

                rp = real_01[i, 0][mask_i].cpu().numpy()
                fp = fake_01[i, 0][mask_i].cpu().numpy()
                re = real_edge[i, 0][mask_i].cpu().numpy()
                fe = fake_edge[i, 0][mask_i].cpu().numpy()
                rl = real_lap[i, 0][mask_i].cpu().numpy()
                fl = fake_lap[i, 0][mask_i].cpu().numpy()

                real_pixels.append(rp)
                fake_pixels.append(fp)
                real_means.append(float(rp.mean()))
                fake_means.append(float(fp.mean()))
                real_stds.append(float(rp.std()))
                fake_stds.append(float(fp.std()))
                real_edges.append(float(re.mean()))
                fake_edges.append(float(fe.mean()))
                real_lap_vars.append(float(rl.var()))
                fake_lap_vars.append(float(fl.var()))

    real_pixels = np.concatenate(real_pixels) if real_pixels else np.array([0.0], dtype=np.float32)
    fake_pixels = np.concatenate(fake_pixels) if fake_pixels else np.array([0.0], dtype=np.float32)
    hist_bins = np.linspace(0.0, 1.0, 65)
    real_hist, _ = np.histogram(real_pixels, bins=hist_bins, density=True)
    fake_hist, _ = np.histogram(fake_pixels, bins=hist_bins, density=True)
    hist_gap = np.abs(real_hist - fake_hist)

    return {
        "hist_bins": hist_bins,
        "real_hist": real_hist,
        "fake_hist": fake_hist,
        "hist_gap": hist_gap,
        "real_means": np.array(real_means),
        "fake_means": np.array(fake_means),
        "real_stds": np.array(real_stds),
        "fake_stds": np.array(fake_stds),
        "real_edges": np.array(real_edges),
        "fake_edges": np.array(fake_edges),
        "real_lap_vars": np.array(real_lap_vars),
        "fake_lap_vars": np.array(fake_lap_vars),
        "summary": {
            "hist_l1": float(hist_gap.mean()),
            "mean_diff": float(abs(np.mean(real_means) - np.mean(fake_means))) if real_means else 0.0,
            "std_diff": float(abs(np.mean(real_stds) - np.mean(fake_stds))) if real_stds else 0.0,
            "edge_diff": float(abs(np.mean(real_edges) - np.mean(fake_edges))) if real_edges else 0.0,
            "lap_var_diff": float(abs(np.mean(real_lap_vars) - np.mean(fake_lap_vars))) if real_lap_vars else 0.0,
        },
    }


def save_distribution_plots(class_dir, stats):
    diagnostics_dir = os.path.join(class_dir, "diagnostics")
    ensure_dir(diagnostics_dir)

    bin_centers = 0.5 * (stats["hist_bins"][:-1] + stats["hist_bins"][1:])

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, stats["real_hist"], label="Real")
    plt.plot(bin_centers, stats["fake_hist"], label="Generated")
    plt.xlabel("Intensity")
    plt.ylabel("Density")
    plt.title("Masked Intensity Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "intensity_histogram.png"))
    plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.imshow(stats["hist_gap"][None, :], aspect="auto", cmap="magma")
    plt.yticks([])
    plt.xticks(np.linspace(0, len(bin_centers) - 1, 5), [f"{x:.2f}" for x in np.linspace(0, 1, 5)])
    plt.title("Histogram Difference Heatmap")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "histogram_gap_heatmap.png"))
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(stats["real_means"], stats["real_stds"], s=18, alpha=0.65, label="Real")
    plt.scatter(stats["fake_means"], stats["fake_stds"], s=18, alpha=0.65, label="Generated")
    plt.xlabel("Masked Mean Intensity")
    plt.ylabel("Masked Std Intensity")
    plt.title("Per-Image Mean vs Std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "mean_std_scatter.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(stats["real_edges"], bins=24, alpha=0.65, label="Real")
    axes[0].hist(stats["fake_edges"], bins=24, alpha=0.65, label="Generated")
    axes[0].set_title("Edge Energy Distribution")
    axes[0].set_xlabel("Mean Sobel Energy")
    axes[0].legend()

    axes[1].hist(stats["real_lap_vars"], bins=24, alpha=0.65, label="Real")
    axes[1].hist(stats["fake_lap_vars"], bins=24, alpha=0.65, label="Generated")
    axes[1].set_title("Laplacian Variance Distribution")
    axes[1].set_xlabel("Variance")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostics_dir, "texture_statistics.png"))
    plt.close(fig)

    with open(os.path.join(diagnostics_dir, "distribution_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in stats["summary"].items():
            writer.writerow([key, f"{value:.6f}"])


@torch.no_grad()
def save_preview(generator, preview_batch, fixed_noise, epoch, class_dir, device):
    model = generator.module if isinstance(generator, DDP) else generator
    model.eval()
    images = preview_batch["image"].to(device)
    masks = preview_batch["mask"].to(device)
    fake = model(masks, fixed_noise.to(device))
    fake = fake * masks

    grid = make_grid(
        torch.cat([(images + 1.0) * 0.5, masks, (fake + 1.0) * 0.5], dim=0),
        nrow=images.size(0),
    )
    save_image(grid, os.path.join(class_dir, "samples", f"epoch_{epoch:04d}.png"))
    save_image((fake + 1.0) * 0.5, os.path.join(class_dir, "samples", "latest_generated_batch.png"))
    model.train()


def write_hparams(class_dir, args, num_pairs, world_size, class_name):
    with open(os.path.join(class_dir, "hyperparameters.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])
        writer.writerow(["world_size", world_size])
        writer.writerow(["class_name", class_name])
        writer.writerow(["num_pairs", num_pairs])


def plot_history(class_dir, history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["g_total"], label="G Total")
    plt.plot(history["g_adv"], label="G Adv")
    plt.plot(history["g_recon"], label="G Recon")
    plt.plot(history["g_fm"], label="G FeatureMatch")
    plt.plot(history["g_tv"], label="G TV")
    plt.plot(history["d_loss"], label="D Loss")
    plt.plot(history["val_score"], label="Val Score")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Score")
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(class_dir, "loss_curves.png"))
    plt.close()


def train_single_class(class_name, class_idx, args, distributed, rank, world_size, local_rank, device):
    pairs = collect_pairs_for_class(DATA_ROOT, MASKS_ROOT, class_name)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/mask pairs found for class '{class_name}'.")
    class_cfg = get_class_config(args, class_name)

    if args.debug:
        pairs = pairs[: min(len(pairs), 48)]

    dataset = LiverSingleClassDataset(pairs, args.img_size)
    val_len = int(len(dataset) * args.val_split)
    if len(dataset) > 1:
        val_len = max(1, val_len)
    val_len = min(val_len, max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len

    split_generator = torch.Generator().manual_seed(SEED + class_idx)
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
    }
    fixed_noise = torch.randn(preview_batch["image"].size(0), args.noise_ch, args.img_size, args.img_size)

    generator = Generator(noise_ch=args.noise_ch, base_ch=args.base_ch).to(device)
    discriminator = PatchDiscriminator(base_ch=args.base_ch).to(device)
    if distributed:
        generator = DDP(generator, device_ids=[local_rank] if device.type == "cuda" else None)
        discriminator = DDP(discriminator, device_ids=[local_rank] if device.type == "cuda" else None)

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.epochs)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=args.epochs)

    class_dir = os.path.join("GAN", args.run_name, class_name)
    metrics_path = os.path.join(class_dir, "metrics_per_epoch.csv")
    history = {"g_total": [], "g_adv": [], "g_recon": [], "g_fm": [], "g_tv": [], "d_loss": [], "val_l1": [], "val_edge": [], "val_ssim": [], "val_score": []}
    best_score = float("inf")
    best_epoch = -1

    if is_main_process(rank):
        ensure_dir(class_dir)
        ensure_dir(os.path.join(class_dir, "samples"))
        ensure_dir(os.path.join(class_dir, "models"))
        write_hparams(class_dir, args, len(pairs), world_size, class_name)
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "g_total", "g_adv", "g_recon", "g_fm", "g_tv", "d_loss", "val_l1", "val_edge", "val_ssim", "val_score"])

    ddp_barrier(distributed)

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(class_idx * 10000 + epoch)

        generator.train()
        discriminator.train()
        g_total_epoch = 0.0
        g_adv_epoch = 0.0
        g_recon_epoch = 0.0
        g_fm_epoch = 0.0
        g_tv_epoch = 0.0
        d_loss_epoch = 0.0
        num_steps = 0

        for batch in train_loader:
            real = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            real, mask = maybe_augment_batch(real, mask, enabled=args.hflip_aug)
            noise = torch.randn(real.size(0), args.noise_ch, args.img_size, args.img_size, device=device)
            fake = generator(mask, noise) * mask
            real = real * mask

            d_opt.zero_grad()
            real_logits = discriminator(real, mask)
            fake_logits = discriminator(fake.detach(), mask)
            d_loss = discriminator_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            fake_logits, fake_features = discriminator(fake, mask, return_features=True)
            _, real_features = discriminator(real, mask, return_features=True)
            g_adv = generator_hinge_loss(fake_logits)
            g_recon = reconstruction_loss(
                fake,
                real,
                mask,
                class_cfg["lambda_l1"],
                class_cfg["lambda_edge"],
                args.lambda_bg,
            )
            g_fm = feature_matching_loss(fake_features, real_features)
            g_tv = total_variation_loss(fake, mask)
            g_total = g_adv + g_recon + class_cfg["lambda_fm"] * g_fm + class_cfg["lambda_tv"] * g_tv
            g_total.backward()
            g_opt.step()

            g_total_epoch += g_total.item()
            g_adv_epoch += g_adv.item()
            g_recon_epoch += g_recon.item()
            g_fm_epoch += g_fm.item()
            g_tv_epoch += g_tv.item()
            d_loss_epoch += d_loss.item()
            num_steps += 1

        g_scheduler.step()
        d_scheduler.step()

        g_total_mean = reduce_mean(g_total_epoch / max(num_steps, 1), device, world_size)
        g_adv_mean = reduce_mean(g_adv_epoch / max(num_steps, 1), device, world_size)
        g_recon_mean = reduce_mean(g_recon_epoch / max(num_steps, 1), device, world_size)
        g_fm_mean = reduce_mean(g_fm_epoch / max(num_steps, 1), device, world_size)
        g_tv_mean = reduce_mean(g_tv_epoch / max(num_steps, 1), device, world_size)
        d_loss_mean = reduce_mean(d_loss_epoch / max(num_steps, 1), device, world_size)

        val_l1 = 0.0
        val_edge = 0.0
        val_ssim = 0.0
        val_steps = 0
        if val_loader is not None:
            generator.eval()
            with torch.no_grad():
                for batch in val_loader:
                    real = batch["image"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    fake = generator(mask) * mask
                    real = real * mask

                    l1_inside = ((fake - real).abs() * mask).sum() / mask.sum().clamp_min(1.0)
                    edge_inside = ((sobel_edges(fake) - sobel_edges(real)).abs() * mask).sum() / mask.sum().clamp_min(1.0)
                    fake_01 = ((fake + 1.0) * 0.5).clamp(0.0, 1.0)
                    real_01 = ((real + 1.0) * 0.5).clamp(0.0, 1.0)
                    val_l1 += l1_inside.item()
                    val_edge += edge_inside.item()
                    val_ssim += masked_ssim(fake_01, real_01, mask)
                    val_steps += 1

        val_l1 = reduce_mean(val_l1 / max(val_steps, 1), device, world_size)
        val_edge = reduce_mean(val_edge / max(val_steps, 1), device, world_size)
        val_ssim = reduce_mean(val_ssim / max(val_steps, 1), device, world_size)
        val_score = val_l1 + 0.35 * val_edge + (1.0 - val_ssim)

        history["g_total"].append(g_total_mean)
        history["g_adv"].append(g_adv_mean)
        history["g_recon"].append(g_recon_mean)
        history["g_fm"].append(g_fm_mean)
        history["g_tv"].append(g_tv_mean)
        history["d_loss"].append(d_loss_mean)
        history["val_l1"].append(val_l1)
        history["val_edge"].append(val_edge)
        history["val_ssim"].append(val_ssim)
        history["val_score"].append(val_score)

        if is_main_process(rank):
            print(
                f"[{class_name}] Epoch {epoch:03d}/{args.epochs} "
                f"G_total={g_total_mean:.4f} G_adv={g_adv_mean:.4f} G_recon={g_recon_mean:.4f} G_fm={g_fm_mean:.4f} G_tv={g_tv_mean:.4f} "
                f"D={d_loss_mean:.4f} ValL1={val_l1:.4f} ValEdge={val_edge:.4f} ValSSIM={val_ssim:.4f} ValScore={val_score:.4f}"
            )
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{g_total_mean:.6f}",
                    f"{g_adv_mean:.6f}",
                    f"{g_recon_mean:.6f}",
                    f"{g_fm_mean:.6f}",
                    f"{g_tv_mean:.6f}",
                    f"{d_loss_mean:.6f}",
                    f"{val_l1:.6f}",
                    f"{val_edge:.6f}",
                    f"{val_ssim:.6f}",
                    f"{val_score:.6f}",
                ])

            if epoch % args.sample_every == 0 or epoch == 1:
                save_preview(generator, preview_batch, fixed_noise, epoch, class_dir, device)

            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                gen_to_save = generator.module if isinstance(generator, DDP) else generator
                disc_to_save = discriminator.module if isinstance(discriminator, DDP) else discriminator
                torch.save(gen_to_save.state_dict(), os.path.join(class_dir, "models", "generator.pt"))
                torch.save(disc_to_save.state_dict(), os.path.join(class_dir, "models", "discriminator.pt"))
                torch.save(gen_to_save.state_dict(), os.path.join(class_dir, "models", f"generator_epoch_{epoch:04d}.pt"))

    if is_main_process(rank):
        plot_history(class_dir, history)
        eval_ds = val_ds if len(val_ds) > 0 else train_ds
        best_generator = Generator(noise_ch=args.noise_ch, base_ch=args.base_ch).to(device)
        best_generator.load_state_dict(torch.load(os.path.join(class_dir, "models", "generator.pt"), map_location=device))
        stats = collect_distribution_statistics(best_generator, eval_ds, args, device)
        save_distribution_plots(class_dir, stats)
        with open(os.path.join(class_dir, "best_epoch_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["best_epoch", best_epoch])
            writer.writerow(["best_val_score", f"{best_score:.6f}"])
            writer.writerow(["best_model_rule", "saved_only_when_val_score_improves"])
            writer.writerow(["class_lambda_l1", f"{class_cfg['lambda_l1']:.6f}"])
            writer.writerow(["class_lambda_edge", f"{class_cfg['lambda_edge']:.6f}"])
            writer.writerow(["class_lambda_fm", f"{class_cfg['lambda_fm']:.6f}"])
            writer.writerow(["class_lambda_tv", f"{class_cfg['lambda_tv']:.6f}"])
            for key, value in stats["summary"].items():
                writer.writerow([key, f"{value:.6f}"])

    ddp_barrier(distributed)


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed()
    try:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            seed_everything(SEED + class_idx * 100 + rank)
            train_single_class(class_name, class_idx, args, distributed, rank, world_size, local_rank, device)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

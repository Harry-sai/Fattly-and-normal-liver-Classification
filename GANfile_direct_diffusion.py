"""
Mask-conditioned latent diffusion for liver ROI synthesis.

This version is designed to generate sharper liver regions than the previous
pixel-space diffusion baseline by:
- training on larger square ROIs extracted from the liver mask
- applying light spatial/intensity augmentation during training
- learning an AutoencoderKL first, then running diffusion in latent space
- conditioning generation on the resized liver mask plus disease label
- using EMA weights and fixed validation masks for visual tracking

Examples:
    conda run -n monaienv python GANfile_direct_diffusion.py
    conda run -n monaienv python GANfile_direct_diffusion.py --img-size 256 --batch-size 4
    conda run -n monaienv python GANfile_direct_diffusion.py --gpus 2
    conda run -n monaienv python GANfile_direct_diffusion.py --debug
"""

import argparse
import copy
import csv
import importlib
import os
import random
import site
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


IMPORT_WARNING = None


def import_runtime_modules():
    """
    Prefer the active environment, but fall back to user-site packages if the
    conda env is incomplete. This machine currently has a mixed Python setup.
    """
    original_path = list(sys.path)
    user_site = None
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    global IMPORT_WARNING
    errors = []
    candidate_paths = []
    env_prefix = os.path.abspath(sys.prefix)
    env_site_paths = []
    try:
        env_site_paths = [p for p in site.getsitepackages() if os.path.abspath(p).startswith(env_prefix)]
    except Exception:
        env_site_paths = []

    cleaned = list(original_path)
    if user_site:
        cleaned = [p for p in cleaned if os.path.abspath(p) != os.path.abspath(user_site)]

    if env_site_paths:
        env_site_abs = {os.path.abspath(p) for p in env_site_paths}
        env_first = [p for p in cleaned if os.path.abspath(p) not in env_site_abs]
        candidate_paths.append(env_site_paths + env_first)

    if user_site:
        candidate_paths.append(cleaned)
    candidate_paths.append(original_path)

    for candidate in candidate_paths:
        sys.path[:] = candidate
        try:
            torch_mod = importlib.import_module("torch")
            dist_mod = importlib.import_module("torch.distributed")
            mp_mod = importlib.import_module("torch.multiprocessing")
            nn_mod = importlib.import_module("torch.nn")
            fn_mod = importlib.import_module("torch.nn.functional")
            ddp_mod = importlib.import_module("torch.nn.parallel")
            data_mod = importlib.import_module("torch.utils.data")
            sampler_mod = importlib.import_module("torch.utils.data.distributed")
            monai_nets = importlib.import_module("monai.networks.nets")
            monai_sched = importlib.import_module("monai.networks.schedulers")
            tv_utils = importlib.import_module("torchvision.utils")
            if user_site and os.path.abspath(torch_mod.__file__).startswith(os.path.abspath(user_site)):
                IMPORT_WARNING = (
                    "Torch is being imported from user-site instead of the active conda env. "
                    f"Using fallback path: {torch_mod.__file__}"
                )
            return (
                torch_mod,
                dist_mod,
                mp_mod,
                nn_mod,
                fn_mod,
                ddp_mod.DistributedDataParallel,
                data_mod.DataLoader,
                data_mod.Dataset,
                data_mod.random_split,
                sampler_mod.DistributedSampler,
                monai_nets.AutoencoderKL,
                monai_nets.DiffusionModelUNet,
                monai_sched.DDIMScheduler,
                monai_sched.DDPMScheduler,
                tv_utils.make_grid,
                tv_utils.save_image,
            )
        except Exception as exc:
            errors.append(str(exc))
            for module_name in [
                "torchvision.utils",
                "monai.networks.schedulers",
                "monai.networks.nets",
                "torch.utils.data.distributed",
                "torch.utils.data",
                "torch.nn.parallel",
                "torch.nn.functional",
                "torch.nn",
                "torch.multiprocessing",
                "torch.distributed",
                "torch",
            ]:
                sys.modules.pop(module_name, None)

    raise RuntimeError(
        "Unable to import torch/monai runtime. Checked both environment-only and "
        f"user-site paths. Errors: {' | '.join(errors)}"
    )


(
    torch,
    dist,
    mp,
    nn,
    F,
    DDP,
    DataLoader,
    Dataset,
    random_split,
    DistributedSampler,
    AutoencoderKL,
    DiffusionModelUNet,
    DDIMScheduler,
    DDPMScheduler,
    make_grid,
    save_image,
) = import_runtime_modules()


SEED = 42
DATA_ROOT = "data/images"
MASKS_ROOT = "data/masks"
RUN_NAME = "liver_mask_latent_diffusion_v1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("PYTHONNOUSERSITE", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs-ae", type=int, default=60)
    parser.add_argument("--epochs-diff", type=int, default=200)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--diff-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=80)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--mask-loss-weight", type=float, default=4.0)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--cache-data", dest="cache_data", action="store_true")
    parser.add_argument("--no-cache-data", dest="cache_data", action="store_false")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(cache_data=True)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_labeled_pairs(images_root, masks_root):
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    pairs = []
    mask_lookup = {}

    for m in list(masks_root.rglob("*.PNG")) + list(masks_root.rglob("*.png")):
        cls = m.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            mask_lookup[(cls, m.stem.lower())] = m

    for img in list(images_root.rglob("*.PNG")) + list(images_root.rglob("*.png")):
        cls = img.parent.name.lower()
        if cls in ["normal", "fatty_liver"]:
            key = (cls, img.stem.lower())
            if key in mask_lookup:
                label = 0 if cls == "normal" else 1
                pairs.append((str(img), str(mask_lookup[key]), label))

    return pairs


def robust_normalize(image):
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi - lo < 1e-6:
        image = np.clip(image / 255.0, 0.0, 1.0)
    else:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32)


def square_crop_from_mask(img, mask, training=False):
    if mask.shape != img.shape:
        mask = np.array(
            Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        )

    mask_bin = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)

    h, w = img.shape
    if len(xs) == 0:
        side = min(h, w)
        y1 = max(0, (h - side) // 2)
        x1 = max(0, (w - side) // 2)
        return img[y1:y1 + side, x1:x1 + side], mask_bin[y1:y1 + side, x1:x1 + side]

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1

    margin = int(0.18 * max(box_h, box_w))
    if training:
        margin = int(margin * random.uniform(0.85, 1.25))

    side = max(box_h, box_w) + 2 * margin
    if training:
        side = int(side * random.uniform(0.95, 1.10))

    cy = 0.5 * (y1 + y2)
    cx = 0.5 * (x1 + x2)
    if training:
        cy += random.uniform(-0.05, 0.05) * side
        cx += random.uniform(-0.05, 0.05) * side

    side = int(max(32, min(side, max(h, w))))
    y1 = int(round(cy - side / 2))
    x1 = int(round(cx - side / 2))
    y1 = max(0, min(y1, h - side))
    x1 = max(0, min(x1, w - side))
    y2 = y1 + side
    x2 = x1 + side

    return img[y1:y2, x1:x2], mask_bin[y1:y2, x1:x2]


def augment_roi(image, mask):
    img_pil = Image.fromarray((image * 255.0).astype(np.uint8))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

    angle = random.uniform(-10.0, 10.0)
    img_pil = img_pil.rotate(angle, resample=Image.BILINEAR)
    mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)

    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.9, 1.15))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.95, 1.08))

    image = np.array(img_pil).astype(np.float32) / 255.0
    mask = (np.array(mask_pil).astype(np.float32) > 127).astype(np.float32)

    if random.random() < 0.35:
        image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape), 0.0, 1.0)

    return image, mask


class LiverROIDataset(Dataset):
    def __init__(self, pairs, img_size, training=False, cache_data=True):
        self.pairs = pairs
        self.img_size = img_size
        self.training = training
        self.cache_data = cache_data
        self.cached_items = []

        if self.cache_data:
            for idx in range(len(self.pairs)):
                self.cached_items.append(self._load_base_item(idx))

    def __len__(self):
        return len(self.pairs)

    def _load_base_item(self, idx):
        img_path, mask_path, label = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        roi, roi_mask = square_crop_from_mask(img, mask, training=False)
        roi = robust_normalize(roi)
        roi_mask = roi_mask.astype(np.float32)

        roi = np.array(
            Image.fromarray((roi * 255.0).astype(np.uint8)).resize(
                (self.img_size, self.img_size), resample=Image.BICUBIC
            )
        ).astype(np.float32) / 255.0
        roi_mask = np.array(
            Image.fromarray((roi_mask * 255).astype(np.uint8)).resize(
                (self.img_size, self.img_size), resample=Image.NEAREST
            )
        ).astype(np.float32)
        roi_mask = (roi_mask > 127).astype(np.float32)

        return roi, roi_mask, label

    def __getitem__(self, idx):
        if self.cache_data:
            roi, roi_mask, label = self.cached_items[idx]
            roi = roi.copy()
            roi_mask = roi_mask.copy()
        else:
            roi, roi_mask, label = self._load_base_item(idx)

        if self.training:
            roi, roi_mask = augment_roi(roi, roi_mask)

        liver_only = roi * roi_mask
        liver_only = liver_only * 2.0 - 1.0
        roi_mask = roi_mask * 2.0 - 1.0

        return (
            torch.from_numpy(liver_only).unsqueeze(0),
            torch.from_numpy(roi_mask).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )


class LatentConditionedUNet(nn.Module):
    def __init__(self, latent_channels=4, num_classes=2, embed_dim=128):
        super().__init__()
        self.null_class = num_classes
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=latent_channels + 1,
            out_channels=latent_channels,
            channels=(128, 192, 256),
            attention_levels=(False, True, True),
            num_res_blocks=2,
            norm_num_groups=16,
            with_conditioning=True,
            cross_attention_dim=embed_dim,
        )

    def forward(self, latent, mask_latent, t, y=None, drop_prob=0.0):
        batch = latent.shape[0]
        if y is None:
            y = torch.full((batch,), self.null_class, device=latent.device, dtype=torch.long)
        else:
            y = y.long()
            if self.training and drop_prob > 0:
                dropped = torch.rand(batch, device=latent.device) < drop_prob
                y = y.clone()
                y[dropped] = self.null_class

        context = self.class_embed(y).unsqueeze(1)
        inputs = torch.cat([latent, mask_latent], dim=1)
        return self.unet(inputs, t, context)


class EMA:
    def __init__(self, model, decay, device):
        self.decay = decay
        self.shadow = copy.deepcopy(model).to(device).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
        for shadow_buffer, model_buffer in zip(self.shadow.buffers(), model.buffers()):
            shadow_buffer.data.copy_(model_buffer.data)


def is_main(rank):
    return rank == 0


def log(message):
    print(message, flush=True)


def reduce_mean(value, world_size):
    if world_size == 1:
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= world_size
    return out


def reduce_scalar_sum(value, device, world_size):
    tensor = torch.tensor(float(value), device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def ddp_barrier(rank, world_size):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[rank])
        else:
            dist.barrier()


def broadcast_model_parameters(model, src, world_size):
    if world_size == 1 or not dist.is_available() or not dist.is_initialized():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=src)


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


def weighted_recon_loss(recon, target, mask, edge_weight, mask_loss_weight):
    mask_01 = (mask + 1.0) * 0.5
    weights = 1.0 + mask_loss_weight * mask_01
    recon_l1 = (weights * (recon - target).abs()).mean()
    edge_l1 = (sobel_edges(recon) - sobel_edges(target)).abs().mean()
    return recon_l1 + edge_weight * edge_l1


def build_loaders(args, rank, world_size, use_cuda):
    pairs = collect_labeled_pairs(DATA_ROOT, MASKS_ROOT)
    if len(pairs) == 0:
        raise RuntimeError("No image/mask pairs were found in data/images and data/masks.")

    shuffled = list(pairs)
    random.Random(SEED).shuffle(shuffled)

    val_len = max(1, int(len(shuffled) * args.val_split))
    train_pairs = shuffled[val_len:]
    val_pairs = shuffled[:val_len]

    if args.debug:
        train_pairs = train_pairs[: min(24, len(train_pairs))]
        val_pairs = val_pairs[: min(8, len(val_pairs))]

    train_ds = LiverROIDataset(train_pairs, args.img_size, training=True, cache_data=args.cache_data)
    val_ds = LiverROIDataset(val_pairs, args.img_size, training=False, cache_data=args.cache_data)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset split produced an empty training or validation set.")

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    # This machine has been showing stalls after a few prefetched batches when
    # combining DDP, cached datasets, and DataLoader worker processes. Cached
    # data is already in memory, so using the main process is usually faster
    # and much more stable here.
    if args.cache_data:
        effective_workers = 0
    elif use_cuda:
        effective_workers = min(args.num_workers, 2 if world_size > 1 else args.num_workers)
    else:
        effective_workers = 0
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None,
        drop_last=False,
    )

    preview_count = min(args.preview_count, len(val_ds))
    preview_items = [val_ds[i] for i in range(preview_count)]
    preview_images = torch.stack([item[0] for item in preview_items], dim=0)
    preview_masks = torch.stack([item[1] for item in preview_items], dim=0)
    preview_labels = torch.stack([item[2] for item in preview_items], dim=0)

    return (
        shuffled,
        train_loader,
        val_loader,
        train_sampler,
        (preview_images, preview_masks, preview_labels),
        effective_workers,
    )


@torch.no_grad()
def estimate_latent_scale(autoencoder, loader, device, use_cuda, world_size, max_batches=8):
    autoencoder.eval()
    std_values = []

    for step, (images, _, _) in enumerate(loader):
        if step >= max_batches:
            break
        images = images.to(device, non_blocking=use_cuda)
        z_mu, z_sigma = autoencoder.encode(images)
        z = autoencoder.sampling(z_mu, z_sigma)
        std_values.append(z.float().std())

    latent_std = torch.stack(std_values).mean() if std_values else torch.tensor(1.0, device=device)
    latent_std = reduce_mean(latent_std, world_size).clamp_min(1e-6)
    return 1.0 / latent_std.item()


@torch.no_grad()
def encode_latents(autoencoder, images, masks, latent_scale):
    z_mu, z_sigma = autoencoder.encode(images)
    latents = autoencoder.sampling(z_mu, z_sigma) * latent_scale
    mask_latent = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
    mask_latent = (mask_latent > 0).to(latents.dtype)
    return latents, mask_latent


@torch.no_grad()
def decode_latents(autoencoder, latents, latent_scale):
    recon = autoencoder.decode(latents / latent_scale)
    return recon.clamp(-1.0, 1.0)


def save_loss_plot(base_dir, ae_train_losses, ae_val_losses, diff_train_losses, diff_val_losses):
    if plt is None:
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ae_train_losses, label="Train")
    plt.plot(ae_val_losses, label="Val")
    plt.title("Autoencoder Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(diff_train_losses, label="Train")
    plt.plot(diff_val_losses, label="Val")
    plt.title("Latent Diffusion Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Noise MSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "loss_curves.png"))
    plt.close()


@torch.no_grad()
def save_autoencoder_preview(base_dir, autoencoder, device, preview_batch, latent_scale):
    images, masks, _ = preview_batch
    images = images.to(device)
    masks = masks.to(device)

    z_mu, z_sigma = autoencoder.encode(images)
    latents = autoencoder.sampling(z_mu, z_sigma) * latent_scale
    recon = decode_latents(autoencoder, latents, latent_scale)

    mask_vis = (masks + 1.0) * 0.5
    image_vis = (images + 1.0) * 0.5
    recon_vis = (recon + 1.0) * 0.5
    grid = torch.cat([image_vis.cpu(), mask_vis.cpu(), recon_vis.cpu()], dim=0)
    save_image(grid, os.path.join(base_dir, "samples", "autoencoder_preview.png"), nrow=images.size(0))


@torch.no_grad()
def sample_from_mask(model, autoencoder, mask, label, device, args, latent_scale):
    model.eval()
    autoencoder.eval()

    latent_h = args.img_size // 4
    latent_w = args.img_size // 4
    mask = mask.to(device)
    mask_latent = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
    mask_latent = (mask_latent > 0).float()

    ddim = DDIMScheduler(num_train_timesteps=args.timesteps, clip_sample=False)
    x = torch.randn(1, args.latent_channels, latent_h, latent_w, device=device)
    y = torch.tensor([label], device=device)
    ddim.set_timesteps(args.sample_steps)

    for t in ddim.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        eps_cond = model(x, mask_latent, t_batch, y, drop_prob=0.0)
        eps_uncond = model(x, mask_latent, t_batch, None, drop_prob=0.0)
        eps = eps_uncond + args.guidance_scale * (eps_cond - eps_uncond)
        x, _ = ddim.step(eps, t, x)

    recon = decode_latents(autoencoder, x, latent_scale)
    return recon.clamp(-1.0, 1.0)


@torch.no_grad()
def save_diffusion_preview(base_dir, model, autoencoder, device, preview_batch, args, latent_scale, name):
    images, masks, labels = preview_batch
    images = images.to(device)
    masks = masks.to(device)
    labels = labels.to(device)

    generated = []
    for idx in range(images.size(0)):
        sample = sample_from_mask(
            model=model,
            autoencoder=autoencoder,
            mask=masks[idx : idx + 1],
            label=int(labels[idx].item()),
            device=device,
            args=args,
            latent_scale=latent_scale,
        )
        generated.append(sample.cpu())

    generated = torch.cat(generated, dim=0)
    mask_vis = (masks.cpu() + 1.0) * 0.5
    image_vis = (images.cpu() + 1.0) * 0.5
    gen_vis = (generated + 1.0) * 0.5
    grid = torch.cat([image_vis, mask_vis, gen_vis], dim=0)
    save_image(grid, os.path.join(base_dir, "samples", name), nrow=images.size(0))


def train_worker(rank, world_size, args, sync_file):
    seed_everything(SEED + rank)

    use_cuda = torch.cuda.is_available()
    if world_size > 1 and not use_cuda:
        raise RuntimeError("Multi-GPU mode was requested but CUDA is not available.")

    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    try:
        if world_size > 1:
            log(f"[Rank {rank}] Initializing distributed process group...")
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{sync_file}",
                rank=rank,
                world_size=world_size,
            )
            log(f"[Rank {rank}] Distributed process group ready on {device}.")

        base_dir = os.path.join("GAN", RUN_NAME)
        ensure_dir(base_dir)
        ensure_dir(os.path.join(base_dir, "samples"))
        ensure_dir(os.path.join(base_dir, "models"))
        ensure_dir(os.environ["MPLCONFIGDIR"])

        if is_main(rank):
            with open(os.path.join(base_dir, "hyperparameters.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                for key, value in sorted(vars(args).items()):
                    writer.writerow([key.upper(), value])

        pairs, train_loader, val_loader, train_sampler, preview_batch, effective_workers = build_loaders(
            args, rank, world_size, use_cuda
        )

        if is_main(rank):
            log(f"Using device: {device}")
            log(f"World size: {world_size}")
            log(f"Python executable: {sys.executable}")
            log(f"Torch path: {torch.__file__}")
            log(f"Torch version: {torch.__version__}")
            if IMPORT_WARNING:
                log(f"WARNING: {IMPORT_WARNING}")
            log(f"Visible CUDA devices: {torch.cuda.device_count() if use_cuda else 0}")
            log(f"Total samples: {len(pairs)}")
            log(
                f"Train batches/epoch: {len(train_loader)} | Val batches/epoch: {len(val_loader)} | "
                f"Preview every {args.preview_every} diffusion epochs | "
                f"cache_data={args.cache_data} | workers={effective_workers}"
            )

        autoencoder = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256),
            attention_levels=(False, False, True),
            latent_channels=args.latent_channels,
            num_res_blocks=2,
            norm_num_groups=16,
        ).to(device)
        diffusion = LatentConditionedUNet(latent_channels=args.latent_channels).to(device)

        if world_size > 1:
            autoencoder = DDP(
                autoencoder,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            diffusion = DDP(
                diffusion,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        raw_autoencoder = autoencoder.module if world_size > 1 else autoencoder
        raw_diffusion = diffusion.module if world_size > 1 else diffusion
        ema = EMA(raw_diffusion, decay=args.ema_decay, device=device)

        ae_optimizer = torch.optim.AdamW(
            autoencoder.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay
        )
        diff_optimizer = torch.optim.AdamW(
            diffusion.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay
        )
        ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=args.epochs_ae)
        diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(diff_optimizer, T_max=args.epochs_diff)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        mse = nn.MSELoss()
        ddpm = DDPMScheduler(num_train_timesteps=args.timesteps, clip_sample=False)

        ae_train_losses = []
        ae_val_losses = []
        diff_train_losses = []
        diff_val_losses = []
        best_ae_val = float("inf")
        best_diff_val = float("inf")
        best_ae_path = os.path.join(base_dir, "models", "best_autoencoder.pt")
        best_diff_path = os.path.join(base_dir, "models", "best_mask_latent_diffusion.pt")

        max_train_batches = 3 if args.debug else None
        max_val_batches = 2 if args.debug else None

        for epoch in range(args.epochs_ae):
            epoch_start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if is_main(rank):
                log(f"[AE {epoch + 1:03d}/{args.epochs_ae}] starting training epoch")

            autoencoder.train()
            train_total = 0.0
            train_steps = 0

            for step, (images, masks, _) in enumerate(train_loader):
                if max_train_batches is not None and step >= max_train_batches:
                    break

                images = images.to(device, non_blocking=use_cuda)
                masks = masks.to(device, non_blocking=use_cuda)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    recon, z_mu, z_sigma = autoencoder(images)
                    recon_loss = weighted_recon_loss(
                        recon, images, masks, args.edge_weight, args.mask_loss_weight
                    )
                    kl_loss = torch.mean(
                        -0.5 * torch.sum(
                            1 + torch.log(z_sigma.square() + 1e-6) - z_mu.square() - z_sigma.square(),
                            dim=[1, 2, 3],
                        )
                    )
                    loss = recon_loss + args.kl_weight * kl_loss

                ae_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(ae_optimizer)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
                scaler.step(ae_optimizer)
                scaler.update()

                train_total += loss.detach().item()
                train_steps += 1

                if is_main(rank) and ((step + 1) % max(1, args.log_every) == 0):
                    log(
                        f"[AE {epoch + 1:03d}/{args.epochs_ae}] "
                        f"step={step + 1}/{len(train_loader)} "
                        f"loss={loss.detach().item():.5f} "
                        f"elapsed={time.time() - epoch_start:.1f}s"
                    )

            autoencoder.eval()
            val_total = 0.0
            val_steps = 0
            if is_main(rank):
                log(f"[AE {epoch + 1:03d}/{args.epochs_ae}] starting validation")

            with torch.no_grad():
                for step, (images, masks, _) in enumerate(val_loader):
                    if max_val_batches is not None and step >= max_val_batches:
                        break

                    images = images.to(device, non_blocking=use_cuda)
                    masks = masks.to(device, non_blocking=use_cuda)

                    recon, z_mu, z_sigma = autoencoder(images)
                    recon_loss = weighted_recon_loss(
                        recon, images, masks, args.edge_weight, args.mask_loss_weight
                    )
                    kl_loss = torch.mean(
                        -0.5 * torch.sum(
                            1 + torch.log(z_sigma.square() + 1e-6) - z_mu.square() - z_sigma.square(),
                            dim=[1, 2, 3],
                        )
                    )
                    loss = recon_loss + args.kl_weight * kl_loss
                    val_total += loss.detach().item()
                    val_steps += 1

            train_total = reduce_scalar_sum(train_total, device, world_size)
            val_total = reduce_scalar_sum(val_total, device, world_size)
            total_train_steps = reduce_scalar_sum(train_steps, device, world_size)
            total_val_steps = reduce_scalar_sum(val_steps, device, world_size)
            train_loss = train_total / max(1.0, total_train_steps)
            val_loss = val_total / max(1.0, total_val_steps)
            ae_train_losses.append(train_loss)
            ae_val_losses.append(val_loss)
            ae_scheduler.step()

            if val_loss < best_ae_val:
                best_ae_val = val_loss
                if is_main(rank):
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model": raw_autoencoder.state_dict(),
                            "val_loss": val_loss,
                            "args": vars(args),
                        },
                        best_ae_path,
                    )

            if is_main(rank):
                log(
                    f"[AE {epoch + 1:03d}/{args.epochs_ae}] "
                    f"train={train_loss:.5f} val={val_loss:.5f} "
                    f"lr={ae_scheduler.get_last_lr()[0]:.6f} "
                    f"time={time.time() - epoch_start:.1f}s"
                )

        ddp_barrier(rank, world_size)
        if is_main(rank):
            ae_checkpoint = torch.load(best_ae_path, map_location=device)
            raw_autoencoder.load_state_dict(ae_checkpoint["model"])
        broadcast_model_parameters(raw_autoencoder, src=0, world_size=world_size)
        raw_autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad = False

        latent_scale = estimate_latent_scale(
            raw_autoencoder, train_loader, device, use_cuda, world_size, max_batches=8
        )

        if is_main(rank):
            log(f"Estimated latent scale: {latent_scale:.6f}")
            save_autoencoder_preview(base_dir, raw_autoencoder, device, preview_batch, latent_scale)
        ddp_barrier(rank, world_size)

        for epoch in range(args.epochs_diff):
            epoch_start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(args.epochs_ae + epoch)
            if is_main(rank):
                log(f"[DIFF {epoch + 1:03d}/{args.epochs_diff}] starting training epoch")

            diffusion.train()
            train_total = 0.0
            train_steps = 0

            for step, (images, masks, labels) in enumerate(train_loader):
                if max_train_batches is not None and step >= max_train_batches:
                    break

                images = images.to(device, non_blocking=use_cuda)
                masks = masks.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)

                with torch.no_grad():
                    latents, mask_latent = encode_latents(raw_autoencoder, images, masks, latent_scale)

                noise = torch.randn_like(latents)
                t = torch.randint(0, args.timesteps, (latents.size(0),), device=device).long()
                noisy_latents = ddpm.add_noise(latents, noise, t)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    pred_noise = diffusion(
                        noisy_latents, mask_latent, t, labels, drop_prob=args.cfg_dropout
                    )
                    loss = mse(pred_noise, noise)

                diff_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(diff_optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
                scaler.step(diff_optimizer)
                scaler.update()
                ema.update(raw_diffusion)

                train_total += loss.detach().item()
                train_steps += 1

                if is_main(rank) and ((step + 1) % max(1, args.log_every) == 0):
                    log(
                        f"[DIFF {epoch + 1:03d}/{args.epochs_diff}] "
                        f"step={step + 1}/{len(train_loader)} "
                        f"loss={loss.detach().item():.5f} "
                        f"elapsed={time.time() - epoch_start:.1f}s"
                    )

            diffusion.eval()
            val_total = 0.0
            val_steps = 0
            if is_main(rank):
                log(f"[DIFF {epoch + 1:03d}/{args.epochs_diff}] starting validation")

            with torch.no_grad():
                for step, (images, masks, labels) in enumerate(val_loader):
                    if max_val_batches is not None and step >= max_val_batches:
                        break

                    images = images.to(device, non_blocking=use_cuda)
                    masks = masks.to(device, non_blocking=use_cuda)
                    labels = labels.to(device, non_blocking=use_cuda)

                    latents, mask_latent = encode_latents(raw_autoencoder, images, masks, latent_scale)
                    noise = torch.randn_like(latents)
                    t = torch.randint(0, args.timesteps, (latents.size(0),), device=device).long()
                    noisy_latents = ddpm.add_noise(latents, noise, t)
                    pred_noise = diffusion(noisy_latents, mask_latent, t, labels, drop_prob=0.0)
                    loss = mse(pred_noise, noise)
                    val_total += loss.detach().item()
                    val_steps += 1

            train_total = reduce_scalar_sum(train_total, device, world_size)
            val_total = reduce_scalar_sum(val_total, device, world_size)
            total_train_steps = reduce_scalar_sum(train_steps, device, world_size)
            total_val_steps = reduce_scalar_sum(val_steps, device, world_size)
            train_loss = train_total / max(1.0, total_train_steps)
            val_loss = val_total / max(1.0, total_val_steps)
            diff_train_losses.append(train_loss)
            diff_val_losses.append(val_loss)
            diff_scheduler.step()

            if val_loss < best_diff_val:
                best_diff_val = val_loss
                if is_main(rank):
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "ema": ema.shadow.state_dict(),
                            "latent_scale": latent_scale,
                            "val_loss": val_loss,
                            "args": vars(args),
                        },
                        best_diff_path,
                    )

            if is_main(rank):
                log(
                    f"[DIFF {epoch + 1:03d}/{args.epochs_diff}] "
                    f"train={train_loss:.5f} val={val_loss:.5f} "
                    f"lr={diff_scheduler.get_last_lr()[0]:.6f} "
                    f"time={time.time() - epoch_start:.1f}s"
                )
                if ((epoch + 1) % max(1, args.preview_every) == 0) or (epoch + 1 == args.epochs_diff):
                    save_diffusion_preview(
                        base_dir,
                        ema.shadow,
                        raw_autoencoder,
                        device,
                        preview_batch,
                        args,
                        latent_scale,
                        "latest_samples.png",
                    )
            ddp_barrier(rank, world_size)

        if is_main(rank):
            diff_checkpoint = torch.load(best_diff_path, map_location=device)
            ema.shadow.load_state_dict(diff_checkpoint["ema"])
            save_diffusion_preview(
                base_dir,
                ema.shadow,
                raw_autoencoder,
                device,
                preview_batch,
                args,
                latent_scale,
                "final_samples.png",
            )
            save_loss_plot(
                base_dir,
                ae_train_losses,
                ae_val_losses,
                diff_train_losses,
                diff_val_losses,
            )
        ddp_barrier(rank, world_size)
    finally:
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            ddp_barrier(rank, world_size)
            dist.destroy_process_group()


def main():
    args = parse_args()
    seed_everything(SEED)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1.")
    if args.gpus > 1 and available_gpus < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} GPUs but only {available_gpus} are available.")

    if args.debug:
        args.epochs_ae = min(args.epochs_ae, 2)
        args.epochs_diff = min(args.epochs_diff, 2)
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = 0
        args.preview_count = min(args.preview_count, 2)
        args.preview_every = 1
        args.log_every = 1

    if args.gpus > 1:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            sync_file = tmp.name
        try:
            mp.spawn(train_worker, args=(args.gpus, args, sync_file), nprocs=args.gpus, join=True)
        finally:
            if os.path.exists(sync_file):
                os.remove(sync_file)
    else:
        train_worker(rank=0, world_size=1, args=args, sync_file="")


if __name__ == "__main__":
    main()

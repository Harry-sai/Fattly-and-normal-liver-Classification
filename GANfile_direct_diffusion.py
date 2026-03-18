"""
Direct conditional diffusion for liver ROI generation.

This version is written to be runnable on this machine without relying on
`torchrun`. It can:
- run on CPU
- run on a single GPU
- spawn multi-GPU training from the same script with `--gpus 2`

Examples:
    conda run -n monaienv python GANfile_direct_diffusion.py --epochs 5
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
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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

    errors = []
    candidate_paths = []

    if user_site:
        cleaned = [p for p in original_path if os.path.abspath(p) != os.path.abspath(user_site)]
        candidate_paths.append(cleaned)
    candidate_paths.append(original_path)

    for candidate in candidate_paths:
        sys.path[:] = candidate
        try:
            torch_mod = importlib.import_module("torch")
            dist_mod = importlib.import_module("torch.distributed")
            mp_mod = importlib.import_module("torch.multiprocessing")
            nn_mod = importlib.import_module("torch.nn")
            ddp_mod = importlib.import_module("torch.nn.parallel")
            data_mod = importlib.import_module("torch.utils.data")
            sampler_mod = importlib.import_module("torch.utils.data.distributed")
            monai_nets = importlib.import_module("monai.networks.nets")
            monai_sched = importlib.import_module("monai.networks.schedulers")
            tv_utils = importlib.import_module("torchvision.utils")
            return (
                torch_mod,
                dist_mod,
                mp_mod,
                nn_mod,
                ddp_mod.DistributedDataParallel,
                data_mod.DataLoader,
                data_mod.Dataset,
                data_mod.random_split,
                sampler_mod.DistributedSampler,
                monai_nets.DiffusionModelUNet,
                monai_sched.DDIMScheduler,
                monai_sched.DDPMScheduler,
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
    DDP,
    DataLoader,
    Dataset,
    random_split,
    DistributedSampler,
    DiffusionModelUNet,
    DDIMScheduler,
    DDPMScheduler,
    save_image,
) = import_runtime_modules()


SEED = 42
DATA_ROOT = "data/images"
MASKS_ROOT = "data/masks"
RUN_NAME = "liver_direct_diffusion_v2"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--cfg-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gpus", type=int, default=1)
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


def make_roi(img, mask, pad=6):
    if mask.shape != img.shape:
        mask = np.array(
            Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
        )
    mask = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        h, w = img.shape
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return img[y1:y1 + side, x1:x1 + side]

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    y1 = max(y1 - pad, 0)
    y2 = min(y2 + pad, img.shape[0])
    x1 = max(x1 - pad, 0)
    x2 = min(x2 + pad, img.shape[1])
    return img[y1:y2, x1:x2]


class LiverROIDataset(Dataset):
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
        roi = np.array(
            Image.fromarray(roi).resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        )
        roi = roi.astype(np.float32) / 255.0
        roi = roi * 2.0 - 1.0

        return torch.from_numpy(roi).unsqueeze(0), torch.tensor(label, dtype=torch.long)


class ConditionedUNet(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64):
        super().__init__()
        self.null_class = num_classes
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 96, 128),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            norm_num_groups=16,
            with_conditioning=True,
            cross_attention_dim=embed_dim,
        )

    def forward(self, x, t, y=None, drop_prob=0.0):
        batch = x.shape[0]
        if y is None:
            y = torch.full((batch,), self.null_class, device=x.device, dtype=torch.long)
        else:
            y = y.long()
            if self.training and drop_prob > 0:
                dropped = torch.rand(batch, device=x.device) < drop_prob
                y = y.clone()
                y[dropped] = self.null_class

        context = self.class_embed(y).unsqueeze(1)
        return self.unet(x, t, context)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).cpu().eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for key, value in self.shadow.state_dict().items():
            source = model_state[key].detach().cpu()
            value.copy_(value * self.decay + source * (1.0 - self.decay))


def is_main(rank):
    return rank == 0


def reduce_mean(value, world_size):
    if world_size == 1:
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= world_size
    return out


def build_loaders(args, rank, world_size, use_cuda):
    pairs = collect_labeled_pairs(DATA_ROOT, MASKS_ROOT)
    if len(pairs) == 0:
        raise RuntimeError("No image/mask pairs were found in data/images and data/masks.")

    full_dataset = LiverROIDataset(pairs, args.img_size)
    val_len = max(1, int(len(full_dataset) * args.val_split))
    train_len = len(full_dataset) - val_len
    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=split_gen)

    if args.debug:
        train_ds = torch.utils.data.Subset(train_ds, range(min(16, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(8, len(val_ds))))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    effective_workers = args.num_workers if use_cuda else 0
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return pairs, train_loader, val_loader, train_sampler


def sample_one(model, device, args, label):
    model.eval()
    ddim = DDIMScheduler(num_train_timesteps=args.timesteps)
    x = torch.randn(1, 1, args.img_size, args.img_size, device=device)
    y = torch.tensor([label], device=device)
    ddim.set_timesteps(args.sample_steps)

    for t in ddim.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        eps_cond = model(x, t_batch, y, drop_prob=0.0)
        eps_uncond = model(x, t_batch, None, drop_prob=0.0)
        eps = eps_uncond + args.guidance_scale * (eps_cond - eps_uncond)
        x, _ = ddim.step(eps, t, x)

    return ((x + 1.0) / 2.0).clamp(0, 1)


def save_epoch_samples(base_dir, ema_model, device, args):
    ema_model.to(device)
    torch.manual_seed(SEED)
    normal = sample_one(ema_model, device, args, label=0)
    torch.manual_seed(SEED + 1)
    fatty = sample_one(ema_model, device, args, label=1)

    save_image(normal.cpu(), os.path.join(base_dir, "samples", "latest_normal.png"))
    save_image(fatty.cpu(), os.path.join(base_dir, "samples", "latest_fatty.png"))
    ema_model.to("cpu")


def save_final_samples(base_dir, ema_model, device, args):
    ema_model.to(device)
    for i in range(5):
        torch.manual_seed(SEED + i)
        normal = sample_one(ema_model, device, args, label=0)
        save_image(normal.cpu(), os.path.join(base_dir, "samples", f"normal_{i}.png"))

        torch.manual_seed(SEED + 100 + i)
        fatty = sample_one(ema_model, device, args, label=1)
        save_image(fatty.cpu(), os.path.join(base_dir, "samples", f"fatty_{i}.png"))
    ema_model.to("cpu")


def save_loss_plot(base_dir, train_losses, val_losses):
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Noise MSE")
    plt.title("Direct Diffusion Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "loss_curves.png"))
    plt.close()


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
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{sync_file}",
                rank=rank,
                world_size=world_size,
            )

        base_dir = os.path.join("GAN", RUN_NAME)
        ensure_dir(base_dir)
        ensure_dir(os.path.join(base_dir, "samples"))
        ensure_dir(os.path.join(base_dir, "models"))

        if is_main(rank):
            with open(os.path.join(base_dir, "hyperparameters.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                for key, value in sorted(vars(args).items()):
                    writer.writerow([key.upper(), value])

        pairs, train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size, use_cuda)
        if is_main(rank):
            print(f"Using device: {device}")
            print(f"World size: {world_size}")
            print(f"Python executable: {sys.executable}")
            print(f"Torch path: {torch.__file__}")
            print(f"Torch version: {torch.__version__}")
            print(f"Visible CUDA devices: {torch.cuda.device_count() if use_cuda else 0}")
            print(f"Total samples: {len(pairs)}")

        model = ConditionedUNet(num_classes=2).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        raw_model = model.module if world_size > 1 else model
        ema = EMA(raw_model, decay=args.ema_decay)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        mse = nn.MSELoss()
        ddpm = DDPMScheduler(num_train_timesteps=args.timesteps)

        train_losses = []
        val_losses = []
        best_val = float("inf")

        max_train_batches = 2 if args.debug else None
        max_val_batches = 1 if args.debug else None

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_total = 0.0
            for step, (x, y) in enumerate(train_loader):
                if max_train_batches is not None and step >= max_train_batches:
                    break

                x = x.to(device, non_blocking=use_cuda)
                y = y.to(device, non_blocking=use_cuda)

                noise = torch.randn_like(x)
                t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
                noisy_x = ddpm.add_noise(x, noise, t)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    pred_noise = model(noisy_x, t, y, drop_prob=args.cfg_dropout)
                    loss = mse(pred_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update(raw_model)

                train_total += reduce_mean(loss.detach(), world_size).item()

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    if max_val_batches is not None and step >= max_val_batches:
                        break

                    x = x.to(device, non_blocking=use_cuda)
                    y = y.to(device, non_blocking=use_cuda)

                    noise = torch.randn_like(x)
                    t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
                    noisy_x = ddpm.add_noise(x, noise, t)
                    pred_noise = model(noisy_x, t, y, drop_prob=0.0)
                    loss = mse(pred_noise, noise)
                    val_total += reduce_mean(loss.detach(), world_size).item()

            train_batches = min(len(train_loader), max_train_batches or len(train_loader))
            val_batches = min(len(val_loader), max_val_batches or len(val_loader))
            train_loss = train_total / max(1, train_batches)
            val_loss = val_total / max(1, val_batches)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lr_scheduler.step()

            if is_main(rank):
                print(
                    f"[Epoch {epoch + 1:03d}/{args.epochs}] "
                    f"train={train_loss:.5f} val={val_loss:.5f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.6f}"
                )
                save_epoch_samples(base_dir, ema.shadow, device, args)

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model": raw_model.state_dict(),
                            "ema": ema.shadow.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "args": vars(args),
                        },
                        os.path.join(base_dir, "models", "best_direct_diffusion.pt"),
                    )

        if is_main(rank):
            save_final_samples(base_dir, ema.shadow, device, args)
            save_loss_plot(base_dir, train_losses, val_losses)
    finally:
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
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
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)

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

"""
train_vae_mario.py
------------------
Train a VAE on Mario frames with strong parallelism and fast paths enabled.

Outputs:
- checkpoints/vae_best.pt          (best model only)
- checkpoints/loss_history.json    (list of epoch losses)
"""

import os
import json
import time
from datetime import timedelta
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_mario_dataset import MarioFramesDataset
from vae_model import VAE, LATENT_DIM_DEFAULT


# ----------------------------
# Device helpers and fast flags
# ----------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hhmmss(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


device = pick_device()
print(f"Using device: {device}")

# Enable fast kernels on NVIDIA
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ----------------------------
# Data
# ----------------------------
DATA_ROOT = "/Users/phillipliu/Documents/UCT/Honours/Thesis/Code/capture_output"

train_dataset = MarioFramesDataset(
    root_dir=DATA_ROOT,
    image_size=128,
    return_actions=True
)

# Worker count: use many, but leave two cores free
cpu_count = os.cpu_count() or 4
num_workers = max(4, min(16, cpu_count - 2))

# Prefetch factor per worker
PREFETCH = 6

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    pin_memory_device=("cuda" if device.type == "cuda" else ""),
    persistent_workers=True,
    prefetch_factor=PREFETCH,
    drop_last=True,
    multiprocessing_context=("forkserver" if os.name != "nt" else None),
)

# ----------------------------
# Model and optimiser
# ----------------------------
model = VAE(latent_dim=LATENT_DIM_DEFAULT).to(device)
# Channels last layout speeds up many convolution kernels
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

# Try to compile on CUDA (no effect on MPS/CPU)
if device.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        print("Model compiled with torch.compile.")
    except Exception as e:
        print(f"torch.compile not used: {e}")

# Prefer fused AdamW if available on this build
try:
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=(device.type == "cuda"))
    print("Using AdamW (fused=True)" if device.type == "cuda" else "Using AdamW")
except TypeError:
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("Using AdamW")

recon_loss_fn = nn.MSELoss(reduction="mean")


# ----------------------------
# AMP (CUDA only)
# ----------------------------
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

autocast_ctx = (
    torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), enabled=True)
    if use_amp else nullcontext()
)


# ----------------------------
# Loss
# ----------------------------
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    # Rescale to [0,1] for pixel loss
    recon = recon_loss_fn((recon_x + 1) / 2, (x + 1) / 2)
    # KL divergence (mean over batch)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld, recon, kld


# ----------------------------
# CUDA prefetcher to overlap H2D copies with compute
# ----------------------------
class CUDAPrefetcher:
    """
    Wrap a DataLoader and prefetch next batch to GPU using a separate CUDA stream.
    This overlaps host to device copies with compute on the default stream.
    Only used on CUDA.
    """
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.next = None

    def __len__(self):
        return len(self.loader)

    def _prefetch(self):
        try:
            images, actions = next(self.iter)
        except StopIteration:
            self.next = None
            return
        # Transfer on the prefetch stream
        with torch.cuda.stream(self.stream):
            images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            self.next = (images, actions)

    def __iter__(self):
        self.iter = iter(self.loader)
        self._prefetch()
        while self.next is not None:
            # Wait for the prefetch stream to finish the copy
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next
            self._prefetch()
            yield batch


# ----------------------------
# One epoch
# ----------------------------
def train_one_epoch(epoch: int, run_start_time: float) -> Tuple[float, float]:
    model.train()
    start = time.perf_counter()
    running_loss = 0.0

    # Choose the most parallel iterator available
    if device.type == "cuda":
        batch_iter = CUDAPrefetcher(train_loader, device)
    else:
        batch_iter = train_loader

    progress = tqdm(batch_iter, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)

    for images, _ in progress:
        # If not using the CUDA prefetcher, copy here with non blocking and channels last
        if device.type != "cuda":
            images = images.to(device, non_blocking=False)
        optim.zero_grad(set_to_none=True)

        with autocast_ctx:
            recon, mu, logvar = model(images)
            loss, rec, kld = vae_loss_function(recon, images, mu, logvar)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        running_loss += float(loss.detach().item())
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "rec": f"{rec.item():.4f}",
            "kl": f"{kld.item():.4f}",
        })

    avg_loss = running_loss / len(train_loader)
    epoch_secs = time.perf_counter() - start
    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Time: {hhmmss(epoch_secs)} | Elapsed: {hhmmss(time.perf_counter()-run_start_time)}")
    return avg_loss, epoch_secs


# ----------------------------
# Main
# ----------------------------
def main():
    n_epochs = 1000
    run_start = time.perf_counter()

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        avg_loss, _ = train_one_epoch(epoch, run_start)
        loss_history.append(avg_loss)

        # Save best model only
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }
            torch.save(ckpt, "checkpoints/vae_best.pt")
            print(f"Saved new best model at epoch {epoch} (loss={avg_loss:.6f})")

        # Save history each epoch
        with open("checkpoints/loss_history.json", "w") as f:
            json.dump(loss_history, f)

    print("Training complete.")


if __name__ == "__main__":
    main()

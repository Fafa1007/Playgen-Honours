# e_dit_test.py
import os
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from a_load_mario_dataset import MarioTransitionsDataset
from b_vae_model import VAE, LATENT_DIM_DEFAULT
from d_dit_model import LatentDiT


# ========= Utilities =========

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_pil_from_tensor(x: torch.Tensor) -> Image.Image:
    """
    Expects a CHW tensor in [-1, 1]. Returns a PIL Image (uint8).
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = (x * 255.0).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def save_two_row_grid(
    originals: List[torch.Tensor],
    predictions: List[torch.Tensor],
    out_dir: Path,
    prefix: str = "pred_grid"
) -> Path:
    assert len(originals) == len(predictions) and len(originals) > 0
    pil_top = [to_pil_from_tensor(t) for t in originals]
    pil_bot = [to_pil_from_tensor(t) for t in predictions]

    w, h = pil_top[0].size
    n = len(pil_top)
    canvas = Image.new("RGB", (n * w, 2 * h), (0, 0, 0))

    for i, im in enumerate(pil_top):
        if im.size != (w, h):
            im = im.resize((w, h), Image.BICUBIC)
        canvas.paste(im, (i * w, 0))

    for i, im in enumerate(pil_bot):
        if im.size != (w, h):
            im = im.resize((w, h), Image.BICUBIC)
        canvas.paste(im, (i * w, h))

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_{ts}.png"
    canvas.save(out_path)
    return out_path


@torch.no_grad()
def encode_mu_and_skips(vae: VAE, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Matches your VAE API where encoder returns (mu, logvar_raw, skips).
    We use mu as the latent and keep skips for the decoder.
    """
    vae.eval()
    out = vae.encoder(x)
    if not isinstance(out, (list, tuple)) or len(out) < 3:
        raise RuntimeError("Unexpected encoder output; expected (mu, logvar_raw, skips).")
    mu, _, skips = out
    return mu, skips


@torch.no_grad()
def decode_from_latent(vae: VAE, z: torch.Tensor, skips) -> torch.Tensor:
    """
    Matches your Decoder API: decoder(z, skips=...) returns a frame in [-1, 1].
    """
    vae.eval()
    x = vae.decoder(z, skips=skips)
    return x


# ========= Core roll‑out =========

@torch.no_grad()
def rollout_sequence(
    vae: VAE,
    dit: nn.Module,
    dataset: MarioTransitionsDataset,
    start_idx: int,
    seq_len: int,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Uses transitions starting at start_idx:
      For i in 0..seq_len-1, dataset[start_idx + i] yields (x_t, a_t, x_tp1).
      Top row = [x_{t+1} ... x_{t+seq_len}]
      Bottom row = predictions decoded from DiT latents, rolling from z_t (mu of first x_t).
      Important: we hold the initial encoder skips and pass them to every decode to match your decoder shape.
    """
    xs_t, as_t, xs_tp1 = [], [], []
    for i in range(seq_len):
        x_t, a_t, x_tp1 = dataset[start_idx + i]
        xs_t.append(x_t.to(device, non_blocking=True))
        as_t.append(a_t.to(device, non_blocking=True))
        xs_tp1.append(x_tp1.to(device, non_blocking=True))

    # Encode the first frame to get z_t and its skips
    z_t, skips = encode_mu_and_skips(vae, xs_t[0].unsqueeze(0))  # z_t: [1, D]

    preds: List[torch.Tensor] = []
    for i in range(seq_len):
        a_vec = as_t[i].unsqueeze(0)          # [1, action_dim]
        z_t = dit(z_t, a_vec)                 # next latent
        x_pred = decode_from_latent(vae, z_t, skips=skips)  # [1, 3, H, W]
        preds.append(x_pred[0])

    originals = [x for x in xs_tp1]
    return originals, preds


# ========= Main =========

def main():
    parser = argparse.ArgumentParser(description="Test VAE+DiT sequential prediction and save a two row frame grid.")
    parser.add_argument("--data_root", type=str, default="small_mario_data",
                        help="Folder containing frames/ and actions.txt")
    parser.add_argument("--vae_ckpt", type=str, default="vae_best.pt",
                        help="Path to VAE checkpoint")
    parser.add_argument("--dit_ckpt", type=str, default="dit_model.pt",
                        help="Path to DiT checkpoint")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Image size used by dataset and VAE (square)")
    parser.add_argument("--seq_len", type=int, default=5,
                        help="Number of sequential transitions to visualise")
    parser.add_argument("--out_dir", type=str, default="predicted frames",
                        help="Directory to save the output image")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"Using device: {device}")

    # Dataset
    ds = MarioTransitionsDataset(root_dir=args.data_root, image_size=args.image_size)
    print(f"[Dataset] {len(ds)} transitions available")

    # VAE
    vae = VAE(latent_dim=LATENT_DIM_DEFAULT)
    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    if isinstance(vae_ckpt, dict) and "state_dict" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=False)
    elif isinstance(vae_ckpt, dict) and "model" in vae_ckpt and isinstance(vae_ckpt["model"], dict):
        vae.load_state_dict(vae_ckpt["model"], strict=False)
    else:
        vae.load_state_dict(vae_ckpt, strict=False)
    vae.to(device).eval()
    print(f"[VAE] Loaded weights from {args.vae_ckpt}")

    # Infer action dimension
    _, a_sample, _ = ds[0]
    action_dim = a_sample.numel()

    # DiT
    dit_ckpt = torch.load(args.dit_ckpt, map_location="cpu")
    if isinstance(dit_ckpt, dict) and "config" in dit_ckpt and isinstance(dit_ckpt["config"], dict):
        cfg = dict(dit_ckpt["config"])
        dit = LatentDiT(**cfg)
    else:
        dit = LatentDiT(
            latent_channels=0,
            grid_size=0,
            action_dim=action_dim,
            hidden_dim=512,
            depth=8,
            num_heads=8,
            dropout=0.0,
            is_vector=True,
            vector_dim=LATENT_DIM_DEFAULT,
        )
    if isinstance(dit_ckpt, dict) and "state_dict" in dit_ckpt:
        dit.load_state_dict(dit_ckpt["state_dict"], strict=False)
    elif isinstance(dit_ckpt, dict) and "model" in dit_ckpt and isinstance(dit_ckpt["model"], dict):
        dit.load_state_dict(dit_ckpt["model"], strict=False)
    else:
        dit.load_state_dict(dit_ckpt, strict=False)
    dit.to(device).eval()
    print(f"[DiT] Loaded weights from {args.dit_ckpt}")

    # Start index
    max_start = len(ds) - args.seq_len
    if max_start < 0:
        raise ValueError(f"seq_len={args.seq_len} is longer than available transitions ({len(ds)}).")
    start_idx = random.randint(0, max_start)

    # Roll out and save
    originals, preds = rollout_sequence(
        vae=vae,
        dit=dit,
        dataset=ds,
        start_idx=start_idx,
        seq_len=args.seq_len,
        device=device
    )

    out_path = save_two_row_grid(
        originals=originals,
        predictions=preds,
        out_dir=Path(args.out_dir),
        prefix=f"two_row_seq_len{args.seq_len}"
    )
    print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
    main()

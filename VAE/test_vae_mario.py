"""
test_vae_mario.py
-----------------
Evaluate a trained VAE by reconstructing 5 random Mario frames and saving
one page with two rows:
  - Row 1: five original images with labels
  - Row 2: the corresponding five reconstructed images with labels

Output:
- checkpoints/recons_two_row_panel.png
"""

import os
import random
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from load_mario_dataset import MarioFramesDataset
from vae_model import VAE, LATENT_DIM_DEFAULT


# ----------------------------
# Device selection
# ----------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Image helpers
# ----------------------------
def to_uint8_img(t: torch.Tensor) -> Image.Image:
    """Convert a tensor in [-1, 1] to a PIL image [0, 255]."""
    t = t.clamp(-1, 1)
    t = (t * 0.5 + 0.5)           # [-1,1] -> [0,1]
    t = (t * 255.0).round().byte()
    return TF.to_pil_image(t)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    """
    Robust text measurement across Pillow versions.
    Prefers textbbox; falls back to font.getbbox; then to a simple estimate.
    """
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l), (b - t)
    try:
        l, t, r, b = font.getbbox(text)
        return (r - l), (b - t)
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (max(1, 6 * len(text)), 12)


def add_title_above(img: Image.Image, text: str, title_h: int = None) -> Image.Image:
    """Return a new image with a white title bar above the given image."""
    if title_h is None:
        title_h = max(24, img.height // 10)
    canvas = Image.new("RGB", (img.width, img.height + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = measure_text(draw, text, font)
    x = max(0, (img.width - w) // 2)
    y = max(0, (title_h - h) // 2)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return canvas


def make_two_row_panel(orig_imgs: List[Image.Image],
                       recon_imgs: List[Image.Image],
                       labels: List[int],
                       out_path: str,
                       col_gap: int = 16,
                       row_gap: int = 24) -> None:
    """
    Create a single page with two rows of five columns:
      - top row originals with labels "Original idx"
      - bottom row reconstructions with labels "Recon idx"
    Saves to out_path.
    """
    assert len(orig_imgs) == len(recon_imgs) == len(labels)
    k = len(orig_imgs)
    assert k > 0

    # Build labelled tiles
    labelled_top = [add_title_above(o, f"Original {idx}") for o, idx in zip(orig_imgs, labels)]
    labelled_bot = [add_title_above(r, f"Recon {idx}") for r, idx in zip(recon_imgs, labels)]

    # Assume all tiles share same size
    tw, th = labelled_top[0].size

    # Canvas size: sum of column widths plus gaps; two rows plus row gap
    canvas_w = k * tw + (k - 1) * col_gap
    canvas_h = 2 * th + row_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # Paste top row
    x = 0
    for tile in labelled_top:
        canvas.paste(tile, (x, 0))
        x += tw + col_gap

    # Paste bottom row
    x = 0
    y = th + row_gap
    for tile in labelled_bot:
        canvas.paste(tile, (x, y))
        x += tw + col_gap

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


# ----------------------------
# Main
# ----------------------------
def main():
    device = pick_device()
    print(f"Testing on device: {device}")

    # Load model
    model = VAE(latent_dim=LATENT_DIM_DEFAULT).to(device)
    model.eval()
    ckpt_path = "checkpoints/vae_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)

    # Load dataset
    data_root = "/Users/phillipliu/Documents/UCT/Honours/Thesis/Code/capture_output"
    ds = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=True)
    assert len(ds) > 0, "Dataset appears to be empty."

    # Choose exactly 5 random indices
    k = min(5, len(ds))
    random.seed()  # set to a fixed integer for reproducible samples if needed
    indices = sorted(random.sample(range(len(ds)), k=k))

    # Build batch
    imgs = [ds[i][0] for i in indices]
    batch = torch.stack(imgs, dim=0).to(device)

    # Forward pass
    with torch.no_grad():
        recon, _, _ = model(batch)

    # Convert to PIL
    orig_imgs = [to_uint8_img(batch[i].cpu()) for i in range(k)]
    recon_imgs = [to_uint8_img(recon[i].cpu()) for i in range(k)]

    # Save one page with two rows and labels
    out_panel = "checkpoints/recons_two_row_panel.png"
    make_two_row_panel(orig_imgs, recon_imgs, indices, out_panel)
    print(f"Saved two row panel to {out_panel}")


if __name__ == "__main__":
    main()

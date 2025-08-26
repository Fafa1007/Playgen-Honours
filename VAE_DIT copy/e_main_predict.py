import argparse
import os
from typing import List, Tuple
import torch
from PIL import Image
import torchvision.utils as vutils

from a_load_mario_dataset import MarioTransitionsDataset
from b_vae_model import VAE, vae_encode_mu, vae_decode_no_skips
from d_dit_model import LatentDiT


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """
    img: [C,H,W] in [-1,1] -> uint8 [C,H,W] 0..255
    """
    x = (img.clamp(-1, 1) + 1.0) * 0.5
    x = (x * 255.0).round().to(torch.uint8)
    return x


@torch.no_grad()
def rollout_sequence(vae, dit, x0: torch.Tensor, actions: torch.Tensor, steps: int, device: torch.device) -> List[torch.Tensor]:
    """
    Returns list of predicted frames [T, 3, H, W] in [-1,1]
    """
    vae.eval(); dit.eval()
    z = vae_encode_mu(vae, x0.to(device))  # [1,C,H',W']
    preds = []
    for t in range(steps):
        a_t = actions[t: t + 1].to(device)  # [1, A]
        z = dit(z, a_t)                     # latent step
        x_pred = vae_decode_no_skips(vae, z)
        preds.append(x_pred[0].detach().cpu())
    return preds


def make_two_row_grid(originals: List[torch.Tensor], preds: List[torch.Tensor], out_path: str):
    """
    originals: list of [3,H,W] in [-1,1]
    preds: list of [3,H,W] in [-1,1]
    Saves a two row grid where first row are originals, second row are predictions.
    """
    assert len(preds) >= 1
    # Use the first original plus blanks if not provided per step
    first = to_uint8(originals[0])
    cols = min(5, len(preds))
    top = [first] + [to_uint8(originals[0]) for _ in range(cols - 1)]
    bottom = [to_uint8(p) for p in preds[:cols]]

    top_row = torch.stack(top, dim=0)   # [cols,3,H,W]
    bot_row = torch.stack(bottom, dim=0)

    grid = torch.cat([top_row, bot_row], dim=0)  # [2*cols,3,H,W]
    grid_img = vutils.make_grid(grid, nrow=cols, padding=2)
    Image.fromarray(grid_img.permute(1, 2, 0).numpy()).save(out_path)
    print(f"Saved grid to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="small_mario_data")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--vae_ckpt", type=str, default="vae_best.pt")
    parser.add_argument("--dit_ckpt", type=str, default="dit_model.pt")
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--latent_hw", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out_grid", type=str, default="predictions_grid.png")
    args = parser.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    # Data to get the start frame and actions
    ds = MarioTransitionsDataset(args.data_root, image_size=args.image_size)
    act_dim = ds.actions.shape[1]

    # Use the very first frame as x0, and the next T actions
    x0 = ds[0][0].unsqueeze(0)  # [1,3,H,W]
    actions = ds.actions[: args.steps]  # [T, A]

    # Models
    vae = VAE(image_size=args.image_size, z_ch=args.latent_channels)
    if os.path.isfile(args.vae_ckpt):
        print(f"Loading VAE checkpoint: {args.vae_ckpt}")
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"))
    vae.to(device).eval()

    dit = LatentDiT(
        latent_channels=args.latent_channels, act_dim=act_dim,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads, latent_hw=args.latent_hw
    )
    print(f"Loading DiT checkpoint: {args.dit_ckpt}")
    dit.load_state_dict(torch.load(args.dit_ckpt, map_location="cpu"))
    dit.to(device).eval()

    # Rollout
    preds = rollout_sequence(vae, dit, x0, actions, steps=args.steps, device=device)

    # Save a two row grid: originals row is x0 repeated, second row are predictions
    originals = [x0[0].cpu()]
    make_two_row_grid(originals, preds, args.out_grid)


if __name__ == "__main__":
    main()

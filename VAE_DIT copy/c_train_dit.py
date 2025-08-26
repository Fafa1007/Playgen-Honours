import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from a_load_mario_dataset import MarioTransitionsDataset
from b_vae_model import VAE, vae_encode_mu, vae_decode_no_skips
from d_dit_model import LatentDiT


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_actions(a: torch.Tensor) -> torch.Tensor:
    """
    If actions are already in {0,1}, map to [-1,1].
    If one hot, the same mapping still provides symmetry.
    """
    return a * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="small_mario_data")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vae_ckpt", type=str, default="vae_best.pt")
    parser.add_argument("--dit_ckpt_out", type=str, default="dit_model.pt")
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--latent_hw", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--unroll_steps", type=int, default=1)
    parser.add_argument("--lambda_img", type=float, default=0.25)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    # Data
    ds = MarioTransitionsDataset(args.data_root, image_size=args.image_size)
    act_dim = ds.actions.shape[1]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # VAE
    vae = VAE(image_size=args.image_size, z_ch=args.latent_channels)
    if os.path.isfile(args.vae_ckpt):
        print(f"Loading VAE checkpoint: {args.vae_ckpt}")
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"))
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # DiT
    dit = LatentDiT(
        latent_channels=args.latent_channels, act_dim=act_dim,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads, latent_hw=args.latent_hw
    ).to(device)

    optimiser = AdamW(dit.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        dit.train()
        running_lat, running_img = 0.0, 0.0

        for batch in loader:
            x_t, a_t, x_tp1 = [b.to(device) for b in batch]
            a_t = normalise_actions(a_t)

            with torch.no_grad():
                mu_t = vae_encode_mu(vae, x_t)
                mu_tp1 = vae_encode_mu(vae, x_tp1)

            z_t = mu_t
            total_latent_loss = 0.0
            total_image_loss = 0.0

            for k in range(args.unroll_steps):
                z_pred = dit(z_t, a_t)
                latent_loss = F.mse_loss(z_pred, mu_tp1)

                with torch.no_grad():
                    x_pred = vae_decode_no_skips(vae, z_pred)
                image_loss = F.mse_loss(x_pred, x_tp1)

                loss = latent_loss + args.lambda_img * image_loss

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
                optimiser.step()

                total_latent_loss += float(latent_loss.detach().cpu())
                total_image_loss += float(image_loss.detach().cpu())

                z_t = z_pred.detach()

            running_lat += total_latent_loss / args.unroll_steps
            running_img += total_image_loss / args.unroll_steps
            global_step += 1

        mean_lat = running_lat / len(loader)
        mean_img = running_img / len(loader)
        mean_total = mean_lat + args.lambda_img * mean_img
        print(f"Epoch {epoch:03d} | latent {mean_lat:.6f} | image {mean_img:.6f} | total {mean_total:.6f}")

        # save best
        if mean_total < best_loss:
            best_loss = mean_total
            torch.save(dit.state_dict(), args.dit_ckpt_out)
            print(f"Saved improved DiT to {args.dit_ckpt_out}")

    # final save
    torch.save(dit.state_dict(), args.dit_ckpt_out)
    print(f"Training complete. Final DiT saved to {args.dit_ckpt_out}")


if __name__ == "__main__":
    main()

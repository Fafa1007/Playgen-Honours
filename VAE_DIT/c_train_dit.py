
# c_train_dit.py (patched to handle vector latents, and non-CUDA pin_memory)
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from a_load_mario_dataset import MarioTransitionsDataset
from d_dit_model import LatentDiT
from b_vae_model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
use_cuda = device.type == "cuda"

DEFAULTS: Dict[str, float] = {
    "batch_size": 32,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "image_weight": 0.0,
    "num_workers": 4,
    "action_dim": 5,
    "hidden_dim": 512,
    "depth": 8,
    "num_heads": 8,
    "dropout": 0.0,
}

@torch.no_grad()
def try_get_mu_and_logvar_from(obj):
    # Tries to extract (mu, logvar) from various outputs
    if isinstance(obj, dict):
        mu = obj.get('mu')
        logvar = obj.get('logvar')
        if mu is None:
            # try fuzzy
            for k, v in obj.items():
                if 'mu' in k.lower():
                    mu = v; break
        if logvar is None:
            for k, v in obj.items():
                if 'log' in k.lower() and 'var' in k.lower():
                    logvar = v; break
        return mu, logvar
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            return obj[0], obj[1]
        if len(obj) == 1:
            return obj[0], None
    # fallback: treat as mu only
    return obj, None

@torch.no_grad()
def encode_to_stats(vae: VAE, x: torch.Tensor):
    if hasattr(vae, "encoder"):
        out = vae.encoder(x)
        mu, logvar = try_get_mu_and_logvar_from(out)
        if mu is not None:
            return mu, (logvar if logvar is not None else torch.zeros_like(mu))
    if hasattr(vae, "encode"):
        out = vae.encode(x)
        mu, logvar = try_get_mu_and_logvar_from(out)
        if mu is not None:
            return mu, (logvar if logvar is not None else torch.zeros_like(mu))
    # Try forward
    out = vae(x)
    mu, logvar = try_get_mu_and_logvar_from(out)
    if mu is not None:
        return mu, (logvar if logvar is not None else torch.zeros_like(mu))
    raise RuntimeError("Could not extract mu/logvar from VAE outputs.")

@torch.no_grad()
def decode_from_latent(vae: VAE, z: torch.Tensor):
    if hasattr(vae, "decoder"):
        return vae.decoder(z)
    return vae.decode(z)

def train_latent_dit(data_root: str, vae_ckpt_path: str, params: Dict = None):
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    ds = MarioTransitionsDataset(data_root, image_size=128)
    dl = DataLoader(ds, batch_size=p["batch_size"], shuffle=True, num_workers=p["num_workers"], pin_memory=use_cuda)

    # Load VAE and freeze
    vae = VAE(latent_dim=256).to(device)
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    vae.load_state_dict(state, strict=False)
    vae.eval()
    for par in vae.parameters():
        par.requires_grad = False

    # Peek a batch to infer latent shape
    x_t, a_t, x_tp1 = next(iter(dl))
    x_t = x_t.to(device)
    x_tp1 = x_tp1.to(device)
    with torch.no_grad():
        mu_t, _ = encode_to_stats(vae, x_t)      # could be [B,C,H,W] or [B,D]
        mu_tp1, _ = encode_to_stats(vae, x_tp1)

    if mu_t.ndim == 4:
        # spatial
        C, H, W = mu_t.shape[1:4]
        model = LatentDiT(latent_channels=C, grid_size=H, action_dim=p["action_dim"],
                          hidden_dim=p["hidden_dim"], depth=p["depth"], num_heads=p["num_heads"],
                          dropout=p["dropout"], is_vector=False).to(device)
    elif mu_t.ndim == 2:
        # vector
        D = mu_t.shape[1]
        model = LatentDiT(latent_channels=1, grid_size=1, action_dim=p["action_dim"],
                          hidden_dim=p["hidden_dim"], depth=p["depth"], num_heads=p["num_heads"],
                          dropout=p["dropout"], is_vector=True, vector_dim=D).to(device)
    else:
        raise ValueError(f"Unsupported latent shape {mu_t.shape}; expected [B,C,H,W] or [B,D].")

    opt = optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
    mse = nn.MSELoss()

    for epoch in range(p["epochs"]):
        model.train()
        total_latent_loss = 0.0
        total_image_loss = 0.0
        total = 0

        for x_t, a_t, x_tp1 in dl:
            x_t = x_t.to(device, non_blocking=use_cuda)
            a_t = a_t.to(device, non_blocking=use_cuda)
            x_tp1 = x_tp1.to(device, non_blocking=use_cuda)

            with torch.no_grad():
                mu_t, _ = encode_to_stats(vae, x_t)
                mu_tp1, _ = encode_to_stats(vae, x_tp1)

            pred_z_tp1 = model(mu_t, a_t)

            latent_loss = mse(pred_z_tp1, mu_tp1)
            image_loss = 0.0
            if p["image_weight"] > 0.0:
                with torch.no_grad():
                    pred_x_tp1 = decode_from_latent(vae, pred_z_tp1)
                    gt_x_tp1 = x_tp1
                image_loss = mse(pred_x_tp1, gt_x_tp1)

            loss = latent_loss + (p["image_weight"] * image_loss if isinstance(image_loss, (float,int)) else p["image_weight"] * image_loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_latent_loss += latent_loss.item() * x_t.size(0)
            total_image_loss += (image_loss if isinstance(image_loss, float) else image_loss.item()) * x_t.size(0)
            total += x_t.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{p['epochs']}: latent MSE {total_latent_loss/total:.6f} | image MSE {total_image_loss/total:.6f}")

    torch.save(model.state_dict(), os.path.join(data_root, "latent_dit.pt"))
    print(f"Saved DiT to {os.path.join(data_root, 'latent_dit.pt')}")

if __name__ == '__main__':
    train_latent_dit(
        data_root='small_mario_data',
        vae_ckpt_path='vae_best.pt',
        params=None
    )

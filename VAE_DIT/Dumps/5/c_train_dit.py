# c_train_dit.py
import os
import math
import random
from typing import Tuple
from d_dit_model import TokenDiT, ConvDynamics, ConvTransDynamics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from b_vae_model import VAE
from a_load_mario_dataset import MarioFramesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Hyperparameters
# --------------------------
H = {
    "latent_dim": 256,            # VAE latent size (vector)
    "grid_h": 16,                 # reshape 256 -> 1x16x16
    "grid_w": 16,
    "action_dim": 5,              # raw action size (one-hot or categorical->one-hot upstream)
    "act_emb_dim": 64,            # learnable action embedding dim
    "model_type": "hybrid",       # "token_dit" | "conv" | "hybrid"
    "hidden_dim": 256,            # channels for conv/hybrid blocks, or width for token DiT
    "depth": 4,                   # transformer depth (token & hybrid)
    "num_heads": 8,               # transformer heads (token & hybrid)
    "dropout": 0.1,
    "batch_size": 32,
    "epochs": 10,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 500,
    "max_steps": 50_000,          # for cosine schedule; will auto-clip to num_steps
    "grad_clip": 1.0,
    "lambda_contrast": 0.5,       # strength of wrong-action contrastive hinge loss
    "contrast_margin": 0.10,      # margin in latent-space MSE
    "lambda_img": 0.25,           # optional pixel-space loss via decoder
    "img_loss_every": 1,          # compute image loss every N steps (1 = every step)
    "action_dropout_p": 0.1,      # zero out action embedding with this prob
    "scheduled_sampling_p0": 0.0, # start prob of replacing z_t with (z_t + prev_pred)
    "scheduled_sampling_p1": 0.3, # end prob (linearly increases across training)
    "num_workers": 2,
    "seed": 42
}

torch.manual_seed(H["seed"])
random.seed(H["seed"])

# --------------------------
# Dataset that pre-encodes latents + stores images
# --------------------------
class MarioLatentDataset(Dataset):
    """
    Returns:
      z_t [D], a_t [action_dim], z_tp1 [D], skips_t (list of tensors), img_tp1 [3,128,128]
    """
    def __init__(self, data_root: str, vae: VAE, image_size: int = 128):
        self.vae = vae.eval()
        base = MarioFramesDataset(root_dir=data_root, image_size=image_size, return_actions=True)
        self.latents, self.skips, self.actions, self.images = [], [], [], []
        with torch.no_grad():
            for img, action in base:
                self.images.append(img)  # cpu tensor [3,128,128], in [-1,1] if your loader does that
                img = img.unsqueeze(0).to(device)
                mu, logvar, s = self.vae.encoder(img)
                self.latents.append(mu.squeeze(0).cpu())
                self.skips.append([t.squeeze(0).cpu() for t in s])
                self.actions.append(torch.tensor(action, dtype=torch.float32))

        # t -> t+1 pairs
        self.z_t = self.latents[:-1]
        self.a_t = self.actions[:-1]
        self.z_tp1 = self.latents[1:]
        self.skips_t = self.skips[:-1]
        self.img_tp1 = self.images[1:]

    def __len__(self): return len(self.z_t)

    def __getitem__(self, i):
        return (
            self.z_t[i],
            self.a_t[i],
            self.z_tp1[i],
            self.skips_t[i],
            self.img_tp1[i],
        )

# --------------------------
# Training helpers
# --------------------------
def cosine_warmup_lr(step, base_lr, warmup, max_steps):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))

def wrong_action_contrastive_loss(z_pred, z_tp1, z_pred_wrong, margin):
    """Hinge loss: max(0, m - d_wrong + d_right), d = MSE per-sample."""
    d_right = F.mse_loss(z_pred, z_tp1, reduction="none").mean(dim=1)
    d_wrong = F.mse_loss(z_pred_wrong, z_tp1, reduction="none").mean(dim=1)
    return torch.clamp(margin - d_wrong + d_right, min=0.0).mean()


# --------------------------
# Main train
# --------------------------
def train(data_root: str, vae_path: str, out_path: str = "dit_model.pt"):
    # VAE
    vae = VAE(latent_dim=H["latent_dim"]).to(device)
    ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(ckpt["model"], strict=True)
    vae.eval()

    # Data
    ds = MarioLatentDataset(data_root, vae, image_size=128)
    dl = DataLoader(ds, batch_size=H["batch_size"], shuffle=True, num_workers=H["num_workers"], pin_memory=True)

    # Model
    if H["model_type"] == "token_dit":
        model = TokenDiT(
            H["latent_dim"], H["action_dim"], H["act_emb_dim"],
            H["hidden_dim"], H["depth"], H["num_heads"], H["dropout"]
        )
    elif H["model_type"] == "conv":
        model = ConvDynamics(
            H["latent_dim"], H["grid_h"], H["grid_w"], H["action_dim"], H["act_emb_dim"],
            H["hidden_dim"], H["dropout"]
        )
    else:
        model = ConvTransDynamics(
            H["latent_dim"], H["grid_h"], H["grid_w"], H["action_dim"], H["act_emb_dim"],
            H["hidden_dim"], H["depth"], H["num_heads"], H["dropout"]
        )
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=H["lr"], weight_decay=H["weight_decay"])
    criterion = nn.MSELoss()

    step = 0
    num_steps_total = min(H["max_steps"], H["epochs"] * math.ceil(len(ds)/H["batch_size"]))
    print(f"[train] total steps (for scheduler): {num_steps_total}")

    for epoch in range(H["epochs"]):
        model.train()
        total_loss = 0.0

        # scheduled sampling prob this epoch (linear)
        p_sched = H["scheduled_sampling_p0"] + (H["scheduled_sampling_p1"] - H["scheduled_sampling_p0"]) * (epoch / max(1, H["epochs"] - 1))

        for z_t, a_t, z_tp1, skips_t, img_tp1 in dl:
            z_t   = z_t.to(device).float()
            z_tp1 = z_tp1.to(device).float()
            a_t   = a_t.to(device).float()
            img_tp1 = img_tp1.to(device).float()
            # scheduled sampling: sometimes shift z_t by a past self-prediction (single-step proxy)
            if p_sched > 0 and random.random() < p_sched:
                with torch.no_grad():
                    z_t = z_t + model(z_t, a_t, action_dropout_p=0.0)

            # forward (right action)
            pred_delta = model(z_t, a_t, action_dropout_p=H["action_dropout_p"])
            z_pred = z_t + pred_delta
            loss_lat = criterion(z_pred, z_tp1)  # predict absolute z_{t+1}

            # contrastive wrong-action loss
            idx_perm = torch.randperm(a_t.size(0), device=device)
            a_wrong = a_t[idx_perm]
            with torch.no_grad():
                z_pred_wrong = z_t + model(z_t, a_wrong, action_dropout_p=0.0)
            loss_contrast = wrong_action_contrastive_loss(z_pred, z_tp1, z_pred_wrong, margin=H["contrast_margin"])

            # optional image-space loss
            do_img = (H["lambda_img"] > 0) and (step % H["img_loss_every"] == 0)
            if do_img:
                with torch.no_grad():
                    # decode with skips from time t (your decoder expects skips from current frame)
                    recon_pred = vae.decoder(z_pred, skips=[s.to(device) for s in skips_t])
                # img_tp1 presumably in [-1,1]; decoder likely outputs [-1,1] too
                loss_img = F.mse_loss(recon_pred, img_tp1)
            else:
                loss_img = torch.tensor(0.0, device=device)

            loss = loss_lat + H["lambda_contrast"] * loss_contrast + H["lambda_img"] * loss_img

            # optimize
            lr = cosine_warmup_lr(step, H["lr"], H["warmup_steps"], num_steps_total)
            for pg in opt.param_groups: pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            opt.step()

            total_loss += loss.item() * z_t.size(0)
            step += 1

        avg = total_loss / len(ds)
        print(f"[epoch {epoch+1}/{H['epochs']}] loss={avg:.6f}  p_sched={p_sched:.3f}  lr={lr:.2e}")

    # save
    torch.save(model.state_dict(), out_path)
    print(f"Saved dynamics model to {out_path}")


if __name__ == "__main__":
    train(
        data_root="small_mario_data",
        vae_path="vae_best.pt",
        out_path={
            "token_dit": "token_dit.pt",
            "conv": "conv_dynamics.pt",
            "hybrid": "hybrid_convtrans.pt",
        }[H["model_type"]],
    )

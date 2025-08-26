# c_train_dit.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from b_vae_model import VAE, USE_SKIPS
import os
from a_load_mario_dataset import MarioFramesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Hyperparameters 
# --------------------------
HYPERPARAMS = {
    "latent_dim": 256,          # keep the same for VAE consistency
    "action_dim": 5,            # unchanged
    "hidden_dim": 1024,         # increase model capacity to capture finer details
    "batch_size": 16,           # smaller batch can help stabilize training on limited GPU memory
    "epochs": 50,               # longer training to improve latent predictions
    "learning_rate": 5e-4,      # slightly lower LR for stable convergence
    "weight_decay": 1e-5,       # reduce to avoid underfitting
    "scheduler_step": 10,       # slower decay
    "scheduler_gamma": 0.8,     # gentler decay to keep learning
    "scheduled_sampling_p": 0.25, # can keep or gradually increase over epochs
    "dropout": 0.1              # optional: add to DiT blocks to reduce overfitting and noise
}

# --------------------------
# Dataset for DiT with VAE latents
# --------------------------
class MarioLatentDataset(Dataset):
    """
    Produces pairs (z_t, action_t, skips_t) -> z_{t+1}.
    Skips are used for reconstruction but not predicted.
    """
    def __init__(self, data_root, vae: VAE):
        self.vae = vae.eval()
        self.dataset = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=True)
        self.latents, self.skips, self.actions = [], [], []

        print("[Dataset] Encoding frames into latent vectors with skips...")
        with torch.no_grad():
            for img, action in self.dataset:
                img = img.unsqueeze(0).to(device)
                mu, logvar, skips = vae.encoder(img)
                self.latents.append(mu.squeeze(0).cpu())
                self.skips.append([s.squeeze(0).cpu() for s in skips])
                self.actions.append(action)

        # Prepare t -> t+1 sequences
        self.latents_t = self.latents[:-1]
        self.latents_tp1 = self.latents[1:]
        self.skips_t = self.skips[:-1]
        self.actions = self.actions[:-1]  # action moving t -> t+1

    def __len__(self):
        return len(self.latents_t)

    def __getitem__(self, idx):
        return (self.latents_t[idx],
                torch.tensor(self.actions[idx], dtype=torch.float32),
                self.latents_tp1[idx],
                self.skips_t[idx])

# --------------------------
# Diffusion Transformer (DiT) block
# --------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


class DiffusionTransformer(nn.Module):
    """
    Minimal DiT: takes (z_t + action) as input sequence, predicts z_{t+1}.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=512, depth=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, action_t):
        # concat latent + action
        x = torch.cat([z_t, action_t], dim=-1).unsqueeze(1)  # shape [B, 1, D]
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.output_proj(x.squeeze(1))  # back to [B, latent_dim]
        return x

# --------------------------
# Training function
# --------------------------
def train_dit(data_root, vae_path, params):
    vae = VAE(latent_dim=params["latent_dim"]).to(device)
    ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(ckpt["model"], strict=True)
    vae.eval()

    dataset = MarioLatentDataset(data_root, vae)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = DiffusionTransformer(
        latent_dim=params["latent_dim"],
        action_dim=params["action_dim"],
        hidden_dim=params["hidden_dim"],
        depth=4,
        num_heads=8
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["scheduler_step"], gamma=params["scheduler_gamma"])
    criterion = nn.MSELoss()

    p = params.get("scheduled_sampling_p", 0.0)

    for epoch in range(params["epochs"]):
        total_loss = 0.0
        for z_t, a_t, z_tp1, _ in loader:
            z_t, a_t, z_tp1 = z_t.to(device), a_t.to(device), z_tp1.to(device)
            a_t = a_t.float()

            # Scheduled Sampling: randomly replace z_t with previous prediction
            if torch.rand(1).item() < p:
                with torch.no_grad():
                    z_t = z_t + model(z_t, a_t)  # feed predicted latent for next step

            # Delta target
            delta_target = z_tp1 - z_t
            pred_delta = model(z_t, a_t)
            loss = criterion(pred_delta, delta_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * z_t.size(0)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{params['epochs']}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6e}")

    torch.save(model.state_dict(), "dit_model.pt")
    print("DiT model saved to dit_model.pt")
    

if __name__ == "__main__":
    train_dit(
        data_root=r"small_mario_data",
        vae_path="vae_best.pt",
        params=HYPERPARAMS
    )

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
# Hyperparameters (easy to tweak per trial)
# --------------------------
HYPERPARAMS = {
    "latent_dim": 256,
    "action_dim": 4,
    "hidden_dim": 512,       
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,    
    "scheduler_step": 5,     
    "scheduler_gamma": 0.5,  
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
# Simple DiT model
# --------------------------
class SimpleDiT(nn.Module):
    """
    Predicts next latent given current latent + action.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, action_t):
        x = torch.cat([z_t, action_t], dim=-1)
        return self.fc(x)

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

    model = SimpleDiT(params["latent_dim"], params["action_dim"], params["hidden_dim"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["scheduler_step"], gamma=params["scheduler_gamma"])
    criterion = nn.MSELoss()

    for epoch in range(params["epochs"]):
        total_loss = 0.0
        for z_t, a_t, z_tp1, _ in loader:
            z_t, a_t, z_tp1 = z_t.to(device), a_t.to(device), z_tp1.to(device)
            a_t_oh = nn.functional.one_hot(a_t.long(), num_classes=params["action_dim"]).float()
            pred = model(z_t, a_t_oh)
            loss = criterion(pred, z_tp1)

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

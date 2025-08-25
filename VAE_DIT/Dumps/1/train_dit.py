# train_dit.py
# Train DiT conditioned on actions using latent frames from a pretrained VAE

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vae_model import VAE
from vae_dit import DiTConditioned
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Dataset: Frames -> VAE Latents =====
class MarioLatentDataset(Dataset):
    """Returns (current latent, next latent, action)"""
    def __init__(self, frames_dir, actions_txt, vae, transform):
        self.frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        with open(actions_txt) as f:
            self.actions = [list(map(int, line.strip().split(","))) for line in f]
        self.vae = vae
        self.transform = transform

        assert len(self.frame_paths) == len(self.actions), \
            f"Frames ({len(self.frame_paths)}) vs actions ({len(self.actions)}) mismatch"

    def __len__(self):
        return len(self.frame_paths) - 1  # predict next

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        img_next = Image.open(self.frame_paths[idx + 1]).convert("RGB")
        action = torch.tensor(self.actions[idx], dtype=torch.float32)

        x = self.transform(img).unsqueeze(0).to(device)
        x_next = self.transform(img_next).unsqueeze(0).to(device)

        # Encode frames to latent vectors (mu only)
        with torch.no_grad():
            _, mu, _ = self.vae(x)
            _, mu_next, _ = self.vae(x_next)

        return mu.squeeze(0).cpu(), mu_next.squeeze(0).cpu(), action

# ===== Training Loop =====
def train_dit():
    # --- Load pretrained VAE ---
    vae = VAE(latent_dim=128)  # must match latent_dim used when checkpoint was created
    vae.eval()                  # freeze during DiT training

    # Load checkpoint on CPU first
    checkpoint = torch.load("VAE/vae_best.pt", map_location="cpu")
    vae.load_state_dict(checkpoint["model"], strict=True)  # only load weights
    vae = vae.to(device)

    # --- Transformations ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([-1, -1, -1], [2, 2, 2])
    ])

    # --- Dataset & DataLoader ---
    dataset = MarioLatentDataset(
        frames_dir="Data Generation/Small Mario Data/frames",
        actions_txt="Data Generation/Small Mario Data/actions.txt",
        vae=vae,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- DiT model ---
    dit = DiTConditioned().to(device)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # --- Training ---
    for epoch in range(20):
        dit.train()
        total_loss = 0
        for z, z_next, a in dataloader:
            z, z_next, a = z.to(device), z_next.to(device), a.to(device)

            pred = dit(z, a)
            loss = loss_fn(pred, z_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # --- Save trained DiT ---
    os.makedirs("VAE_DiT", exist_ok=True)
    torch.save(dit.state_dict(), "VAE_DiT/dit_best.pt")
    print("Saved DiT checkpoint to VAE_DiT/dit_best.pt")

# ===== Main =====
if __name__ == "__main__":
    train_dit()

# test_vae_console.py
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from b_vae_model import VAE
from a_load_mario_dataset import MarioFramesDataset

# --------------------------
# Device
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------
# Load VAE
# --------------------------
vae = VAE(latent_dim=256).to(device)
vae.eval()

ckpt_path = r"vae_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
vae.load_state_dict(ckpt["model"], strict=True)
print("VAE checkpoint loaded successfully.")

# --------------------------
# Load dataset
# --------------------------
data_root = r"balanced_frames_200"  
dataset = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=False)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# --------------------------
# Take one batch
# --------------------------
images, _ = next(iter(loader))
images = images.to(device)

# --------------------------
# Forward pass
# --------------------------
with torch.no_grad():
    recon, mu, logvar_bounded = vae(images)

# --------------------------
# Print encoder outputs
# --------------------------
print("\nEncoder mu:")
print(mu)
print("\nEncoder logvar (bounded):")
print(logvar_bounded)

# --------------------------
# Sample latent vector
# --------------------------
std = torch.exp(0.5 * logvar_bounded)
eps = torch.randn_like(std)
z = mu + eps * std
print("\nSampled latent z:")
print(z)

# --------------------------
# Show original and reconstructed images
# --------------------------
images_np = images.cpu() * 0.5 + 0.5  # undo normalization [-1,1] -> [0,1]
recon_np = recon.cpu() * 0.5 + 0.5

fig, axes = plt.subplots(2, images.size(0), figsize=(12, 4))
for i in range(images.size(0)):
    # Original
    axes[0, i].imshow(images_np[i].permute(1, 2, 0))
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")
    
    # Reconstruction
    axes[1, i].imshow(recon_np[i].permute(1, 2, 0))
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstruction")

plt.tight_layout()
plt.show()

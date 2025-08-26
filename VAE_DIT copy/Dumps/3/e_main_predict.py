# e_main_predict.py
import torch
from b_vae_model import VAE, USE_SKIPS
from d_dit_model import SimpleDiT
from a_load_mario_dataset import MarioFramesDataset
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load VAE
# --------------------------
vae = VAE(latent_dim=256).to(device)
ckpt = torch.load("vae_best.pt", map_location=device)
vae.load_state_dict(ckpt["model"], strict=True)
vae.eval()

# --------------------------
# Load DiT
# --------------------------
dit = SimpleDiT().to(device)
dit.load_state_dict(torch.load("dit_model.pt", map_location=device))
dit.eval()

# --------------------------
# Load dataset
# --------------------------
data_root = r"small_mario_data"
dataset = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=True)

# --------------------------
# Output folder
# --------------------------
output_dir = r"predicted_frames\\frames"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Predict frames sequentially
# --------------------------
with torch.no_grad():
    # Start from first frame
    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)

    # Encode first frame
    mu, logvar, skips = vae.encoder(img)
    z_t = mu  # start latent

    for idx, action in enumerate(dataset.actions):
        a_t = torch.zeros(1, 4, device=device)
        a_t[0, action] = 1.0

        # Predict next latent
        z_tp1 = dit(z_t, a_t)

        # Decode using skips from current frame
        recon = vae.decoder(z_tp1, skips=skips)
        recon_img = (recon[0].cpu() * 0.5 + 0.5).clamp(0, 1)
        recon_img = (recon_img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(recon_img).save(os.path.join(output_dir, f"frame_{idx+1:07d}.png"))

        # Encode reconstructed frame for next step to get updated skips
        mu, logvar, skips = vae.encoder(recon)
        z_t = mu

print(f"Predicted frames saved in {output_dir}")

# --------------------------
# Make one comparison grid (2 rows x 4 columns)
# --------------------------
import random

pred_dir = os.path.join("predicted_frames", "frames")
gt_dir = os.path.join("small_mario_data", "frames")
comp_path = os.path.join("predicted_frames", "comparison_grid.png")

# Pick 4 random predicted frames
pred_files = sorted(os.listdir(pred_dir))
sampled = random.sample(pred_files, 4)

gt_imgs, pred_imgs = [], []
for fname in sampled:
    pred_imgs.append(Image.open(os.path.join(pred_dir, fname)))
    gt_img = Image.open(os.path.join(gt_dir, fname)).resize((128, 128), Image.BILINEAR)
    gt_imgs.append(gt_img)

# Assume all frames are 128x128 now
w, h = 128, 128

# Create final grid: width = 4 * w, height = 2 * h
final_img = Image.new("RGB", (4 * w, 2 * h))

# Paste ground truth on first row
for i, gt in enumerate(gt_imgs):
    final_img.paste(gt, (i * w, 0))

# Paste predictions on second row
for i, pred in enumerate(pred_imgs):
    final_img.paste(pred, (i * w, h))

final_img.save(comp_path)
print(f"Saved comparison grid at {comp_path}")
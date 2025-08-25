# Load pretrained VAE, a DiT for latent prediction, and run inference on Mario frames with actions.

import torch
import torch.nn as nn
from torchvision import transforms
from vae_model import VAE
from PIL import Image
import json

# ===== Load pretrained VAE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = VAE(latent_dim=256).to(device)
vae.load_state_dict(torch.load("VAE/vae_best.pt", map_location=device))
vae.eval()

# ===== Simple DiT stub for conditioning on actions =====
class DiTConditioned(nn.Module):
    def __init__(self, latent_dim=256, action_dim=5, hidden_dim=512):
        super().__init__()
        # Simple MLP: input = latent + action
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # predict next latent
        )

    def forward(self, z, action):
        # z: [B, latent_dim], action: [B, action_dim]
        x = torch.cat([z, action], dim=-1)
        return self.net(x)

# Load trained DiT
dit = DiTConditioned().to(device)
dit.load_state_dict(torch.load("DiT/dit_best.pt", map_location=device))
dit.eval()

# ===== Preprocessing for frames =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([-1, -1, -1], [2, 2, 2])  # if training used [-1,1] scaling
])

# ===== Inference function =====
def predict_next_frame(frame_path, action):
    # Load + preprocess frame
    img = Image.open(frame_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # Encode to latent
    with torch.no_grad():
        recon, mu, logvar = vae(x)
        z = mu  # use mean as deterministic latent

    # Action conditioning
    action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        z_next = dit(z, action_tensor)

    # Decode to next frame
    with torch.no_grad():
        next_frame = vae.decoder(z_next)

    return next_frame  # tensor in [-1,1]

# Example usage:
# action = [0,1,0,1,0]   # left, right, down, up, speed
# next_frame = predict_next_frame("Data/frame_0000001.png", action)
# torchvision.utils.save_image(next_frame, "pred_next.png", normalize=True)

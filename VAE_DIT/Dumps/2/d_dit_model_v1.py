# dit_model.py
# Not a real DiT, just a simple MLP for demonstration
import torch
import torch.nn as nn

class SimpleDiT(nn.Module):
    """
    Predicts next latent given current latent + action.
    """
    def __init__(self, latent_dim=256, action_dim=5, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, action_t):
        x = torch.cat([z_t, action_t], dim=-1)
        return self.fc(x)

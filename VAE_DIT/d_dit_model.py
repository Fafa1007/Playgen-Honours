# dit_model.py
import torch
import torch.nn as nn

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
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


class DiffusionTransformer(nn.Module):
    """
    Minimal DiT: takes (z_t + action) as input sequence, predicts z_{t+1}.
    """
    def __init__(self, latent_dim=256, action_dim=5, hidden_dim=512, depth=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, action_t):
        x = torch.cat([z_t, action_t], dim=-1).unsqueeze(1)  # [B, 1, D]
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.output_proj(x.squeeze(1))
        return x

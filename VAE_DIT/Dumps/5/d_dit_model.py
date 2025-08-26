# d_dit_model.py
import torch
import torch.nn as nn

# --------------------------
# Utils: vector <-> grid
# --------------------------
class VecGridAdapter(nn.Module):
    """Converts 256-d vector <-> 1x16x16 grid to enable convs without changing your VAE."""
    def __init__(self, latent_dim=256, h=16, w=16):
        super().__init__()
        assert h * w == latent_dim, "grid_h*grid_w must equal latent_dim"
        self.h, self.w = h, w

    def vec_to_grid(self, z):  # [B, D] -> [B, 1, H, W]
        B, D = z.shape
        return z.view(B, 1, self.h, self.w)

    def grid_to_vec(self, g):  # [B, 1, H, W] -> [B, D]
        B = g.shape[0]
        return g.view(B, -1)


# --------------------------
# Transformer block (token & hybrid)
# --------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


# --------------------------
# Model A: Token DiT
# --------------------------
class TokenDiT(nn.Module):
    def __init__(self, latent_dim, action_dim, act_emb_dim, hidden_dim, depth, num_heads, dropout):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, act_emb_dim)
        self.action_proj = nn.Linear(act_emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_vec, a_onehot, action_dropout_p=0.0):
        B = z_vec.size(0)
        # Action embedding + dropout
        a_emb = self.action_emb(a_onehot)
        if self.training and action_dropout_p > 0:
            mask = (torch.rand(B, 1, device=z_vec.device) > action_dropout_p).float()
            a_emb = a_emb * mask
        z_tok = self.latent_proj(z_vec).unsqueeze(1)  # [B,1,H]
        a_tok = self.action_proj(a_emb).unsqueeze(1)  # [B,1,H]
        x = torch.cat([z_tok, a_tok], dim=1)          # [B,2,H]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])                        # latent token output
        return self.out(x)                            # Δz


# --------------------------
# Model B: ConvDynamics
# --------------------------
class ConvDynamics(nn.Module):
    def __init__(self, latent_dim, grid_h, grid_w, action_dim, act_emb_dim, hidden_dim, dropout):
        super().__init__()
        self.adapter = VecGridAdapter(latent_dim, grid_h, grid_w)
        self.action_emb = nn.Linear(action_dim, act_emb_dim)
        self.film_gamma = nn.Linear(act_emb_dim, hidden_dim)
        self.film_beta  = nn.Linear(act_emb_dim, hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.out = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, z_vec, a_onehot, action_dropout_p=0.0):
        B = z_vec.size(0)
        g = self.adapter.vec_to_grid(z_vec)  # [B,1,H,W]
        x = self.stem(g)

        a_emb = self.action_emb(a_onehot)
        if self.training and action_dropout_p > 0:
            mask = (torch.rand(B, 1, device=z_vec.device) > action_dropout_p).float()
            a_emb = a_emb * mask

        gamma = self.film_gamma(a_emb).unsqueeze(-1).unsqueeze(-1)
        beta  = self.film_beta(a_emb).unsqueeze(-1).unsqueeze(-1)
        x = (1 + gamma) * x + beta

        x = x + self.body(x)
        delta_grid = self.out(x)
        return self.adapter.grid_to_vec(delta_grid)  # Δz


# --------------------------
# Model C: ConvTransDynamics (hybrid)
# --------------------------
class ConvTransDynamics(nn.Module):
    def __init__(self, latent_dim, grid_h, grid_w, action_dim, act_emb_dim, hidden_dim, depth, num_heads, dropout):
        super().__init__()
        self.adapter = VecGridAdapter(latent_dim, grid_h, grid_w)
        self.action_emb = nn.Linear(action_dim, act_emb_dim)
        self.film_gamma = nn.Linear(act_emb_dim, hidden_dim)
        self.film_beta  = nn.Linear(act_emb_dim, hidden_dim)

        self.conv_in = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, z_vec, a_onehot, action_dropout_p=0.0):
        B = z_vec.size(0)
        g = self.adapter.vec_to_grid(z_vec)           # [B,1,H,W]
        x = self.conv_in(g)                           # [B,C,H,W]

        # FiLM by action
        a_emb = self.action_emb(a_onehot)
        if self.training and action_dropout_p > 0:
            mask = (torch.rand(B, 1, device=z_vec.device) > action_dropout_p).float()
            a_emb = a_emb * mask
        gamma = self.film_gamma(a_emb).unsqueeze(-1).unsqueeze(-1)
        beta  = self.film_beta(a_emb).unsqueeze(-1).unsqueeze(-1)
        x = (1 + gamma) * x + beta

        # Flatten to sequence [B, HW, C]
        B, C, Hh, Ww = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x_seq = blk(x_seq)
        x_seq = self.norm(x_seq)

        # Back to grid
        x = x_seq.transpose(1, 2).view(B, C, Hh, Ww)
        delta_grid = self.conv_out(x)
        return self.adapter.grid_to_vec(delta_grid)  # Δz

def load_dynamics(H):
    """Returns the dynamics model according to model_type."""
    model_type = H.get("model_type", "hybrid")
    if model_type == "token_dit":
        from d_dit_model import TokenDiT
        return TokenDiT(
            latent_dim=H["latent_dim"],
            action_dim=H["action_dim"],
            act_emb_dim=64,           # same as training
            hidden_dim=H["hidden_dim"],
            depth=H["depth"],
            num_heads=H["num_heads"],
            dropout=0.1               # same as training
        )
    elif model_type == "conv":
        from d_dit_model import ConvDynamics
        return ConvDynamics(
            latent_dim=H["latent_dim"],
            grid_h=16, grid_w=16,     # same as training
            action_dim=H["action_dim"],
            act_emb_dim=64,
            hidden_dim=H["hidden_dim"],
            dropout=0.1
        )
    else:  # hybrid / ConvTransDynamics
        from d_dit_model import ConvTransDynamics
        return ConvTransDynamics(
            latent_dim=H["latent_dim"],
            grid_h=16, grid_w=16,
            action_dim=H["action_dim"],
            act_emb_dim=64,
            hidden_dim=H["hidden_dim"],
            depth=H["depth"],
            num_heads=H["num_heads"],
            dropout=0.1
        )
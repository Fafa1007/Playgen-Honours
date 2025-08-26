
# d_dit_model.py (patched for vector or spatial latents)
import math
import torch
import torch.nn as nn

def sinusoidal_position_embeddings(n_positions: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [N, dim]

class FiLM(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], cond: [B, C]
        gamma_beta = self.fc(cond)  # [B, 2D]
        B, N, D = x.shape
        gamma, beta = gamma_beta[:, :D], gamma_beta[:, D:]
        gamma = gamma.unsqueeze(1).expand(B, N, D)
        beta = beta.unsqueeze(1).expand(B, N, D)
        return x * (1 + gamma) + beta

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.film1 = FiLM(dim, cond_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.film2 = FiLM(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.film1(h, cond)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h2 = self.norm2(x)
        h2 = self.film2(h2, cond)
        h2 = self.mlp(h2)
        x = x + h2
        return x

class LatentDiT(nn.Module):
    """
    Supports both spatial latents [B, C, H, W] and vector latents [B, D].
    For vectors, we treat each channel as one token (sequence length = D).
    """
    def __init__(self, latent_channels: int, grid_size: int, action_dim: int, hidden_dim: int = 512,
                 depth: int = 8, num_heads: int = 8, dropout: float = 0.0, is_vector: bool = False, vector_dim: int = None):
        super().__init__()
        self.is_vector = is_vector
        self.action_dim = action_dim
        self.Dhid = hidden_dim

        if not is_vector:
            # spatial
            self.C = latent_channels
            self.H = grid_size
            self.W = grid_size
            self.N = self.H * self.W
            in_feat = self.C
            out_feat = self.C
        else:
            # vector
            assert vector_dim is not None, "vector_dim must be provided for vector latents"
            self.vec_dim = vector_dim
            self.N = self.vec_dim  # one token per latent channel
            in_feat = 1            # each token has scalar value
            out_feat = 1

        self.in_proj = nn.Linear(in_feat, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_feat)

        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        pe = sinusoidal_position_embeddings(self.N, hidden_dim)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, cond_dim=hidden_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def tokens_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.is_vector:
            # z: [B, C, H, W] -> [B, N, C]
            B, C, H, W = z.shape
            tokens = z.view(B, C, H * W).permute(0, 2, 1).contiguous()
        else:
            # z: [B, D] -> [B, N=D, 1]
            tokens = z.unsqueeze(-1)
        return tokens

    def latent_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.is_vector:
            # tokens: [B, N, C] -> [B, C, H, W]
            B, N, C = tokens.shape
            H = W = int(self.N ** 0.5)
            x = tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)
        else:
            # tokens: [B, N=D, 1] -> [B, D]
            x = tokens.squeeze(-1)
        return x

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens_from_latent(z_t)       # [B, N, in_feat]
        x = self.in_proj(tokens)                    # [B, N, Dhid]
        x = x + self.pos_embed
        cond = self.action_enc(a_t)
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.norm(x)
        delta_tokens = self.out_proj(x)             # [B, N, out_feat]
        delta = self.latent_from_tokens(delta_tokens)
        z_tp1 = z_t + self.scale * delta
        return z_tp1

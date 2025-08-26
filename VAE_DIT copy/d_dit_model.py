import torch
import torch.nn as nn


class ActionFiLM(nn.Module):
    def __init__(self, act_dim: int, embed_dim: int):
        super().__init__()
        self.to_scale = nn.Sequential(
            nn.Linear(act_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.to_shift = nn.Sequential(
            nn.Linear(act_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, tokens: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D], a: [B, A]
        gamma = torch.tanh(self.to_scale(a)).unsqueeze(1)  # [B,1,D]
        beta = self.to_shift(a).unsqueeze(1)               # [B,1,D]
        return tokens * (1.0 + gamma) + beta


class LatentDiT(nn.Module):
    """
    Predicts z_{t+1} from z_t and action a_t using a transformer encoder over spatial tokens.
    Outputs z_t + learned_scale * delta.
    """
    def __init__(
        self,
        latent_channels: int,
        act_dim: int,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        latent_hw: int = 16,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw

        self.in_proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.randn(1, latent_hw * latent_hw, embed_dim) / (embed_dim ** 0.5))
        self.film = ActionFiLM(act_dim, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.out_proj = nn.Conv2d(embed_dim, latent_channels, kernel_size=1)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_t.shape
        assert H * W == self.latent_hw * self.latent_hw, "latent spatial size must match model configuration"

        x = self.in_proj(z_t)                      # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)           # [B, N, D]
        x = x + self.pos_embed[:, : x.size(1), :]  # add positions

        x = self.film(x, a_t)                      # modulate by actions
        x = self.encoder(x)

        x = x.transpose(1, 2).reshape(B, -1, H, W) # [B, D, H, W]
        delta = self.out_proj(x)                   # [B, C, H, W]
        z_tp1 = z_t + self.delta_scale * delta
        return z_tp1

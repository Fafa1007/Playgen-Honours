# vae_model.py
# Variational Autoencoder for 128x128 RGB frames.
# Uses a 16x16 spatial bottleneck with a decoder that refines features after upsampling.
# Includes optional skip connections for sharper detail.
#
# New in this version:
# - We map the raw log-variance to a safe range [-6, 2] with a smooth tanh transform
#   to avoid the variance explosion that can cause the KL to blow up when it switches on.

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== Configuration switches ======
USE_SKIPS = True              # If True, pass encoder feature maps to the decoder
LATENT_DIM_DEFAULT = 256      # Latent dimension. 256 gives more capacity than 128

# Bounds used to stabilise log-variance
LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    A small refinement block with two 3x3 convolutions, GroupNorm and ReLU.
    This helps the model sharpen edges after each upsample step.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Encoder that maps an image to the mean and log variance of q(z|x)."""
    def __init__(self, latent_dim: int = LATENT_DIM_DEFAULT):
        super().__init__()

        # 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)

        # 64 -> 32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 128)

        # 32 -> 16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, 256)

        # Extra refinement at 16x16
        self.conv3_refine = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.gn3_refine = nn.GroupNorm(8, 256)

        # Heads for latent stats
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x: torch.Tensor):
        # Downsample with three stages
        h1 = F.relu(self.gn1(self.conv1(x)))          # [B, 64, 64, 64]
        h2 = F.relu(self.gn2(self.conv2(h1)))         # [B, 128, 32, 32]
        h3 = F.relu(self.gn3(self.conv3(h2)))         # [B, 256, 16, 16]
        h3 = F.relu(self.gn3_refine(self.conv3_refine(h3)))  # refine at 16x16

        # Map to latent statistics
        h_flat = self.flatten(h3)
        mu = self.fc_mu(h_flat)
        logvar_raw = self.fc_logvar(h_flat)           # unbounded log variance

        if USE_SKIPS:
            # Return the features for optional skip connections
            return mu, logvar_raw, (h1, h2, h3)
        else:
            return mu, logvar_raw


class Decoder(nn.Module):
    """Decoder that maps a latent vector back to an RGB frame."""
    def __init__(self, latent_dim: int = LATENT_DIM_DEFAULT):
        super().__init__()

        # Expand the latent vector to a 16x16 feature map
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)

        # 16 -> 32 at coarse scale with a transposed convolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        # 32 -> 64 via bilinear upsample and refinement
        up2_in_ch = 128 + (128 if USE_SKIPS else 0)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(up2_in_ch, 64),
        )

        # Further refinement at 64x64, then the final upsample to 128
        up3_core_in_ch = 64 + (64 if USE_SKIPS else 0)
        self.up3_core = conv_block(up3_core_in_ch, 32)
        self.up3_tail = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # outputs in [-1, 1] to match input scaling
        )

    def forward(self, z: torch.Tensor, skips: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        h = self.fc(z)
        h = h.view(-1, 256, 16, 16)

        # 16 -> 32
        h = self.up1(h)

        # Concatenate 32x32 skip if provided
        if USE_SKIPS and skips is not None:
            h1, h2, h3 = skips
            h = torch.cat([h, h2], dim=1)

        # 32 -> 64 with refinement
        h = self.up2(h)

        # Concatenate 64x64 skip if provided
        if USE_SKIPS and skips is not None:
            h = torch.cat([h, h1], dim=1)

        # Refine at 64x64 and upsample to the final size
        h = self.up3_core(h)
        x_rec = self.up3_tail(h)
        return x_rec


class VAE(nn.Module):
    """VAE wrapper that ties encoder and decoder and applies the reparameterisation trick."""
    def __init__(self, latent_dim: int = LATENT_DIM_DEFAULT):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

        # Careful weight initialisation for stable training
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar_bounded: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) with the usual reparameterisation trick."""
        std = torch.exp(0.5 * logvar_bounded)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Get latent parameters and optional skips
        if USE_SKIPS:
            mu, logvar_raw, skips = self.encoder(x)
        else:
            mu, logvar_raw = self.encoder(x)
            skips = None

        # Smoothly bound log-variance to a safe range to avoid KL explosions
        # Map raw values through tanh to [-1, 1], then scale to [LOGVAR_MIN, LOGVAR_MAX]
        logvar_bounded = 0.5 * (LOGVAR_MAX - LOGVAR_MIN) * torch.tanh(logvar_raw) + 0.5 * (LOGVAR_MAX + LOGVAR_MIN)

        # Sample latent and decode
        z = self.reparameterize(mu, logvar_bounded)
        recon = self.decoder(z, skips=skips)

        # Return reconstructions and the bounded logvar used for loss
        return recon, mu, logvar_bounded

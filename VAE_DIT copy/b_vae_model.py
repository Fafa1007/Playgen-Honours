import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ===== A compact convolutional VAE (keep or replace with your own) =====
class Encoder(nn.Module):
    def __init__(self, in_ch=3, z_ch=128, hw=16):
        super().__init__()
        self.hw = hw
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.ReLU(inplace=True),   # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),     # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),    # 32 -> 16
            nn.Conv2d(256, z_ch, 3, 1, 1)
        )
        # simple diagonal gaussian
        self.mu = nn.Conv2d(z_ch, z_ch, 1)
        self.logvar = nn.Conv2d(z_ch, z_ch, 1)

    def forward(self, x):
        h = self.conv(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar  # keep signature familiar


class Decoder(nn.Module):
    def __init__(self, z_ch=128, out_ch=3, use_skips: bool = False):
        super().__init__()
        self.use_skips = use_skips  # only a flag; skips are ignored unless provided
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(z_ch, 256, 4, 2, 1), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),   # 32 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),    # 64 -> 128
            nn.Conv2d(64, out_ch, 3, 1, 1),
            nn.Tanh(),  # map to [-1,1]
        )

    def forward(self, z, skips: Optional[object] = None):
        # Ignoring skips during prediction is the default behaviour
        return self.deconv(z)


class VAE(nn.Module):
    def __init__(self, image_size=128, z_ch=128):
        super().__init__()
        assert image_size % 8 == 0, "image_size must be divisible by 8"
        hw = image_size // 8
        self.encoder = Encoder(in_ch=3, z_ch=z_ch, hw=hw)
        self.decoder = Decoder(z_ch=z_ch, out_ch=3, use_skips=False)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_rec = self.decoder(z, skips=None)
        return x_rec, mu, logvar


# ===== Helpers to make any VAE usable for sequence prediction =====

def _has_kwarg(fn, name: str) -> bool:
    import inspect
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False


@torch.no_grad()
def vae_encode_mu(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Returns the encoder mean regardless of exact encoder return type.
    Works with (mu, logvar) or custom tuples.
    """
    vae.eval()
    out = vae.encoder(x)
    if isinstance(out, (tuple, list)):
        mu = out[0]
    else:
        mu = out
    return mu


@torch.no_grad()
def vae_decode_no_skips(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Decodes a latent without any encoder skip connections.
    If the decoder supports a 'skips' kwarg, passes skips=None.
    """
    vae.eval()
    dec = getattr(vae, "decoder", None)
    if dec is None:
        raise RuntimeError("Provided VAE has no .decoder")
    if _has_kwarg(dec.forward, "skips"):
        return dec(z, skips=None)
    return dec(z)

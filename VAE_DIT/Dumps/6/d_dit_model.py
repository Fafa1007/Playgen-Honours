# d_dit_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
# Enhanced Multi-Hot Action Encoder
# --------------------------
class MultiHotActionEncoder(nn.Module):
    """Enhanced encoder for multi-hot action vectors with better combination modeling."""
    def __init__(self, action_dim, act_emb_dim):
        super().__init__()
        self.action_dim = action_dim
        self.act_emb_dim = act_emb_dim
        
        # Individual button embeddings
        self.button_embeddings = nn.ModuleList([
            nn.Linear(1, act_emb_dim // 2) for _ in range(action_dim)
        ])
        
        # Combination modeling via cross-button attention
        self.combination_attn = nn.MultiheadAttention(
            embed_dim=act_emb_dim // 2, 
            num_heads=4, 
            batch_first=True
        )
        
        # Final projection to full embedding dimension
        self.final_proj = nn.Sequential(
            nn.Linear(act_emb_dim // 2, act_emb_dim),
            nn.ReLU(),
            nn.Linear(act_emb_dim, act_emb_dim)
        )
        
        # Learnable positional encoding for buttons
        self.button_pos_embed = nn.Parameter(torch.randn(action_dim, act_emb_dim // 2))

    def forward(self, action_vec):
        B = action_vec.size(0)
        
        # Embed each button individually
        button_embeds = []
        for i, button_emb in enumerate(self.button_embeddings):
            button_val = action_vec[:, i:i+1]  # [B, 1]
            embed = button_emb(button_val)     # [B, act_emb_dim//2]
            button_embeds.append(embed)
        
        # Stack and add positional encoding
        button_embeds = torch.stack(button_embeds, dim=1)  # [B, action_dim, act_emb_dim//2]
        button_embeds = button_embeds + self.button_pos_embed.unsqueeze(0)
        
        # Cross-button attention to model combinations
        attended, _ = self.combination_attn(button_embeds, button_embeds, button_embeds)
        
        # Aggregate across buttons (weighted sum by action values)
        action_weights = action_vec.unsqueeze(-1)  # [B, action_dim, 1]
        weighted_embeds = attended * action_weights  # [B, action_dim, act_emb_dim//2]
        aggregated = weighted_embeds.sum(dim=1)      # [B, act_emb_dim//2]
        
        # Final projection
        action_embed = self.final_proj(aggregated)   # [B, act_emb_dim]
        
        return action_embed

# --------------------------
# Spatial Position Encoding
# --------------------------
class SpatialPositionEncoding(nn.Module):
    """2D sinusoidal position encoding for spatial attention."""
    def __init__(self, dim, height, width):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        
        # Create position encodings
        pe = torch.zeros(height * width, dim)
        position = torch.arange(0, height * width).unsqueeze(1).float()
        
        # Create 2D positions
        y_pos = (torch.arange(height * width) // width).float()
        x_pos = (torch.arange(height * width) % width).float()
        
        # Sinusoidal encoding
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        pe[:, 0::4] = torch.sin(y_pos.unsqueeze(1) * div_term[:dim//4])
        pe[:, 1::4] = torch.cos(y_pos.unsqueeze(1) * div_term[:dim//4])
        pe[:, 2::4] = torch.sin(x_pos.unsqueeze(1) * div_term[:dim//4])
        pe[:, 3::4] = torch.cos(x_pos.unsqueeze(1) * div_term[:dim//4])
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, H*W, dim]

    def forward(self, x):
        # x: [B, H*W, dim]
        return x + self.pe

# --------------------------
# Cross-Attention Block (Spatial <-> Action)
# --------------------------
class CrossAttentionBlock(nn.Module):
    """Cross-attention between spatial features and action embeddings."""
    def __init__(self, dim, action_dim, num_heads, dropout=0.0):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.action_norm = nn.LayerNorm(action_dim)
        
        # Cross-attention: spatial queries, action keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Action projection to match spatial dimension
        self.action_proj = nn.Linear(action_dim, dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, spatial_tokens, action_embed):
        # spatial_tokens: [B, H*W, dim]
        # action_embed: [B, action_dim]
        
        B = spatial_tokens.size(0)
        
        # Normalize inputs
        spatial_q = self.spatial_norm(spatial_tokens)
        action_normalized = self.action_norm(action_embed)
        
        # Project action to spatial dimension and expand to sequence
        action_kv = self.action_proj(action_normalized).unsqueeze(1)  # [B, 1, dim]
        
        # Cross-attention
        attended, _ = self.cross_attn(spatial_q, action_kv, action_kv)
        spatial_tokens = spatial_tokens + attended
        
        # Feed-forward
        ff_input = self.ff_norm(spatial_tokens)
        spatial_tokens = spatial_tokens + self.ff(ff_input)
        
        return spatial_tokens

# --------------------------
# Linear Attention (Efficient Self-Attention)
# --------------------------
class LinearAttention(nn.Module):
    """Linear attention for efficient processing of spatial tokens."""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Linear attention: softmax over feature dimension instead of spatial
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        
        # Compute attention
        context = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, num_heads, head_dim, head_dim]
        out = torch.einsum('bhnd,bhde->bhne', q, context)  # [B, num_heads, N, head_dim]
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

# --------------------------
# Enhanced DiT Block with Linear Attention
# --------------------------
class EnhancedDiTBlock(nn.Module):
    """Enhanced transformer block with linear attention and better normalization."""
    def __init__(self, dim, num_heads, dropout=0.0, use_linear_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        if use_linear_attn:
            self.attn = LinearAttention(dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, 
                num_heads=num_heads, 
                batch_first=True, 
                dropout=dropout
            )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.use_linear_attn = use_linear_attn

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        if self.use_linear_attn:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h)
        x = x + h
        
        # Feed-forward
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h

# --------------------------
# Enhanced FiLM Conditioning
# --------------------------
class EnhancedFiLMConditioning(nn.Module):
    """Enhanced FiLM conditioning with multi-scale action influence."""
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        
        # Multi-scale conditioning
        self.global_gamma = nn.Linear(action_dim, hidden_dim)
        self.global_beta = nn.Linear(action_dim, hidden_dim)
        
        # Channel-wise conditioning
        self.channel_gamma = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        self.channel_beta = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Spatial conditioning (learnable spatial basis)
        self.spatial_basis = nn.Parameter(torch.randn(hidden_dim, 16, 16))
        self.spatial_weight = nn.Linear(action_dim, 1)

    def forward(self, x, action_embed):
        # x: [B, C, H, W]
        # action_embed: [B, action_dim]
        
        B, C, H, W = x.shape
        
        # Global conditioning
        global_gamma = self.global_gamma(action_embed).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        global_beta = self.global_beta(action_embed).unsqueeze(-1).unsqueeze(-1)
        
        # Channel-wise conditioning
        channel_gamma = self.channel_gamma(action_embed).unsqueeze(-1).unsqueeze(-1)
        channel_beta = self.channel_beta(action_embed).unsqueeze(-1).unsqueeze(-1)
        
        # Spatial conditioning
        spatial_weight = torch.sigmoid(self.spatial_weight(action_embed))  # [B, 1]
        spatial_mod = spatial_weight.unsqueeze(-1).unsqueeze(-1) * self.spatial_basis.unsqueeze(0)  # [B, C, H, W]
        
        # Combine all conditioning
        gamma = global_gamma + channel_gamma
        beta = global_beta + channel_beta + spatial_mod
        
        return (1 + gamma) * x + beta

# --------------------------
# Model: Enhanced ConvTransDynamics 
# --------------------------
class ConvTransDynamics(nn.Module):
    def __init__(self, latent_dim, grid_h, grid_w, action_dim, act_emb_dim, hidden_dim, depth, num_heads, dropout):
        super().__init__()
        self.adapter = VecGridAdapter(latent_dim, grid_h, grid_w)
        self.grid_h, self.grid_w = grid_h, grid_w
        
        # Enhanced multi-hot action encoder
        self.action_encoder = MultiHotActionEncoder(action_dim, act_emb_dim)
        
        # Enhanced FiLM conditioning
        self.film_conditioning = EnhancedFiLMConditioning(act_emb_dim, hidden_dim)
        
        # Improved convolutional input processing
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        
        # Spatial position encoding
        self.pos_encoding = SpatialPositionEncoding(hidden_dim, grid_h, grid_w)
        
        # Cross-attention blocks (spatial <-> action)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, act_emb_dim, num_heads, dropout)
            for _ in range(min(2, depth))  # Use fewer cross-attention blocks
        ])
        
        # Enhanced DiT blocks with mix of regular and linear attention
        self.dit_blocks = nn.ModuleList()
        for i in range(depth):
            use_linear = (i % 2 == 1) and (depth > 2)  # Alternate between regular and linear attention
            self.dit_blocks.append(
                EnhancedDiTBlock(hidden_dim, num_heads, dropout, use_linear_attn=use_linear)
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Improved output processing
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_vec, a_onehot, action_dropout_p=0.0):
        B = z_vec.size(0)
        
        # Convert to grid and process with conv
        g = self.adapter.vec_to_grid(z_vec)           # [B,1,H,W]
        x = self.conv_in(g)                           # [B,C,H,W]
        
        # Enhanced action encoding with dropout
        action_embed = self.action_encoder(a_onehot)  # [B, act_emb_dim]
        if self.training and action_dropout_p > 0:
            mask = (torch.rand(B, 1, device=z_vec.device) > action_dropout_p).float()
            action_embed = action_embed * mask
        
        # Enhanced FiLM conditioning
        x = self.film_conditioning(x, action_embed)
        
        # Flatten to sequence and add positional encoding [B, HW, C]
        B, C, Hh, Ww = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.pos_encoding(x_seq)
        
        # Cross-attention blocks (spatial <-> action)
        for cross_block in self.cross_attn_blocks:
            x_seq = cross_block(x_seq, action_embed)
        
        # Enhanced DiT blocks with mixed attention
        for dit_block in self.dit_blocks:
            x_seq = dit_block(x_seq)
            
        x_seq = self.norm(x_seq)
        
        # Back to grid and output processing
        x = x_seq.transpose(1, 2).view(B, C, Hh, Ww)
        delta_grid = self.conv_out(x)
        delta_vec = self.adapter.grid_to_vec(delta_grid)
        
        # Add residual connection to encourage small updates
        delta_vec = delta_vec * self.residual_weight
        
        return delta_vec  # Δz
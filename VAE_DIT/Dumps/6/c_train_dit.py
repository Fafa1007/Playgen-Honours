# c_train_dit.py
import os
import math
import random
import numpy as np
from typing import Tuple
from d_dit_model import ConvTransDynamics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from b_vae_model import VAE
from a_load_mario_dataset import MarioFramesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Enhanced Hyperparameters
# --------------------------
USE_SKIPS = True
H = {
    "latent_dim": 256,        # keep latent dim same as VAE
    "grid_h": 16,             
    "grid_w": 16,             
    "action_dim": 5,          # multi-hot action vector size
    "act_emb_dim": 128,       # larger action embedding for better combination modeling
    "hidden_dim": 384,        # increased hidden dimension for better capacity
    "depth": 6,               # more transformer layers for better modeling
    "num_heads": 12,          # more attention heads
    "dropout": 0.15,          # slightly higher dropout for regularization
    "batch_size": 24,         # larger batch size for stable training
    "epochs": 5,              # more epochs for convergence
    "lr": 5e-4,               # slightly lower learning rate for stability
    "weight_decay": 1e-4,     # reduced weight decay
    "warmup_steps": 200,      # longer warmup
    "max_steps": 5_000,       # more training steps
    "grad_clip": 0.5,         # tighter gradient clipping
    "lambda_contrast": 1.0,   # increased contrastive loss weight
    "contrast_margin": 0.15,  # larger margin for better separation
    "lambda_img": 0.5,        # increased image reconstruction loss
    "img_loss_every": 2,      # compute image loss more frequently
    "lambda_multistep": 0.3,  # multi-step prediction loss
    "multistep_horizon": 3,   # predict 3 steps ahead
    "lambda_action_reg": 0.1, # action consistency regularization
    "lambda_smooth": 0.05,    # temporal smoothness regularization
    "action_dropout_p": 0.15, # increased action dropout
    "scheduled_sampling_p0": 0.0,
    "scheduled_sampling_p1": 0.4,  # higher scheduled sampling
    "num_workers": 2,
    "seed": 42,
    "label_smoothing": 0.1,   # label smoothing for robustness
    "ema_decay": 0.99,        # exponential moving average for model weights
    "gradient_penalty": 0.01  # gradient penalty for stability
}

torch.manual_seed(H["seed"])
random.seed(H["seed"])
np.random.seed(H["seed"])

# --------------------------
# Enhanced Dataset with Multi-Step Sequences
# --------------------------
class EnhancedMarioLatentDataset(Dataset):
    def __init__(self, data_root: str, vae: VAE, image_size: int = 128, sequence_length: int = 4):
        self.vae = vae.eval()
        self.sequence_length = sequence_length
        base = MarioFramesDataset(root_dir=data_root, image_size=image_size, return_actions=True)
        
        self.latents, self.skips, self.actions, self.images = [], [], [], []
        print(f"[Dataset] Encoding {len(base)} frames to latents...")
        
        with torch.no_grad():
            for i, (img, action) in enumerate(base):
                if i % 500 == 0:
                    print(f"  Progress: {i}/{len(base)}")
                    
                self.images.append(img)  
                img_batch = img.unsqueeze(0).to(device)
                mu, logvar, s = self.vae.encoder(img_batch)
                
                self.latents.append(mu.squeeze(0).cpu())
                self.skips.append([t.squeeze(0).cpu() for t in s] if s else None)
                
                # Ensure action is properly formatted as multi-hot vector
                if isinstance(action, (list, tuple)):
                    action = torch.tensor(action, dtype=torch.float32)
                elif action.dtype != torch.float32:
                    action = action.float()
                    
                self.actions.append(action)

        print(f"[Dataset] Encoded {len(self.latents)} frames")
        
        # Create sequences for multi-step prediction
        self.sequences = []
        for i in range(len(self.latents) - sequence_length + 1):
            seq = {
                'latents': self.latents[i:i+sequence_length],
                'actions': self.actions[i:i+sequence_length-1],  # N-1 actions for N states
                'skips': self.skips[i:i+sequence_length] if self.skips[0] else None,
                'images': self.images[i:i+sequence_length]
            }
            self.sequences.append(seq)

    def __len__(self): 
        return len(self.sequences)

    def __getitem__(self, i):
        seq = self.sequences[i]
        
        # Return first state, all actions, and all subsequent states
        z_0 = seq['latents'][0]  # Initial state
        actions = seq['actions']  # List of actions
        z_targets = seq['latents'][1:]  # Target states
        skips_0 = seq['skips'][0] if seq['skips'] else None
        img_targets = seq['images'][1:]  # Target images
        
        return z_0, actions, z_targets, skips_0, img_targets

# --------------------------
# Enhanced Training Utilities
# --------------------------
def cosine_warmup_lr(step, base_lr, warmup, max_steps):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))

def label_smoothed_mse_loss(pred, target, smoothing=0.1):
    """MSE loss with label smoothing for robustness."""
    mse = F.mse_loss(pred, target, reduction='none')
    # Add small noise to targets for smoothing
    if smoothing > 0:
        noise = torch.randn_like(target) * smoothing * target.std()
        smooth_target = target + noise
        smooth_mse = F.mse_loss(pred, smooth_target, reduction='none')
        mse = 0.7 * mse + 0.3 * smooth_mse
    return mse.mean()

def enhanced_contrastive_loss(z_pred, z_target, z_pred_wrong, margin, temperature=0.1):
    """Enhanced contrastive loss with temperature scaling."""
    d_right = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=1)
    d_wrong = F.mse_loss(z_pred_wrong, z_target, reduction="none").mean(dim=1)
    
    # Temperature scaling for better gradients
    d_right = d_right / temperature
    d_wrong = d_wrong / temperature
    
    # Contrastive loss with exponential form for better stability
    loss = torch.clamp(margin - d_wrong + d_right, min=0.0)
    return loss.mean()

def action_consistency_loss(actions, pred_actions_features):
    """Regularization to ensure action embeddings are consistent."""
    # This is a placeholder - implement based on your action encoding
    return torch.tensor(0.0, device=actions.device)

def temporal_smoothness_loss(z_sequence):
    """Encourage smooth transitions between predicted states."""
    if len(z_sequence) < 2:
        return torch.tensor(0.0, device=z_sequence[0].device)
    
    smoothness = 0.0
    for i in range(len(z_sequence) - 1):
        diff = z_sequence[i+1] - z_sequence[i]
        smoothness += F.mse_loss(diff, torch.zeros_like(diff))
    
    return smoothness / (len(z_sequence) - 1)

def gradient_penalty(model, z_vec, a_onehot):
    """Gradient penalty for training stability."""
    z_vec.requires_grad_(True)
    output = model(z_vec, a_onehot)
    
    gradients = torch.autograd.grad(
        outputs=output.sum(),
        inputs=z_vec,
        create_graph=True,
        retain_graph=True
    )[0]
    
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# --------------------------
# Main Enhanced Training Loop
# --------------------------
def train(data_root: str, vae_path: str, out_path: str = "enhanced_dit_model.pt"):
    # VAE
    vae = VAE(latent_dim=H["latent_dim"]).to(device)
    ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(ckpt["model"], strict=True)
    vae.eval()

    # Enhanced Data
    ds = EnhancedMarioLatentDataset(
        data_root, vae, image_size=128, 
        sequence_length=H["multistep_horizon"]+1
    )
    dl = DataLoader(
        ds, batch_size=H["batch_size"], shuffle=True, 
        num_workers=H["num_workers"], pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )

    # Enhanced Model
    model = ConvTransDynamics(
        H["latent_dim"], H["grid_h"], H["grid_w"], H["action_dim"], H["act_emb_dim"],
        H["hidden_dim"], H["depth"], H["num_heads"], H["dropout"]
    ).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    model.apply(init_weights)

    # Optimizer and EMA
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=H["lr"], 
        weight_decay=H["weight_decay"],
        betas=(0.9, 0.95)
    )
    ema = ExponentialMovingAverage(model, decay=H["ema_decay"])

    # Loss functions
    criterion = nn.MSELoss()
    
    step = 0
    num_steps_total = min(H["max_steps"], H["epochs"] * len(dl))
    print(f"[train] total steps (for scheduler): {num_steps_total}")
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5

    for epoch in range(H["epochs"]):
        model.train()
        total_loss = 0.0
        total_latent_loss = 0.0
        total_contrast_loss = 0.0
        total_img_loss = 0.0
        total_multistep_loss = 0.0
        total_smooth_loss = 0.0
        
        # Dynamic scheduled sampling
        p_sched = H["scheduled_sampling_p0"] + (H["scheduled_sampling_p1"] - H["scheduled_sampling_p0"]) * (epoch / max(1, H["epochs"] - 1))

        for batch_idx, (z_0, actions_seq, z_targets, skips_0, img_targets) in enumerate(dl):
            # Move to device
            z_0 = z_0.to(device).float()
            z_targets = [z_t.to(device).float() for z_t in z_targets]
            actions_seq = [a.to(device).float() for a in actions_seq]
            img_targets = [img.to(device).float() for img in img_targets]
            
            B = z_0.size(0)
            
            # Multi-step prediction
            z_current = z_0
            z_predictions = []
            
            for step_idx, action in enumerate(actions_seq):
                # Scheduled sampling: sometimes use predicted state instead of ground truth
                if step_idx > 0 and p_sched > 0 and random.random() < p_sched:
                    # Use previous prediction (with detached gradients for stability)
                    z_current = z_predictions[-1].detach()
                elif step_idx > 0:
                    # Use ground truth from previous step
                    z_current = z_targets[step_idx - 1]
                
                # Predict next state
                pred_delta = model(z_current, action, action_dropout_p=H["action_dropout_p"])
                z_pred = z_current + pred_delta
                z_predictions.append(z_pred)
            
            # === Primary Loss: Latent Space Prediction ===
            loss_latent = 0.0
            for i, (z_pred, z_target) in enumerate(zip(z_predictions, z_targets)):
                step_weight = 0.8 ** i  # Exponentially decay weight for future steps
                loss_latent += step_weight * label_smoothed_mse_loss(
                    z_pred, z_target, smoothing=H["label_smoothing"]
                )
            loss_latent /= len(z_predictions)
            
            # === Enhanced Contrastive Loss ===
            loss_contrast = 0.0
            if H["lambda_contrast"] > 0:
                for i, (z_pred, z_target, action) in enumerate(zip(z_predictions, z_targets, actions_seq)):
                    # Create wrong actions by shuffling multi-hot vectors
                    idx_perm = torch.randperm(B, device=device)
                    action_wrong = action[idx_perm]
                    
                    # Ensure wrong action is actually different
                    same_action_mask = (action_wrong == action).all(dim=1)
                    if same_action_mask.any():
                        # For identical actions, flip a random button
                        n_same = same_action_mask.sum()
                        rand_buttons = torch.randint(0, H["action_dim"], (n_same,), device=device)
                        action_wrong[same_action_mask, rand_buttons] = 1 - action_wrong[same_action_mask, rand_buttons]
                    
                    with torch.no_grad():
                        z_base = z_0 if i == 0 else z_targets[i-1]
                        z_pred_wrong = z_base + model(z_base, action_wrong, action_dropout_p=0.0)
                    
                    step_weight = 0.9 ** i
                    loss_contrast += step_weight * enhanced_contrastive_loss(
                        z_pred, z_target, z_pred_wrong, 
                        margin=H["contrast_margin"], temperature=0.1
                    )
                loss_contrast /= len(z_predictions)
            
            # === Multi-step Consistency Loss ===
            loss_multistep = 0.0
            if H["lambda_multistep"] > 0 and len(z_predictions) >= 2:
                # Predict from first state to final state directly
                cumulative_action = torch.zeros_like(actions_seq[0])
                for action in actions_seq:
                    # Combine actions (for multi-hot, this is element-wise OR/max)
                    cumulative_action = torch.maximum(cumulative_action, action)
                
                direct_delta = model(z_0, cumulative_action, action_dropout_p=0.0)
                direct_pred = z_0 + direct_delta
                final_pred = z_predictions[-1]
                
                loss_multistep = F.mse_loss(direct_pred, final_pred.detach())
            
            # === Temporal Smoothness Loss ===
            loss_smooth = temporal_smoothness_loss(z_predictions)
            
            # === Image Reconstruction Loss ===
            loss_img = torch.tensor(0.0, device=device)
            if H["lambda_img"] > 0 and (step % H["img_loss_every"] == 0):
                for i, (z_pred, img_target) in enumerate(zip(z_predictions, img_targets)):
                    with torch.no_grad():
                        if USE_SKIPS and skips_0 is not None:
                            recon_pred = vae.decoder(z_pred, skips=[s.to(device) for s in skips_0])
                        else:
                            recon_pred = vae.decoder(z_pred)
                    
                    step_weight = 0.8 ** i
                    loss_img += step_weight * F.mse_loss(recon_pred, img_target)
                loss_img /= len(z_predictions)
            
            # === Gradient Penalty for Stability ===
            loss_gp = 0.0
            if H["gradient_penalty"] > 0:
                loss_gp = gradient_penalty(model, z_0, actions_seq[0])
            
            # === Combined Loss ===
            loss = (loss_latent + 
                   H["lambda_contrast"] * loss_contrast + 
                   H["lambda_img"] * loss_img +
                   H["lambda_multistep"] * loss_multistep +
                   H["lambda_smooth"] * loss_smooth +
                   H["gradient_penalty"] * loss_gp)

            # === Optimization Step ===
            lr = cosine_warmup_lr(step, H["lr"], H["warmup_steps"], num_steps_total)
            for pg in opt.param_groups: 
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping with norm monitoring
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            
            opt.step()
            ema.update()  # Update EMA weights

            # Accumulate losses for logging
            total_loss += loss.item()
            total_latent_loss += loss_latent.item()
            total_contrast_loss += loss_contrast.item()
            total_img_loss += loss_img.item()
            total_multistep_loss += loss_multistep.item()
            total_smooth_loss += loss_smooth.item()
            
            step += 1
            
            # Log progress every 100 steps
            if step % 100 == 0:
                print(f"[Step {step}] Loss: {loss.item():.6f}, "
                      f"Latent: {loss_latent.item():.6f}, "
                      f"Contrast: {loss_contrast.item():.6f}, "
                      f"Grad Norm: {grad_norm:.4f}")

        # Epoch summary
        avg_loss = total_loss / len(dl)
        avg_latent = total_latent_loss / len(dl)
        avg_contrast = total_contrast_loss / len(dl)
        avg_img = total_img_loss / len(dl)
        avg_multistep = total_multistep_loss / len(dl)
        avg_smooth = total_smooth_loss / len(dl)
        
        print(f"\n[Epoch {epoch+1}/{H['epochs']}]")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Latent Loss: {avg_latent:.6f}")
        print(f"  Contrast Loss: {avg_contrast:.6f}")
        print(f"  Image Loss: {avg_img:.6f}")
        print(f"  Multi-step Loss: {avg_multistep:.6f}")
        print(f"  Smooth Loss: {avg_smooth:.6f}")
        print(f"  Scheduled Sampling p: {p_sched:.3f}")
        print(f"  Learning Rate: {lr:.2e}")
        
        # Early stopping and model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model with EMA weights
            ema.apply_shadow()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'hyperparameters': H
            }, out_path.replace('.pt', '_best.pt'))
            ema.restore()
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {patience_limit} epochs without improvement")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'hyperparameters': H
    }, out_path)
    
    print(f"\nTraining completed!")
    print(f"Final model saved to: {out_path}")
    print(f"Best model saved to: {out_path.replace('.pt', '_best.pt')}")
    print(f"Best loss achieved: {best_loss:.6f}")


if __name__ == "__main__":
    train(
        data_root="small_mario_data",
        vae_path="vae_best.pt",
        out_path="enhanced_hybrid_convtrans.pt"
    )
    
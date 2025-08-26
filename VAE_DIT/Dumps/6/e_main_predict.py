# e_main_predict.py
import torch
from b_vae_model import VAE, USE_SKIPS
from d_dit_model import ConvTransDynamics
from a_load_mario_dataset import MarioFramesDataset
from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load Enhanced Models
# --------------------------
def load_models(vae_path="vae_best.pt", dit_path="enhanced_hybrid_convtrans_best.pt"):
    # Load VAE
    vae = VAE(latent_dim=256).to(device)
    ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(ckpt["model"], strict=True)
    vae.eval()
    
    # Load Enhanced DiT
    dit_checkpoint = torch.load(dit_path, map_location=device)
    
    # Extract hyperparameters from checkpoint if available
    if 'hyperparameters' in dit_checkpoint:
        H = dit_checkpoint['hyperparameters']
    else:
        # Fallback to default enhanced hyperparameters
        H = {
            "latent_dim": 256, "grid_h": 16, "grid_w": 16, "action_dim": 5,
            "act_emb_dim": 128, "hidden_dim": 384, "depth": 6, "num_heads": 12, "dropout": 0.15
        }
    
    dit = ConvTransDynamics(
        latent_dim=H["latent_dim"],
        grid_h=H["grid_h"], 
        grid_w=H["grid_w"],
        action_dim=H["action_dim"],
        act_emb_dim=H["act_emb_dim"],
        hidden_dim=H["hidden_dim"],
        depth=H["depth"],
        num_heads=H["num_heads"],
        dropout=H["dropout"]
    ).to(device)

    dit.load_state_dict(dit_checkpoint['model_state_dict'])
    dit.eval()
    
    return vae, dit

# --------------------------
# Action Visualization Helper
# --------------------------
def action_to_string(action_vector):
    """Convert multi-hot action vector to readable string."""
    action_names = ['Left', 'Right', 'Up', 'Down', 'Jump']  # Adjust based on your dataset
    active_actions = [action_names[i] for i, val in enumerate(action_vector) if val > 0.5]
    return ' + '.join(active_actions) if active_actions else 'None'

def add_text_to_image(image, text, position=(10, 10), font_size=16):
    """Add text overlay to image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Add black outline for better visibility
    x, y = position
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), text, font=font, fill="black")
    
    draw.text(position, text, font=font, fill="white")
    return image

# --------------------------
# Enhanced Prediction with Action Analysis
# --------------------------
def predict_sequence_with_analysis(vae, dit, dataset, start_idx=0, num_steps=50, reset_every_k=10):
    """Enhanced prediction with detailed action analysis and error metrics."""
    
    output_dir = "predicted_frames/enhanced_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = []
    ground_truths = []
    action_effects = []
    prediction_errors = []
    
    with torch.no_grad():
        # Initialize with ground truth frame
        img, _ = dataset[start_idx]
        img_batch = img.unsqueeze(0).to(device)
        mu, logvar, skips = vae.encoder(img_batch)
        z_current = mu
        
        print(f"Starting prediction from frame {start_idx}")
        
        for step in range(num_steps):
            frame_idx = start_idx + step
            if frame_idx >= len(dataset) - 1:
                break
                
            # Get current action
            _, current_action = dataset[frame_idx]
            if not isinstance(current_action, torch.Tensor):
                current_action = torch.tensor(current_action, dtype=torch.float32)
            
            action_batch = current_action.unsqueeze(0).to(device)
            action_str = action_to_string(current_action.cpu().numpy())
            
            # Store pre-prediction state for analysis
            z_before = z_current.clone()
            
            # Predict next state
            delta_z = dit(z_current, action_batch)
            z_predicted = z_current + delta_z
            
            # Analyze action effect magnitude
            delta_magnitude = torch.norm(delta_z).item()
            action_effects.append({
                'step': step,
                'action': action_str,
                'delta_magnitude': delta_magnitude,
                'active_buttons': int(current_action.sum().item())
            })
            
            # Decode predicted frame
            if USE_SKIPS:
                predicted_img = vae.decoder(z_predicted, skips=skips)
            else:
                predicted_img = vae.decoder(z_predicted)
                
            # Convert to displayable format
            pred_img_np = (predicted_img[0].cpu() * 0.5 + 0.5).clamp(0, 1)
            pred_img_np = (pred_img_np.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pred_img_pil = Image.fromarray(pred_img_np)
            
            # Add action text overlay
            pred_img_pil = add_text_to_image(
                pred_img_pil, 
                f"Step {step}: {action_str}",
                position=(5, 5)
            )
            
            # Get ground truth for comparison
            gt_img, _ = dataset[frame_idx + 1]
            gt_img_batch = gt_img.unsqueeze(0).to(device)
            gt_mu, _, gt_skips = vae.encoder(gt_img_batch)
            
            # Calculate prediction error
            latent_error = F.mse_loss(z_predicted, gt_mu).item()
            prediction_errors.append(latent_error)
            
            # Save predicted frame
            pred_img_pil.save(os.path.join(output_dir, f"pred_{step:04d}.png"))
            
            predictions.append(z_predicted.cpu())
            ground_truths.append(gt_mu.cpu())
            
            # Reset mechanism
            if (step + 1) % reset_every_k == 0 and step < num_steps - 1:
                print(f"  Reset at step {step + 1}, error so far: {latent_error:.6f}")
                # Re-encode from ground truth
                gt_img, _ = dataset[frame_idx + 1]
                gt_img_batch = gt_img.unsqueeze(0).to(device)
                mu, logvar, skips = vae.encoder(gt_img_batch)
                z_current = mu
            else:
                # Continue with prediction
                # Re-encode prediction to get updated skips
                mu_pred, _, skips_pred = vae.encoder(predicted_img)
                z_current = mu_pred
                skips = skips_pred
    
    # Analyze results
    analyze_prediction_results(action_effects, prediction_errors, output_dir)
    
    return predictions, ground_truths, action_effects

def analyze_prediction_results(action_effects, prediction_errors, output_dir):
    """Analyze and save prediction statistics."""
    import matplotlib.pyplot as plt
    
    # Action effect analysis
    action_types = {}
    for effect in action_effects:
        action = effect['action']
        if action not in action_types:
            action_types[action] = []
        action_types[action].append(effect['delta_magnitude'])
    
    print("\n=== Action Effect Analysis ===")
    for action, magnitudes in action_types.items():
        avg_effect = np.mean(magnitudes)
        std_effect = np.std(magnitudes)
        print(f"{action:15} | Avg: {avg_effect:.4f} ± {std_effect:.4f} | Count: {len(magnitudes)}")
    
    # Error progression analysis
    print(f"\n=== Prediction Error Analysis ===")
    print(f"Mean error: {np.mean(prediction_errors):.6f}")
    print(f"Std error:  {np.std(prediction_errors):.6f}")
    print(f"Max error:  {np.max(prediction_errors):.6f}")
    
    # Save analysis plots
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error over time
        ax1.plot(prediction_errors, 'b-', alpha=0.7)
        ax1.set_xlabel('Prediction Step')
        ax1.set_ylabel('Latent MSE Error')
        ax1.set_title('Prediction Error Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Action effect distribution
        all_effects = [effect['delta_magnitude'] for effect in action_effects]
        ax2.hist(all_effects, bins=20, alpha=0.7, color='green')
        ax2.set_xlabel('Delta Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Action Effects')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

# --------------------------
# Enhanced Comparison Grid
# --------------------------
def create_enhanced_comparison_grid(data_root, predictions_dir, num_samples=6):
    """Create enhanced comparison grid with action information."""
    
    pred_dir = os.path.join(predictions_dir, "enhanced_analysis")
    gt_dir = os.path.join(data_root, "frames")
    
    # Load actions for annotation
    dataset = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=True)
    
    # Pick frames with diverse actions
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    sampled_indices = np.linspace(0, len(pred_files)-1, num_samples, dtype=int)
    
    sampled_files = [pred_files[i] for i in sampled_indices]
    
    # Create grid
    w, h = 128, 128
    grid_width = num_samples
    grid_height = 3  # GT, Pred, Diff
    
    final_img = Image.new("RGB", (grid_width * w, grid_height * h))
    
    for col, fname in enumerate(sampled_files):
        frame_idx = int(fname.split('_')[1].split('.')[0])
        
        # Ground truth
        gt_img = Image.open(os.path.join(gt_dir, fname.replace('pred_', 'frame_'))).resize((w, h))
        
        # Prediction  
        pred_img = Image.open(os.path.join(pred_dir, fname)).resize((w, h))
        
        # Difference (simple visualization)
        gt_array = np.array(gt_img).astype(float)
        pred_array = np.array(pred_img).astype(float)
        diff_array = np.abs(gt_array - pred_array).astype(np.uint8)
        diff_img = Image.fromarray(diff_array)
        
        # Get action for this frame
        if frame_idx < len(dataset):
            _, action = dataset[frame_idx]
            action_str = action_to_string(action.numpy() if isinstance(action, torch.Tensor) else action)
            
            # Add action labels
            gt_img = add_text_to_image(gt_img, f"GT", (5, 5), 12)
            pred_img = add_text_to_image(pred_img, f"Pred: {action_str}", (5, 5), 10)
            diff_img = add_text_to_image(diff_img, "Diff", (5, 5), 12)
        
        # Paste to grid
        final_img.paste(gt_img, (col * w, 0))
        final_img.paste(pred_img, (col * w, h))
        final_img.paste(diff_img, (col * w, 2 * h))
    
    comp_path = os.path.join(predictions_dir, "enhanced_comparison.png")
    final_img.save(comp_path)
    print(f"Enhanced comparison grid saved to: {comp_path}")
    
    return comp_path

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    print("Loading enhanced models...")
    vae, dit = load_models()
    
    print("Loading dataset...")
    data_root = "small_mario_data"
    dataset = MarioFramesDataset(root_dir=data_root, image_size=128, return_actions=True)
    
    print("Running enhanced prediction with analysis...")
    predictions, ground_truths, action_effects = predict_sequence_with_analysis(
        vae, dit, dataset, 
        start_idx=0, 
        num_steps=50, 
        reset_every_k=8
    )
    
    print("Creating enhanced comparison grid...")
    create_enhanced_comparison_grid(data_root, "predicted_frames")
    
    print("\nPrediction complete! Check the following outputs:")
    print("  - predicted_frames/enhanced_analysis/ : Individual predicted frames")
    print("  - predicted_frames/enhanced_comparison.png : Comparison grid")
    print("  - predicted_frames/enhanced_analysis/analysis.png : Statistical analysis")
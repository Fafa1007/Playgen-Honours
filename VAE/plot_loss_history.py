"""
plot_loss_history.py
--------------------
Utility script to plot VAE training loss history.

Inputs:
- checkpoints/loss_history.json

Outputs:
- checkpoints/loss_curve.png : line chart of loss vs epoch
"""

import os
import json
import matplotlib.pyplot as plt

def main():
    hist_path = "checkpoints/loss_history.json"
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"No loss history found at {hist_path}. Run training first.")

    # Load loss values
    with open(hist_path, "r") as f:
        loss_history = json.load(f)

    epochs = list(range(1, len(loss_history) + 1))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss over Epochs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save
    out_path = "checkpoints/loss_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve to {out_path}")

    # Show interactively (optional)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

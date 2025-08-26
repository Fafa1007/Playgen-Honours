
# a_load_mario_dataset.py
# Dataset that yields (frame_t, action_t, frame_{t+1}) triplets for next-frame prediction.
# Images are returned as float32 tensors in [-1, 1] with shape [3, H, W].
# Actions are float32 vectors in [0, 1].

import os
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MarioTransitionsDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 128):
        """
        Expected layout:
          root_dir/
            frames/
              000000.png, 000001.png, ...
            actions.txt  # one comma-separated binary vector per line, e.g. '0,1,0,0,0'

        We construct aligned pairs (t, t+1). If counts mismatch, we clip safely.
        """
        self.root_dir = root_dir
        self.frames_dir = os.path.join(root_dir, "frames")
        self.actions_path = os.path.join(root_dir, "actions.txt")

        # Sorted list of frame files
        self.frame_paths = sorted(
            [os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir) 
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        # Load actions
        self.actions: List[List[float]] = []
        if os.path.exists(self.actions_path):
            with open(self.actions_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # parse full comma-separated binary vector
                    parts = [p.strip() for p in line.split(",")]
                    row = [float(p) for p in parts]
                    self.actions.append(row)
        else:
            # default: no buttons pressed
            self.actions = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(len(self.frame_paths))]

        # Align lengths for transitions (t -> t+1 uses action at t)
        n = min(len(self.frame_paths) - 1, len(self.actions) - 1)
        if n < 1:
            raise ValueError("Not enough aligned frames and actions to build transitions.")
        self.frame_paths = self.frame_paths[: n + 1]   # we need t and t+1 indexed up to n
        self.actions = self.actions[: n + 1]
        self.N = n  # number of usable transitions

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # to [-1, 1]
        ])

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We use (t, t+1) with action at t
        img_t_path = self.frame_paths[idx]
        img_tp1_path = self.frame_paths[idx + 1]

        img_t = Image.open(img_t_path).convert("RGB")
        img_tp1 = Image.open(img_tp1_path).convert("RGB")

        x_t = self.transform(img_t)      # [3, H, W], in [-1, 1]
        x_tp1 = self.transform(img_tp1)  # [3, H, W], in [-1, 1]

        a_t = torch.tensor(self.actions[idx], dtype=torch.float32)  # e.g. length 5

        return x_t, a_t, x_tp1

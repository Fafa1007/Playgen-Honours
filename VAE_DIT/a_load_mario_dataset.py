# load_mario_dataset.py
# Dataset for Mario frames. Ensures images are [3, 128, 128] float32 in [-1, 1].
# Important: Avoid lambdas/closures in transforms so DataLoader workers (spawn/forkserver)
# can pickle the dataset on macOS.

import os
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MarioFramesDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 128, return_actions: bool = True):
        """
        Expected layout:
          root_dir/
            frames/        image files (png, jpg, jpeg)
            actions.txt    optional, one integer per line (or CSV)
        """
        self.root_dir = root_dir
        self.frames_dir = os.path.join(root_dir, "frames")
        self.return_actions = return_actions

        # Collect frame paths
        exts = (".png", ".jpg", ".jpeg")
        self.frame_paths: List[str] = sorted(
            os.path.join(self.frames_dir, f)
            for f in os.listdir(self.frames_dir)
            if f.lower().endswith(exts)
        )

        # Load actions if present; otherwise fill with zeros
        actions_txt = os.path.join(root_dir, "actions.txt")
        actions_csv = os.path.join(root_dir, "actions.csv")
        self.actions: List[int] = []

        if self.return_actions and os.path.exists(actions_txt):
            with open(actions_txt, "r") as f:
                self.actions = [int(line.strip().split(",")[0]) for line in f if line.strip()]
        elif self.return_actions and os.path.exists(actions_csv):
            with open(actions_csv, "r") as f:
                self.actions = [int(line.strip().split(",")[0]) for line in f if line.strip()]
        else:
            self.actions = [0] * len(self.frame_paths)

        # Truncate safely if counts differ
        n_frames = len(self.frame_paths)
        n_actions = len(self.actions)
        if n_frames != n_actions:
            n = min(n_frames, n_actions)
            if n == 0:
                raise RuntimeError("No frames or actions found.")
            print(f"[MarioFramesDataset] Warning: {n_frames} frames but {n_actions} actions. Will use the first {n} of each.")
            self.frame_paths = self.frame_paths[:n]
            self.actions = self.actions[:n]

        # Transform pipeline: resize → ToTensor() → Normalize to [-1, 1]
        # Note: Normalize(x; mean=0.5, std=0.5) maps [0,1] to [-1,1] per channel.
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                                   # [0,1], float32, CHW
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.frame_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)     # float32, [3, H, W], values ~ [-1, 1]
        action = self.actions[idx] if self.return_actions else 0
        return img_t, action

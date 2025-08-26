import os
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _read_actions_txt(path: str) -> torch.Tensor:
    """
    Expects a plain text file with one action per line.
    Each line can be a single integer class id or a comma/space separated list of 0/1 buttons.
    Returns a float tensor [N, A].
    """
    actions: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
            elif " " in s:
                parts = [p.strip() for p in s.split()]
            else:
                # single integer class id -> one hot of size max_class+1 inferred later
                parts = [s]
            row = [float(p) for p in parts]
            actions.append(row)

    # pad rows to equal width
    max_len = max(len(r) for r in actions)
    padded = []
    for r in actions:
        if len(r) == 1 and float(r[0]).is_integer() and max_len == 1:
            # just a scalar class id; keep as [N,1] for now
            padded.append([r[0]])
        else:
            padded.append(r + [0.0] * (max_len - len(r)))

    a = torch.tensor(padded, dtype=torch.float32)
    # if it is scalar class ids but you prefer one hot here, convert:
    if a.shape[1] == 1 and a.numel() > 0 and a.dtype == torch.float32:
        unique = sorted(list(set(a.view(-1).tolist())))
        # If ids are 0..K-1 and K small, one-hot
        if len(unique) <= 16 and unique == list(range(len(unique))):
            K = len(unique)
            oh = torch.zeros(a.shape[0], K, dtype=torch.float32)
            idx = a.view(-1).long().clamp(min=0, max=K - 1)
            oh[torch.arange(a.shape[0]), idx] = 1.0
            a = oh
    return a


class MarioTransitionsDataset(Dataset):
    """
    Produces (x_t, a_t, x_{t+1}) triplets from:
      root_dir/
        frames/            (folder with 000001.png, 000002.png, ...)
        actions.txt        (N-1 lines, format described above)
    Images are mapped to [-1, 1].
    """

    def __init__(self, root_dir: str, image_size: int = 128):
        self.root_dir = root_dir
        frame_dir = os.path.join(root_dir, "frames")
        assert os.path.isdir(frame_dir), f"Frames folder not found at {frame_dir}"

        # discover frames
        names = sorted([n for n in os.listdir(frame_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))])
        assert len(names) >= 2, "Need at least two frames"
        self.frame_paths = [os.path.join(frame_dir, n) for n in names]

        # load actions
        actions_path = os.path.join(root_dir, "actions.txt")
        assert os.path.isfile(actions_path), f"actions.txt not found at {actions_path}"
        self.actions = _read_actions_txt(actions_path)

        # Alignment: actions must be exactly frames - 1
        msg = f"Actions {self.actions.shape[0]} must equal frames-1 {len(self.frame_paths)-1}"
        assert self.actions.shape[0] == len(self.frame_paths) - 1, msg

        # transforms
        self.to_tensor = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] -> [-1,1]
        ])

    def __len__(self) -> int:
        # Number of transition pairs
        return len(self.frame_paths) - 1

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = self._load_image(self.frame_paths[idx])
        x_tp1 = self._load_image(self.frame_paths[idx + 1])
        a_t = self.actions[idx]
        return x_t, a_t, x_tp1

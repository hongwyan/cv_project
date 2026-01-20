# src/brats2d_dataset.py
import torch
from torch.utils.data import Dataset
from brats2p5d_dataset import BraTS2p5D


class BraTS2D(Dataset):
    """
    Wraps your existing BraTS2p5D dataset but returns only the center slice as (1,H,W).
    """
    def __init__(self, *args, **kwargs):
        self.ds = BraTS2p5D(*args, **kwargs)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]              # x: (3,H,W), y: (1,H,W)
        x2d = x[1:2, ...].contiguous()   # keep center slice -> (1,H,W)
        return x2d, y

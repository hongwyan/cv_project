# src/step5_dataloader_sanity.py
# Run: python src/step5_dataloader_sanity.py

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from brats2p5d_dataset import BraTS2p5D


def main():
    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"  # change if needed

    ds = BraTS2p5D(
        root=root,
        max_patients=10,
        only_tumor_slices=True,   # fast sanity
        cache_volumes=False
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,      # Windows-safe; can raise later
        pin_memory=True
    )

    x, y = next(iter(dl))
    print("batch x:", x.shape, x.dtype)  # (B,3,240,240)
    print("batch y:", y.shape, y.dtype)  # (B,1,240,240)
    print("tumor pixels per sample:", y.sum(dim=(1,2,3)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    print("device:", device)
    print("x device:", x.device, "y device:", y.device)


if __name__ == "__main__":
    main()

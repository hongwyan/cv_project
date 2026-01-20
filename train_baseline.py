# src/train_baseline.py
# Run: python src/train_baseline.py
#
# Baseline training loop for 2.5D U-Net on BraTS (binary tumor).
# - Train on a small subset first (max_patients=10, tumor slices only) to sanity-check.
# - Loss: BCEWithLogits + Soft Dice
# - Metrics: Dice score

from __future__ import annotations
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, random_split

from brats2p5d_dataset import BraTS2p5D
from unet import UNet
from losses import BCEDiceLoss, dice_score


def set_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dl, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        dsc = dice_score(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_dice += dsc.item() * bs
        n += bs

    return total_loss / max(n, 1), total_dice / max(n, 1)


def main():
    set_seed(0)

    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Dataset (small subset for sanity)
    ds = BraTS2p5D(
        root=root,
        max_patients=10,
        only_tumor_slices=True,
        cache_volumes=True,
        seed=0,
    )

    # Split
    val_frac = 0.1
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Loaders
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = UNet(in_ch=3, out_ch=1, base_ch=32).to(device)

    # Loss + optim
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    epochs = 30
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_dice = 0.0
        steps = 0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            with torch.no_grad():
                dsc = dice_score(logits, y)

            running_loss += loss.item()
            running_dice += dsc.item()
            steps += 1

        train_loss = running_loss / max(steps, 1)
        train_dice = running_dice / max(steps, 1)

        val_loss, val_dice = evaluate(model, val_dl, criterion, device)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
            f"time={dt:.1f}s"
        )

    # Save checkpoint
    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    ckpt_path = out / "unet_2p5d_baseline.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()

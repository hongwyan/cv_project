# src/train_2d_baseline.py
# Run: python src/train_2d_baseline.py

from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from brats2d_dataset import BraTS2D
from unet import UNet
from losses import BCEDiceLoss, dice_score


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dl, criterion, device):
    model.eval()
    total_loss_pos = 0.0
    total_dice_pos = 0.0
    n_pos = 0

    neg_slices = 0
    fp_slices = 0
    n = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        logits = model(x)
        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            loss_pos = criterion(logits_pos, y_pos)
            dsc_pos = dice_score(logits_pos, y_pos)
            bs_pos = logits_pos.size(0)
            total_loss_pos += loss_pos.item() * bs_pos
            total_dice_pos += dsc_pos.item() * bs_pos
            n_pos += bs_pos

        preds = (torch.sigmoid(logits).view(logits.size(0), -1) > 0.5).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds[neg_mask].sum().item())

    avg_loss = total_loss_pos / max(n_pos, 1)
    avg_dice = total_dice_pos / max(n_pos, 1)
    fpr = fp_slices / max(neg_slices, 1)

    return avg_loss, avg_dice, fpr


def main():
    set_seed(0)
    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = BraTS2D(root=root, 
                 max_patients=50, 
                 only_tumor_slices=False, 
                 neg_to_pos_ratio=3.0, 
                 cache_volumes=True, 
                 seed=1
                 )
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_ch=1, out_ch=1, base_ch=32).to(device)
    criterion = BCEDiceLoss(1.0, 1.0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 30
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        
        running_loss_pos = 0.0
        running_dice_pos = 0.0
        train_pos = 0
        fp_slices = 0
        neg_slices = 0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            with torch.no_grad():
                has_tumor = y.sum(dim=(1, 2, 3)) > 0
                if has_tumor.any():
                    logits_pos = logits[has_tumor]
                    y_pos = y[has_tumor]
                    loss_pos = criterion(logits_pos, y_pos)
                    dsc_pos = dice_score(logits_pos, y_pos)
                    bs_pos = logits_pos.size(0)
                    running_loss_pos += loss_pos.item() * bs_pos
                    running_dice_pos += dsc_pos.item() * bs_pos
                    train_pos += bs_pos

                preds = (torch.sigmoid(logits).view(logits.size(0), -1) > 0.5).any(dim=1)
                neg_mask = ~has_tumor
                if neg_mask.any():
                    neg_slices += int(neg_mask.sum().item())
                    fp_slices += int(preds[neg_mask].sum().item())

        train_loss = running_loss_pos / max(train_pos, 1)
        train_dice = running_dice_pos / max(train_pos, 1)
        train_fpr = fp_slices / max(neg_slices, 1)
        val_loss, val_dice, val_fpr = evaluate(model, val_dl, criterion, device)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_fpr={train_fpr:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_fpr={val_fpr:.4f} | "
            f"time={dt:.1f}s"
        )

    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    ckpt = out / "unet_2d_baseline_seed1.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print("saved:", ckpt)


if __name__ == "__main__":
    main()

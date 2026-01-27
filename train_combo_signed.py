# src/train_combo_signed.py
# Run: python src/train_combo_signed.py

from __future__ import annotations
from pathlib import Path
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from brats2p5d_dataset import BraTS2p5D
from unet import UNet
from losses import dice_score
from boundary_utils import boundary_weight_map  # your existing weight map (distance-to-boundary -> exp)
from boundary_utils_signed import signed_distance_map_2d, normalize_clip_phi
from losses_combo_signed import WeightedBCEDiceSignedBoundaryLoss


def set_seed(seed: int = 0):
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
    fp_slices = 0
    neg_slices = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_np = y.cpu().numpy()  # (B,1,H,W)

        w = np.stack([boundary_weight_map(y_np[i, 0], sigma=3.0, w0=5.0, base=1.0)
                      for i in range(y_np.shape[0])], axis=0)
        phi = np.stack([normalize_clip_phi(signed_distance_map_2d(y_np[i, 0]), clip=10.0)
                        for i in range(y_np.shape[0])], axis=0)

        w = torch.from_numpy(w).unsqueeze(1).to(device, non_blocking=True)
        phi = torch.from_numpy(phi).unsqueeze(1).to(device, non_blocking=True)

        logits = model(x)
        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            w_pos = w[has_tumor]
            phi_pos = phi[has_tumor]
            loss_pos = criterion(logits_pos, y_pos, w_pos, phi_pos)
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

    ds = BraTS2p5D(
        root=root,
        max_patients=10,
        only_tumor_slices=False,
        neg_to_pos_ratio=3.0,
        cache_volumes=True,
        seed=0,
    )

    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_ch=3, out_ch=1, base_ch=32).to(device)

    # Start VERY small to avoid collapse; you can try 0.0005 / 0.001 / 0.002
    criterion = WeightedBCEDiceSignedBoundaryLoss(w_bce=1.0, w_dice=1.0, w_sboundary=0.001).to(device)
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

            y_np = y.cpu().numpy()

            w = np.stack([boundary_weight_map(y_np[i, 0], sigma=3.0, w0=5.0, base=1.0)
                          for i in range(y_np.shape[0])], axis=0)
            phi = np.stack([normalize_clip_phi(signed_distance_map_2d(y_np[i, 0]), clip=10.0)
                            for i in range(y_np.shape[0])], axis=0)

            w = torch.from_numpy(w).unsqueeze(1).to(device, non_blocking=True)
            phi = torch.from_numpy(phi).unsqueeze(1).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y, w, phi)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            with torch.no_grad():
                has_tumor = y.sum(dim=(1, 2, 3)) > 0
                if has_tumor.any():
                    logits_pos = logits[has_tumor]
                    y_pos = y[has_tumor]
                    w_pos = w[has_tumor]
                    phi_pos = phi[has_tumor]
                    loss_pos = criterion(logits_pos, y_pos, w_pos, phi_pos)
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

    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    ckpt_path = out / "unet_2p5d_weighted_plus_signedboundary.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()

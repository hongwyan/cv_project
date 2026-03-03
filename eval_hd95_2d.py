# src/eval_hd95_2d.py
# Run: python src/eval_hd95_2d.py
#
# HD95 evaluation for 2D models (input is 1-channel).
# Uses the SAME val split as training (seed=0, 10% val).

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from brats2d_dataset import BraTS2D
from unet import UNet
from metrics_boundary import hd95_2d
from losses_Dice import dice_score


def load_model(ckpt_path: str, device):
    m = UNet(in_ch=1, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m

@torch.no_grad()
def eval_hd95(model, dl, device):
    hd_vals = []
    dice_total = 0.0
    dice_count = 0
    fp_slices = 0
    neg_slices = 0
    fn_slices = 0
    pos_slices = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        has_tumor = y.sum(dim=(1, 2, 3)) > 0
        if has_tumor.any():
            logits_pos = logits[has_tumor]
            y_pos = y[has_tumor]
            dice_total += dice_score(logits_pos, y_pos).item() * logits_pos.size(0)
            dice_count += logits_pos.size(0)
            pos_slices += int(has_tumor.sum().item())

        thr = 0.7

        # ---- slice-level existence prediction (for FPR/FNR) ----
        preds = (torch.sigmoid(logits).view(logits.size(0), -1) > thr).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds[neg_mask].sum().item())
        if has_tumor.any():
            fn_slices += int((~preds[has_tumor]).sum().item())

        pred = (torch.sigmoid(logits) > thr).float().cpu().numpy()  # (B,1,H,W)
        gt = y.cpu().numpy()  # (B,1,H,W)
        for i in range(pred.shape[0]):
            p = pred[i, 0] > 0.50
            g = gt[i, 0] > 0.50
            # evaluate only when GT exists (tumor slice)
            if g.sum() == 0:
                continue
            hd_vals.append(hd95_2d(p, g))





    hd_vals = np.array(hd_vals, dtype=np.float32)
    hd_vals = hd_vals[np.isfinite(hd_vals)]
    mean_hd = float(hd_vals.mean()) if hd_vals.size else float("nan")
    median_hd = float(np.median(hd_vals)) if hd_vals.size else float("nan")
    fpr = fp_slices / max(neg_slices, 1)
    fnr = fn_slices / max(pos_slices, 1)
    dice = dice_total / max(dice_count, 1)
    return mean_hd, median_hd, int(hd_vals.shape[0]), dice, fpr, fnr

'''@torch.no_grad()
def eval_hd95(model, dl, device):
    vals = []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()  # (B,1,H,W)
        gt = y.numpy()  # (B,1,H,W)
        for i in range(pred.shape[0]):
            p = pred[i, 0] > 0.5
            g = gt[i, 0] > 0.5
            if g.sum() == 0:
                continue
            vals.append(hd95_2d(p, g))
    vals = np.array(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), float(np.median(vals)), int(vals.shape[0])'''


def main():
    '''root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = BraTS2D(root=root, max_patients=50, only_tumor_slices=False, cache_volumes=True, seed=2)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)'''

    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Fixed patient split: use a deterministic subset of patients and evaluate on all slices.
  
    ds = BraTS2D(root=root, max_patients=50, only_tumor_slices=False, cache_volumes=True, seed=3)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Change this to whichever 2D checkpoint you want to evaluate
    ckpt_path = "checkpoints/unet_2d_baseline_seed3.pt"
    ckpt_path2 = "checkpoints/unet_2d_boundary_combo_seed3.pt"

    model = load_model(ckpt_path, device)
    model2 = load_model(ckpt_path2, device)


    mean_h, med_h, n, dice, fpr, fnr = eval_hd95(model, val_dl, device)
    mean_h2, med_h2, n2, dice2, fpr2, fnr2 = eval_hd95(model2, val_dl, device)

    print(f"HD95 (px) over {n} {n2} val slices with tumor:")
    print(f"  {ckpt_path}: mean={mean_h:.3f}, median={med_h:.3f}, dice={dice:.3f}, fpr={fpr:.3f}, fnr={fnr:.3f}")
    print(f"  {ckpt_path2}: mean={mean_h2:.3f}, median={med_h2:.3f}, dice={dice2:.3f}, fpr={fpr2:.3f}, fnr={fnr2:.3f}")


if __name__ == "__main__":
    main()

# src/eval_hd95.py
# Run: python src/eval_hd95.py

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from brats2p5d_dataset import BraTS2p5D
from unet import UNet
from losses import dice_score
from metrics_boundary import hd95_2d


def load_model(ckpt_path, device):
    m = UNet(in_ch=3, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
def eval_metrics(model, dl, device):
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

        preds = (torch.sigmoid(logits).view(logits.size(0), -1) > 0.5).any(dim=1)
        neg_mask = ~has_tumor
        if neg_mask.any():
            neg_slices += int(neg_mask.sum().item())
            fp_slices += int(preds[neg_mask].sum().item())
        if has_tumor.any():
            fn_slices += int((~preds[has_tumor]).sum().item())

        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()  # (B,1,H,W)
        gt = y.cpu().numpy()  # (B,1,H,W)
        for i in range(pred.shape[0]):
            p = pred[i, 0] > 0.5
            g = gt[i, 0] > 0.5
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


def main():
    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Fixed patient split: use a deterministic subset of patients and evaluate on all slices.
    max_patients = 10
    val_fraction = 0.1
    ds = BraTS2p5D(root=root, max_patients=None, only_tumor_slices=False, cache_volumes=True, seed=0)
    patient_dirs = ds.patient_dirs[:max_patients] if max_patients is not None else ds.patient_dirs
    n_val_patients = max(1, int(len(patient_dirs) * val_fraction))
    val_patients = set(patient_dirs[-n_val_patients:])
    val_indices = [i for i, ref in enumerate(ds.slices) if ref.patient_dir in val_patients]
    val_ds = Subset(ds, val_indices)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    base = load_model("checkpoints/unet_2p5d_baseline.pt", device)
    wbd  = load_model("checkpoints/unet_2p5d_boundary.pt", device)
    wpbd = load_model("checkpoints/unet_2p5d_weighted_plus_signedboundary.pt", device)
 

    mean_b, med_b, n, dice_b, fpr_b, fnr_b = eval_metrics(base, val_dl, device)
    mean_w, med_w, _, dice_w, fpr_w, fnr_w = eval_metrics(wbd, val_dl, device)
    mean_bw, med_bw, u, dice_bw, fpr_bw, fnr_bw = eval_metrics(wpbd, val_dl, device)


    print(f"HD95 (px) over {n} val slices with tumor:")
    print(
        f"  baseline: mean={mean_b:.3f}, median={med_b:.3f}, "
        f"dice={dice_b:.3f}, fpr={fpr_b:.3f}, fnr={fnr_b:.3f}"
    )
    print(
        f"  weighted: mean={mean_w:.3f}, median={med_w:.3f}, "
        f"dice={dice_w:.3f}, fpr={fpr_w:.3f}, fnr={fnr_w:.3f}"
    )
    print(
        f"  bw: mean={mean_bw:.3f}, median={med_bw:.3f}, "
        f"dice={dice_bw:.3f}, fpr={fpr_bw:.3f}, fnr={fnr_bw:.3f}"
    )


if __name__ == "__main__":
    main()

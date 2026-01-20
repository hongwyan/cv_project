# src/eval_hd95.py
# Run: python src/eval_hd95.py

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from brats2p5d_dataset import BraTS2p5D
from unet import UNet
from metrics_boundary import hd95_2d


def load_model(ckpt_path, device):
    m = UNet(in_ch=3, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
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
            # evaluate only when GT exists (tumor slice)
            if g.sum() == 0:
                continue
            vals.append(hd95_2d(p, g))
    vals = np.array(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), float(np.median(vals)), int(vals.shape[0])


def main():
    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # IMPORTANT: same split for both models
    ds = BraTS2p5D(root=root, max_patients=10, only_tumor_slices=True, cache_volumes=True, seed=0)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    base = load_model("checkpoints/unet_2p5d_baseline.pt", device)
    wbd  = load_model("checkpoints/unet_2p5d_boundary.pt", device)
    wpbd = load_model("checkpoints/unet_2p5d_weighted_plus_signedboundary.pt", device)
 

    mean_b, med_b, n = eval_hd95(base, val_dl, device)
    mean_w, med_w, _ = eval_hd95(wbd,  val_dl, device)
    mean_bw, med_bw, u = eval_hd95(wpbd, val_dl, device)


    print(f"HD95 (px) over {n} val slices with tumor:")
    print(f"  baseline: mean={mean_b:.3f}, median={med_b:.3f}")
    print(f"  weighted: mean={mean_w:.3f}, median={med_w:.3f}")
    print(f"  bw: mean={mean_bw:.3f}, median={med_bw:.3f}")


if __name__ == "__main__":
    main()

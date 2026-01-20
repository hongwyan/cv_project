# src/visualize_preds.py
# Run: python src/visualize_preds.py

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from brats2p5d_dataset import BraTS2p5D
from unet import UNet


def show(x, y, p, title=""):
    # x: (3,H,W), y/p: (1,H,W)
    x0 = x[1].cpu().numpy()
    y0 = y[0].cpu().numpy()
    p0 = p[0].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x0, cmap="gray"); plt.title("Input (center)"); plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(y0, cmap="gray"); plt.title("GT mask"); plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(p0, cmap="gray"); plt.title("Pred mask"); plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(x0, cmap="gray")
    plt.imshow(y0, alpha=0.35)
    plt.contour(p0, levels=[0.5])
    plt.title("Overlay: GT (fill) + Pred (contour)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    root = r"C:\BraTS_Project\data\BraTS2021_Training_Data"
    ckpt_path = Path("checkpoints/unet_2p5d_weighted_plus_signedboundary.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = BraTS2p5D(root=root, max_patients=10, only_tumor_slices=True, cache_volumes=True, seed=0)

    model = UNet(in_ch=3, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # visualize a few random samples
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=3, replace=False)

    with torch.no_grad():
        for idx in idxs:
            x, y = ds[int(idx)]
            x = x.unsqueeze(0).to(device)   # (1,3,H,W)
            y = y.unsqueeze(0).to(device)   # (1,1,H,W)

            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            show(x[0].cpu(), y[0].cpu(), pred[0].cpu(), title=f"sample idx={int(idx)}")


if __name__ == "__main__":
    main()

# src/step4_build_2p5d_sample.py
# Run: python src/step4_build_2p5d_sample.py

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")  # change if needed


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def robust_zscore(volume: np.ndarray, brain_thr: float = 0.0, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    mask = volume > brain_thr
    if mask.sum() < 100:
        mask = np.ones_like(volume, dtype=bool)

    v = volume[mask]
    lo, hi = np.percentile(v, [p_low, p_high])
    vol = np.clip(volume, lo, hi)

    mu = vol[mask].mean()
    sigma = vol[mask].std() + 1e-8
    return (vol - mu) / sigma


def get_2p5d_indices(t: int, D: int) -> tuple[int, int, int]:
    # clamp edges
    t0 = max(t - 1, 0)
    t1 = t
    t2 = min(t + 1, D - 1)
    return t0, t1, t2


def build_2p5d_sample(flair_norm: np.ndarray, tumor_mask: np.ndarray, t: int):
    D = flair_norm.shape[-1]
    t0, t1, t2 = get_2p5d_indices(t, D)

    x = np.stack(
        [flair_norm[:, :, t0], flair_norm[:, :, t1], flair_norm[:, :, t2]],
        axis=0
    ).astype(np.float32)  # (3, H, W)

    y = tumor_mask[:, :, t1].astype(np.uint8)  # (H, W)
    return x, y, (t0, t1, t2)


def show_triplet_and_mask(x: np.ndarray, y: np.ndarray, idxs, title_prefix=""):
    # x: (3,H,W), y:(H,W)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(x[i], cmap="gray")
        plt.title(f"{title_prefix} slice {idxs[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(x[1], cmap="gray")
    plt.imshow(y, alpha=0.35)
    plt.title(f"{title_prefix} center slice {idxs[1]} + tumor mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    patients = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    p = patients[0]

    flair_path = next(p.glob("*_flair.nii*"))
    seg_path = next(p.glob("*_seg.nii*"))

    flair = load_nifti(flair_path)
    seg = load_nifti(seg_path)
    tumor = (seg > 0).astype(np.uint8)

    flair_n = robust_zscore(flair)

    D = flair.shape[-1]
    # pick a tumor slice if possible, else middle slice
    tumor_pixels_per_slice = tumor.reshape(-1, D).sum(axis=0)
    tumor_slices = np.where(tumor_pixels_per_slice > 0)[0]
    t = int(tumor_slices[len(tumor_slices) // 2]) if len(tumor_slices) > 0 else (D // 2)

    x, y, idxs = build_2p5d_sample(flair_n, tumor, t)

    print("Patient:", p.name)
    print("Chosen t:", t, "Triplet indices:", idxs)
    print("Input x shape:", x.shape, "dtype:", x.dtype)  # (3,H,W)
    print("Label y shape:", y.shape, "dtype:", y.dtype)  # (H,W)
    print("Tumor pixels in y:", int(y.sum()))

    show_triplet_and_mask(x, y, idxs, title_prefix=p.name)


if __name__ == "__main__":
    main()

# src/step3_normalize_and_slice_stats.py
# Run: python src/step3_normalize_and_slice_stats.py

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")  # change if needed


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def robust_zscore(volume: np.ndarray, brain_thr: float = 0.0, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    # Use non-zero region (brain) for robust stats
    mask = volume > brain_thr
    if mask.sum() < 100:  # fallback
        mask = np.ones_like(volume, dtype=bool)

    v = volume[mask]
    lo, hi = np.percentile(v, [p_low, p_high])
    vol = np.clip(volume, lo, hi)

    mu = vol[mask].mean()
    sigma = vol[mask].std() + 1e-8
    return (vol - mu) / sigma


def overlay_show(img2d: np.ndarray, mask2d: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(img2d, cmap="gray")
    plt.imshow(mask2d, alpha=0.35)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    patients = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if not patients:
        raise RuntimeError(f"No patient folders under: {ROOT}")

    # pick one patient for normalization + visualization
    p = patients[0]
    flair_path = next(p.glob("*_flair.nii*"), None)
    seg_path = next(p.glob("*_seg.nii*"), None)
    if flair_path is None or seg_path is None:
        raise RuntimeError(f"Missing flair/seg in: {p}")

    flair = load_nifti(flair_path)
    seg = load_nifti(seg_path)
    tumor = (seg > 0).astype(np.uint8)

    flair_n = robust_zscore(flair)

    # slice-level stats
    D = flair.shape[-1]
    tumor_pixels_per_slice = tumor.reshape(-1, D).sum(axis=0)
    tumor_slices = np.where(tumor_pixels_per_slice > 0)[0]

    print("Patient:", p.name)
    print("Shape:", flair.shape)
    print("Tumor slices:", len(tumor_slices), "/", D)
    if len(tumor_slices) > 0:
        print("First/Last tumor slice:", int(tumor_slices[0]), int(tumor_slices[-1]))

    # visualize: middle slice and one tumor slice (if exists)
    mid = D // 2
    overlay_show(flair_n[:, :, mid], tumor[:, :, mid], f"{p.name} | mid slice {mid} (normalized)")

    if len(tumor_slices) > 0:
        t = int(tumor_slices[len(tumor_slices) // 2])
        overlay_show(flair_n[:, :, t], tumor[:, :, t], f"{p.name} | tumor slice {t} (normalized)")


if __name__ == "__main__":
    main()

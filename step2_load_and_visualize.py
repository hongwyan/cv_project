# src/step2_load_and_visualize.py
# Run: python src/step2_load_and_visualize.py

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")  # <-- change if needed


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


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
        raise RuntimeError(f"No patient folders found under: {ROOT}")

    p = patients[0]
    flair_path = next(p.glob("*_flair.nii*"), None)
    seg_path = next(p.glob("*_seg.nii*"), None)

    if flair_path is None or seg_path is None:
        raise RuntimeError(f"Missing *_flair.nii* or *_seg.nii* in: {p}")

    flair = load_nifti(flair_path)
    seg = load_nifti(seg_path)

    print("Patient:", p.name)
    print("FLAIR:", flair_path.name, "shape:", flair.shape, "dtype:", flair.dtype)
    print("SEG  :", seg_path.name, "shape:", seg.shape, "dtype:", seg.dtype)
    print("SEG unique labels:", np.unique(seg).astype(int))

    tumor = (seg > 0).astype(np.uint8)

    D = flair.shape[-1]
    t = D // 2

    flair_slice = flair[:, :, t]
    tumor_slice = tumor[:, :, t]

    overlay_show(flair_slice, tumor_slice, title=f"{p.name} | slice {t} (FLAIR + tumor mask)")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")
OUT_DIR = Path("visualize")
FIG_SIZE = (6, 6)


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def save_image(img2d: np.ndarray, out_path: Path, mask2d: np.ndarray | None = None, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.imshow(img2d, cmap="gray")
    if mask2d is not None:
        ax.imshow(mask2d, cmap="Reds", alpha=0.35)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_boundary_zoom(img2d: np.ndarray, mask2d: np.ndarray, out_path: Path, title: str = "") -> None:
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0 or xs.size == 0:
        crop_img = img2d
        crop_mask = mask2d
    else:
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        margin = max(8, int(max(y_max - y_min + 1, x_max - x_min + 1) * 0.25))

        y0 = max(0, y_min - margin)
        y1 = min(img2d.shape[0], y_max + margin + 1)
        x0 = max(0, x_min - margin)
        x1 = min(img2d.shape[1], x_max + margin + 1)

        crop_img = img2d[y0:y1, x0:x1]
        crop_mask = mask2d[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.imshow(crop_img, cmap="gray")
    if np.any(crop_mask > 0):
        ax.contour(crop_mask, levels=[0.5], colors="yellow", linewidths=1.5)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_slice(flair: np.ndarray, tumor: np.ndarray) -> int:
    tumor_per_slice = tumor.reshape(-1, tumor.shape[-1]).sum(axis=0)
    tumor_slices = np.where(tumor_per_slice > 0)[0]
    if tumor_slices.size == 0:
        return flair.shape[-1] // 2
    return int(tumor_slices[tumor_slices.size // 2])


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
    tumor = (seg > 0).astype(np.uint8)

    print("Patient:", p.name)
    print("FLAIR:", flair_path.name, "shape:", flair.shape, "dtype:", flair.dtype)
    print("SEG  :", seg_path.name, "shape:", seg.shape, "dtype:", seg.dtype)
    print("SEG unique labels:", np.unique(seg).astype(int))

    t = pick_slice(flair, tumor)
    flair_slice = flair[:, :, t]
    tumor_slice = tumor[:, :, t]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{p.name}_slice{t}"

    save_image(
        flair_slice,
        OUT_DIR / f"{prefix}_tumor_original.png",
        title=f"{p.name} | slice {t} | tumor original",
    )
    save_image(
        flair_slice,
        OUT_DIR / f"{prefix}_groundtruth.png",
        mask2d=tumor_slice,
        title=f"{p.name} | slice {t} | ground truth",
    )
    save_boundary_zoom(
        flair_slice,
        tumor_slice,
        OUT_DIR / f"{prefix}_boundary_zoom.png",
        title=f"{p.name} | slice {t} | boundary zoom",
    )

    print("saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

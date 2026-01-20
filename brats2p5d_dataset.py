# src/brats2p5d_dataset.py
# Minimal BraTS 2021 2.5D (t-1,t,t+1) dataset for binary tumor segmentation on FLAIR.
# Returns: x (FloatTensor [3,H,W]), y (FloatTensor [1,H,W]) in {0,1}
#
# Usage:
#   ds = BraTS2p5D(root=r"C:\BraTS_Project\data\BraTS2021_Training_Data", max_patients=50)
#   x, y = ds[0]
#   print(x.shape, y.shape)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def _load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _robust_zscore(volume: np.ndarray, brain_thr: float = 0.0, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    mask = volume > brain_thr
    if mask.sum() < 100:
        mask = np.ones_like(volume, dtype=bool)

    v = volume[mask]
    lo, hi = np.percentile(v, [p_low, p_high])
    vol = np.clip(volume, lo, hi)

    mu = vol[mask].mean()
    sigma = vol[mask].std() + 1e-8
    return (vol - mu) / sigma


def _clamp_triplet(t: int, D: int) -> Tuple[int, int, int]:
    t0 = max(t - 1, 0)
    t1 = t
    t2 = min(t + 1, D - 1)
    return t0, t1, t2


@dataclass(frozen=True)
class SliceRef:
    patient_dir: Path
    t: int


class BraTS2p5D(Dataset):
    """
    Slice-based BraTS dataset:
      - input: FLAIR slices [t-1, t, t+1] -> (3,H,W)
      - label: binary tumor mask at t -> (1,H,W)
    """

    def __init__(
        self,
        root: str | Path,
        max_patients: Optional[int] = None,
        only_tumor_slices: bool = False,
        cache_volumes: bool = False,
        seed: int = 0,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"BraTS root not found: {self.root}")

        patient_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if max_patients is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(patient_dirs), size=min(max_patients, len(patient_dirs)), replace=False)
            patient_dirs = [patient_dirs[i] for i in sorted(idx)]

        self.patient_dirs: List[Path] = patient_dirs
        self.only_tumor_slices = only_tumor_slices
        self.cache_volumes = cache_volumes

        # Build slice index list
        self.slices: List[SliceRef] = []
        for p in self.patient_dirs:
            flair_path = next(p.glob("*_flair.nii*"), None)
            seg_path = next(p.glob("*_seg.nii*"), None)
            if flair_path is None or seg_path is None:
                continue

            # Read seg only to decide slice indices (cheap enough for week1; can optimize later)
            seg = _load_nifti(seg_path)
            tumor = (seg > 0).astype(np.uint8)
            D = tumor.shape[-1]

            if self.only_tumor_slices:
                tumor_pixels_per_slice = tumor.reshape(-1, D).sum(axis=0)
                ts = np.where(tumor_pixels_per_slice > 0)[0]
                for t in ts.tolist():
                    self.slices.append(SliceRef(p, int(t)))
            else:
                for t in range(D):
                    self.slices.append(SliceRef(p, int(t)))

        if len(self.slices) == 0:
            raise RuntimeError("No slices indexed. Check root path and filenames.")

        # Optional cache: patient_dir -> (flair_norm, tumor_mask)
        self._cache = {} if cache_volumes else None

    def __len__(self) -> int:
        return len(self.slices)

    def _load_patient(self, patient_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache is not None and patient_dir in self._cache:
            return self._cache[patient_dir]

        flair_path = next(patient_dir.glob("*_flair.nii*"))
        seg_path = next(patient_dir.glob("*_seg.nii*"))

        flair = _load_nifti(flair_path)
        seg = _load_nifti(seg_path)
        tumor = (seg > 0).astype(np.uint8)

        flair_n = _robust_zscore(flair)

        if self._cache is not None:
            self._cache[patient_dir] = (flair_n, tumor)
        return flair_n, tumor

    def __getitem__(self, idx: int):
        ref = self.slices[idx]
        flair_n, tumor = self._load_patient(ref.patient_dir)

        H, W, D = flair_n.shape
        t0, t1, t2 = _clamp_triplet(ref.t, D)

        x = np.stack([flair_n[:, :, t0], flair_n[:, :, t1], flair_n[:, :, t2]], axis=0).astype(np.float32)  # (3,H,W)
        y = tumor[:, :, t1].astype(np.float32)[None, ...]  # (1,H,W)

        return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":
    ds = BraTS2p5D(root=r"C:\BraTS_Project\data\BraTS2021_Training_Data", max_patients=10, only_tumor_slices=True)
    x, y = ds[0]
    print("len:", len(ds))
    print("x:", x.shape, x.dtype, "y:", y.shape, y.dtype, "tumor pixels:", int(y.sum().item()))

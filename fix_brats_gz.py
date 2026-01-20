# fix_brats_ext_fast.py
from pathlib import Path

ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")  # your folder with patient dirs

count = 0
for p in ROOT.rglob("*.nii"):
    new_p = p.with_suffix(".nii.gz")
    if not new_p.exists():
        p.rename(new_p)
        count += 1

print(f"Renamed {count} files from .nii to .nii.gz")

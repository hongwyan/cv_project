# src/fix_fake_nii_gz.py
# Run: python src/fix_fake_nii_gz.py

from pathlib import Path

ROOT = Path(r"C:\BraTS_Project\data\BraTS2021_Training_Data")

GZIP_MAGIC = b"\x1f\x8b"

def is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == GZIP_MAGIC
    except OSError:
        return False

def main():
    nii_gz = list(ROOT.rglob("*.nii.gz"))
    fixed = 0
    skipped = 0

    for p in nii_gz:
        if is_gzip(p):
            skipped += 1
            continue

        # Not actually gzipped, but named .nii.gz -> rename to .nii
        new_p = p.with_suffix("")  # removes .gz -> becomes .nii
        # Safety: avoid overwrite
        if new_p.exists():
            print(f"[WARN] target exists, skipping: {p} -> {new_p}")
            continue

        p.rename(new_p)
        fixed += 1

    print(f"Scanned: {len(nii_gz)}")
    print(f"Renamed fake .nii.gz -> .nii: {fixed}")
    print(f"Left real gzip untouched: {skipped}")

if __name__ == "__main__":
    main()

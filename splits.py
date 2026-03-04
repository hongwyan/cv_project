from pathlib import Path

import numpy as np


def _backend(dataset):
    return dataset.ds if hasattr(dataset, "ds") else dataset


def build_patient_split(dataset, seed: int = 0, train_frac: float = 0.8, val_frac: float = 0.1):
    backend = _backend(dataset)
    patient_dirs = list(backend.patient_dirs)
    if len(patient_dirs) < 3:
        raise ValueError("Need at least 3 patients for a train/val/test split.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patient_dirs))
    shuffled = [patient_dirs[i] for i in order]

    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_frac))
    n_val = max(1, int(n_total * val_frac))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1

    train_patients = shuffled[:n_train]
    val_patients = shuffled[n_train:n_train + n_val]
    test_patients = shuffled[n_train + n_val:]
    if not test_patients:
        test_patients = val_patients[-1:]
        val_patients = val_patients[:-1]

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    train_indices = [i for i, ref in enumerate(backend.slices) if ref.patient_dir in train_set]
    val_indices = [i for i, ref in enumerate(backend.slices) if ref.patient_dir in val_set]
    test_indices = [i for i, ref in enumerate(backend.slices) if ref.patient_dir in test_set]

    split = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_patients": [Path(p).name for p in train_patients],
        "val_patients": [Path(p).name for p in val_patients],
        "test_patients": [Path(p).name for p in test_patients],
    }

    if not split["train_indices"] or not split["val_indices"] or not split["test_indices"]:
        raise RuntimeError("Patient-level split produced an empty subset. Adjust max_patients or split ratios.")

    return split

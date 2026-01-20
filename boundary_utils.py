# src/boundary_utils.py  (REPLACE with this version)
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) binary {0,1}
    returns: dist (H,W) distance to GT boundary (0 on boundary), nonnegative
    """
    mask = (mask > 0).astype(bool)
    if mask.sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)

    er = binary_erosion(mask, iterations=1)
    boundary = mask ^ er  # XOR -> boundary pixels
    dist = distance_transform_edt(~boundary)
    return dist.astype(np.float32)


def boundary_weight_map(mask: np.ndarray, sigma: float = 3.0, w0: float = 5.0, base: float = 1.0) -> np.ndarray:
    """
    weight = base + w0 * exp(-(dist^2)/(2*sigma^2))
    High weights near boundary, decays away from boundary.
    """
    dist = distance_to_boundary(mask)
    w = base + w0 * np.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return w.astype(np.float32)

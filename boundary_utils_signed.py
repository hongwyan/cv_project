# src/boundary_utils_signed.py
import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance_map_2d(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) binary {0,1}
    phi > 0 outside, phi < 0 inside, phi = 0 on boundary
    """
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(m, dtype=np.float32)

    dist_out = distance_transform_edt(m == 0)  # outside -> dist to fg
    dist_in  = distance_transform_edt(m == 1)  # inside -> dist to bg
    phi = dist_out - dist_in
    return phi.astype(np.float32)


def normalize_clip_phi(phi: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """
    Clip phi to [-clip, clip] then scale to [-1,1] for stability.
    """
    phi = np.clip(phi, -clip, clip)
    phi = phi / clip
    return phi.astype(np.float32)

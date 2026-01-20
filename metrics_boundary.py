# src/metrics_boundary.py
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist


def _surface_points(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    er = binary_erosion(mask, iterations=1)
    surf = mask ^ er
    pts = np.argwhere(surf)  # (N,2) as (y,x)
    return pts.astype(np.float32)


def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    pred, gt: (H,W) binary {0,1}
    returns HD95 in pixels. If one is empty, returns np.inf (you can skip those in averaging).
    """
    p = _surface_points(pred)
    g = _surface_points(gt)
    if p.shape[0] == 0 or g.shape[0] == 0:
        return np.inf

    d_pg = cdist(p, g).min(axis=1)  # for each p, closest g
    d_gp = cdist(g, p).min(axis=1)  # for each g, closest p
    d = np.concatenate([d_pg, d_gp], axis=0)
    return float(np.percentile(d, 95))

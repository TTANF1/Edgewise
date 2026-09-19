"""Binary mask helpers (Layer 1 primitives)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean mask of values >= `threshold` (used for alpha opacity)."""
    return arr >= threshold


def dilate(mask: np.ndarray) -> np.ndarray:
    """Binary dilation (8-connectivity)."""
    return ndimage.binary_dilation(mask)


def erode(mask: np.ndarray) -> np.ndarray:
    """Binary erosion (8-connectivity)."""
    return ndimage.binary_erosion(mask)

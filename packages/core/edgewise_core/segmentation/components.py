"""Connected components.

Currently unused by the cutout pipeline; provided as a deterministic
primitive for future features such as region pruning and island removal.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected regions of `mask`. Returns (labels, count)."""
    labels, count = ndimage.label(mask)
    return labels, int(count)

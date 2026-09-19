"""Binary mask helpers (Layer 1 primitives)."""

from edgewise_core.mask.background import (
    background_mask,
    color_similar_mask,
    remove_background,
)
from edgewise_core.mask.ops import dilate, erode, threshold

__all__ = [
    "background_mask",
    "color_similar_mask",
    "dilate",
    "erode",
    "remove_background",
    "threshold",
]

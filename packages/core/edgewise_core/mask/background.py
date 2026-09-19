"""Background removal: flood-fill the wall color from image borders.

Given an opaque RGB frame whose subject sits on a flat wall, this module
marks the wall pixels so the pipeline can turn them transparent. Strategy:

1. pixels close to the wall color form a binary mask (tolerance check);
2. connected components of that mask are labeled;
3. any component touching the image border is the background
   (a wall-colored pixel fully enclosed by the subject is kept).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from edgewise_types.candidate import RGB


def color_similar_mask(rgb: np.ndarray, wall_rgb: RGB, tolerance: float) -> np.ndarray:
    """Boolean mask of pixels whose RGB distance from wall_rgb < tolerance."""
    wall = np.asarray(wall_rgb, dtype=np.float64)
    dist = np.sqrt(((rgb.astype(np.float64) - wall) ** 2).sum(axis=2))
    return dist < tolerance


def background_mask(rgb: np.ndarray, wall_rgb: RGB, tolerance: float = 30.0) -> np.ndarray:
    """Detect the background region as above.

    Returns a boolean mask, True on pixels that are wall background.
    """
    similar = color_similar_mask(rgb, wall_rgb, tolerance)
    if not similar.any():
        return np.zeros(rgb.shape[:2], dtype=bool)

    labels, count = ndimage.label(similar)
    if count == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)

    # Labels touching any of the four borders are background.
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels > 0]
    if len(border_labels) == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)

    return np.isin(labels, border_labels)


def remove_background(
    rgb: np.ndarray,
    wall_rgb: RGB,
    tolerance: float = 30.0,
) -> np.ndarray:
    """Build an RGBA array: subject kept, wall pixels set to alpha 0.

    Input `rgb` is HxWx3 (float or uint8); output is HxWx4 float64.
    """
    bg = background_mask(rgb, wall_rgb, tolerance)
    subject = ~bg
    alpha = np.where(subject, 255.0, 0.0)
    return np.dstack([rgb.astype(np.float64), alpha])

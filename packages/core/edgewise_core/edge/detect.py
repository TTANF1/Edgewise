"""Edge detection for cutouts: opaque pixels adjacent to transparency."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from edgewise_core.mask.ops import dilate, threshold


def edge_and_interior_masks(
    alpha: np.ndarray,
    alpha_threshold: float = 128.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition the frame.

    Returns (transparent, opaque, edge, interior) boolean masks:

    - transparent : alpha < threshold
    - opaque      : everything else
    - edge        : opaque pixels adjacent to transparency (8-connectivity)
    - interior    : opaque pixels that are not on the edge
    """
    transparent = ~threshold(alpha, alpha_threshold)
    opaque = ~transparent
    edge = opaque & dilate(transparent)
    interior = opaque & ~edge
    return transparent, opaque, edge, interior


def nearest_interior_indices(
    interior: np.ndarray,
    opaque: np.ndarray | None = None,
    max_distance: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel, the index of the nearest interior pixel (Euclidean).

    Implemented with the distance transform's index map. Pixels inside the
    interior map to themselves. With an opaque mask, distant references and
    references to another component fall back to the pixel itself. Empty
    interiors also return identity indices.
    """
    yy, xx = np.indices(interior.shape)
    if not interior.any():
        return yy, xx
    distance, inds = ndimage.distance_transform_edt(~interior, return_indices=True)
    iy, ix = inds[0].astype(int), inds[1].astype(int)
    if opaque is not None:
        # Thin components without a nearby core keep their own color. Never
        # borrow color from a different disconnected object.
        labels, _ = ndimage.label(opaque, structure=np.ones((3, 3)))
        valid = opaque & (distance <= max_distance) & (labels == labels[iy, ix])
        iy = np.where(valid, iy, yy)
        ix = np.where(valid, ix, xx)
    return iy, ix

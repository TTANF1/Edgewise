"""Local Background Model: per-pixel background color estimation.

Instead of one global wall_rgb, sample the actual background color near
each edge pixel — handles gradients, shadows, and multi-tone backgrounds.
"""
from __future__ import annotations

import numpy as np

from edgewise_types.candidate import RGB


def local_background_sample(
    rgba: np.ndarray,
    edge_mask: np.ndarray,
    interior_mask: np.ndarray,
    sample_radius: int = 3,
) -> np.ndarray:
    """For each edge pixel, estimate the local background color.

    Strategy: walk outward from each edge pixel (away from interior),
    collect background pixels within `sample_radius`, average their color.

    Returns HxWx3 array of local background RGB (0 where no sample found).
    """
    h, w = rgba.shape[:2]
    local_bg = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.int32)

    # For each edge pixel, look at neighbors to find background pixels
    edge_ys, edge_xs = np.where(edge_mask)
    opaque = rgba[:, :, 3] >= 128

    for y, x in zip(edge_ys, edge_xs):
        # Sample a small ring around this pixel
        samples = []
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    # Background pixel = transparent or not opaque
                    if not opaque[ny, nx]:
                        samples.append(rgba[ny, nx, :3])
        if samples:
            avg = np.mean(samples, axis=0)
            local_bg[y, x] = avg
            count[y, x] = len(samples)

    return local_bg


def local_background_distance(
    edge_rgb: RGB,
    local_bg_rgb: np.ndarray,
) -> float:
    """Distance from edge pixel to its local background estimate."""
    edge = np.array(edge_rgb, dtype=np.float64)
    return float(np.sqrt(((edge - local_bg_rgb) ** 2).sum()))

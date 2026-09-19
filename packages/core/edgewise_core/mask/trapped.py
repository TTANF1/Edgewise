"""Detect trapped background pixels inside the subject silhouette.

Flood-fill only removes background connected to the image border. Pixels
trapped between objects (e.g. gap between monitor and wall, chair and desk)
stay opaque but are still wall-colored — they show up as grayish speckles
when composited on a non-wall background.

This module scans all opaque pixels and flags any that are close to the
known wall color, then shrinks their alpha so they blend away.
"""
from __future__ import annotations

import numpy as np

from edgewise_types.candidate import RGB


def detect_trapped_background(
    rgba: np.ndarray,
    wall_rgb: RGB,
    wall_threshold: float = 25.0,
) -> np.ndarray:
    """Return a boolean mask of pixels that look like trapped wall color.

    A pixel is "trapped background" if:
      - it is opaque (alpha >= 128)
      - its RGB is within `wall_threshold` (Euclidean) of `wall_rgb`

    These are pixels flood-fill missed because they're enclosed by subject.
    """
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    wall = np.array(wall_rgb, dtype=np.float64)
    dist = np.sqrt(((rgb - wall) ** 2).sum(axis=2))
    return (alpha >= 128) & (dist < wall_threshold)


def clear_trapped_background(
    rgba: np.ndarray,
    trapped_mask: np.ndarray,
    alpha_decay: float = 0.70,
) -> np.ndarray:
    """Shrink alpha of trapped pixels (they're pure wall, just fade out)."""
    result = rgba.copy()
    result[trapped_mask, 3] = (result[trapped_mask, 3] * (1.0 - alpha_decay)).astype(np.uint8)
    return result

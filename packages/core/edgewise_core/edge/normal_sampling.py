"""Boundary Normal Sampling: analyze both sides of each edge pixel.

For each edge pixel, sample along the boundary normal:
  background side  ←  boundary  →  foreground side

Compare:
  distance(edge, background_side)
  distance(edge, foreground_side)

Edge pixels closer to background side = more likely wall bleed.
"""
from __future__ import annotations

import numpy as np

from edgewise_types.candidate import RGB


def boundary_evidence(
    rgba: np.ndarray,
    edge_mask: np.ndarray,
    interior_mask: np.ndarray,
    sample_radius: int = 2,
) -> dict[tuple[int, int], dict]:
    """For each edge pixel, return boundary evidence.

    Returns {(y, x): {
        "edge_rgb": RGB,
        "bg_side_rgb": RGB,
        "fg_side_rgb": RGB,
        "bg_distance": float,
        "fg_distance": float,
        "local_bg_score": float,  # 0 = foreground, 1 = background
    }}
    """
    h, w = rgba.shape[:2]
    result = {}
    opaque = rgba[:, :, 3] >= 128

    edge_ys, edge_xs = np.where(edge_mask)

    for y, x in zip(edge_ys, edge_xs):
        # Find nearest interior pixel (foreground side direction)
        # Simple: look in 4 directions, find closest opaque pixel
        fg_samples = []
        bg_samples = []

        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if opaque[ny, nx] and not edge_mask[ny, nx]:
                        fg_samples.append(rgba[ny, nx, :3])
                    elif not opaque[ny, nx]:
                        bg_samples.append(rgba[ny, nx, :3])

        if not fg_samples or not bg_samples:
            continue

        fg_avg = np.mean(fg_samples, axis=0)
        bg_avg = np.mean(bg_samples, axis=0)
        edge_rgb = rgba[y, x, :3]

        fg_dist = float(np.sqrt(((edge_rgb - fg_avg) ** 2).sum()))
        bg_dist = float(np.sqrt(((edge_rgb - bg_avg) ** 2).sum()))

        # local_bg_score: 0 = looks like foreground, 1 = looks like background
        total = fg_dist + bg_dist
        local_bg_score = bg_dist / total if total > 0 else 0.5

        result[(int(y), int(x))] = {
            "edge_rgb": tuple(int(v) for v in edge_rgb),
            "bg_side_rgb": tuple(int(v) for v in bg_avg),
            "fg_side_rgb": tuple(int(v) for v in fg_avg),
            "bg_distance": bg_dist,
            "fg_distance": fg_dist,
            "local_bg_score": local_bg_score,
        }

    return result

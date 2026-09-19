"""Rule engine tests (edgewise_core.edge.rules).

Run directly:  python packages/core/tests/test_rules.py
Or via pytest: uv run pytest packages/core/tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from edgewise_core.edge.detect import edge_and_interior_masks, nearest_interior_indices
from edgewise_core.edge.rules import flag_edge_pixels
from edgewise_types.params import DecontaminationParams

PARAMS = DecontaminationParams()
WALL = PARAMS.wall_rgb  # (210, 218, 228)


def _frame(interior_rgb, edge_rgb, size=8):
    """8x8 frame: one interior pixel, 8-connectivity edge around it."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, 3] = 0  # transparent background
    cy = cx = size // 2
    arr[cy, cx, :3] = interior_rgb
    arr[cy, cx, 3] = 255
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue  # keep the center as the interior pixel
            arr[cy + dy, cx + dx, :3] = edge_rgb
            arr[cy + dy, cx + dx, 3] = 255
    return arr


def test_edge_detection_finds_ring():
    arr = _frame((40, 60, 30), (40, 60, 30))
    alpha = arr[:, :, 3].astype(np.float64)
    _, _, edge, interior = edge_and_interior_masks(alpha, PARAMS.alpha_threshold)
    assert interior.sum() == 1  # just the center
    assert edge.sum() == 8  # the ring around it


def test_clean_edge_produces_no_candidate():
    arr = _frame((40, 60, 30), (45, 63, 33))  # tiny difference, below min_color_dist
    r, g, b = (arr[:, :, i].astype(np.float64) for i in range(3))
    alpha = arr[:, :, 3].astype(np.float64)
    _, _, edge, interior = edge_and_interior_masks(alpha, PARAMS.alpha_threshold)
    iy, ix = nearest_interior_indices(interior)
    candidates = flag_edge_pixels(edge, r, g, b, iy, ix, PARAMS)
    assert candidates == []


def test_wall_color_edge_is_forced():
    # Edge is exactly the wall color, interior is dark -> must be forced.
    arr = _frame((40, 60, 30), WALL)
    r, g, b = (arr[:, :, i].astype(np.float64) for i in range(3))
    alpha = arr[:, :, 3].astype(np.float64)
    _, _, edge, interior = edge_and_interior_masks(alpha, PARAMS.alpha_threshold)
    iy, ix = nearest_interior_indices(interior)
    candidates = flag_edge_pixels(edge, r, g, b, iy, ix, PARAMS)
    assert len(candidates) == 8
    assert all(c.force for c in candidates)
    assert all(any("wall=" in reason for reason in c.reasons) for c in candidates)


def test_washed_out_edge_is_flagged():
    # Bright + desaturated edge vs saturated dark interior -> Rule 1 + 2.
    arr = _frame((50, 90, 40), (150, 155, 150))
    r, g, b = (arr[:, :, i].astype(np.float64) for i in range(3))
    alpha = arr[:, :, 3].astype(np.float64)
    _, _, edge, interior = edge_and_interior_masks(alpha, PARAMS.alpha_threshold)
    iy, ix = nearest_interior_indices(interior)
    candidates = flag_edge_pixels(edge, r, g, b, iy, ix, PARAMS)
    assert len(candidates) == 8
    assert all(any("brighter" in reason for reason in c.reasons) for c in candidates)


if __name__ == "__main__":
    test_edge_detection_finds_ring()
    test_clean_edge_produces_no_candidate()
    test_wall_color_edge_is_forced()
    test_washed_out_edge_is_flagged()
    print("test_rules: ok")

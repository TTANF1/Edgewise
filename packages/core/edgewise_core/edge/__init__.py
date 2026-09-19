"""Edge detection and contamination analysis."""

from edgewise_core.edge.detect import edge_and_interior_masks, nearest_interior_indices
from edgewise_core.edge.rules import classify_edge_pixel, flag_edge_pixels

__all__ = [
    "classify_edge_pixel",
    "edge_and_interior_masks",
    "flag_edge_pixels",
    "nearest_interior_indices",
]

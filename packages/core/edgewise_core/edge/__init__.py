"""Edge detection and contamination analysis."""

from edgewise_core.edge.detect import edge_and_interior_masks, nearest_interior_indices
from edgewise_core.edge.local_background import local_background_distance, local_background_sample
from edgewise_core.edge.normal_sampling import boundary_evidence
from edgewise_core.edge.rules import classify_edge_pixel, flag_edge_pixels

__all__ = [
    "boundary_evidence",
    "classify_edge_pixel",
    "edge_and_interior_masks",
    "flag_edge_pixels",
    "local_background_distance",
    "local_background_sample",
    "nearest_interior_indices",
]

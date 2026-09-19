"""Region-level data structures for the Region Analyzer.

Upgrades the pipeline from pixel-by-pixel judgment to region-by-region
judgment: suspicious pixels are grouped into connected components, then
each region gets a feature vector that decides whether it's background
noise or subject detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgewise_types.candidate import RGB


@dataclass(frozen=True, slots=True)
class RegionFeatures:
    """Multi-dimensional feature vector for a candidate region."""

    area: int
    width: int
    height: int
    aspect_ratio: float          # width / height
    border_distance: float      # distance to image edge (min)
    foreground_neighbors: int    # count of opaque neighboring pixels
    background_similarity: float  # 0..1, how close region color is to known wall
    enclosure: float             # 0..1, how enclosed the region is by foreground
    edge_distance: float         # distance to nearest opaque pixel
    local_contrast: float        # contrast vs surrounding foreground
    background_reachability: float  # 0..1, how connected to external background
    narrow_gap_score: float     # 0..1, 1 = likely a 1-3px gap
    isolation_score: float       # 0..1, 1 = isolated island of background color


@dataclass(frozen=True, slots=True)
class CandidateRegion:
    """A connected component of suspicious pixels."""

    id: int
    pixels: np.ndarray          # (N, 2) array of (y, x) coordinates
    area: int
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1) inclusive
    boundary_pixels: np.ndarray  # (M, 2) pixels on the region's edge
    features: RegionFeatures

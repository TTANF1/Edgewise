"""Segmentation (Layer 1): connected components + region analysis."""

from edgewise_core.segmentation.analyzer import detect_regions
from edgewise_core.segmentation.components import connected_components
from edgewise_core.segmentation.region import CandidateRegion, RegionFeatures

__all__ = [
    "CandidateRegion",
    "RegionFeatures",
    "connected_components",
    "detect_regions",
]

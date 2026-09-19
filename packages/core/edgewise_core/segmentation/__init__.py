"""Segmentation (Layer 1): connected components + region analysis."""

from edgewise_core.segmentation.analyzer import detect_regions
from edgewise_core.segmentation.components import connected_components
from edgewise_core.segmentation.confidence import RegionDecision, compute_confidence, decide_region
from edgewise_core.segmentation.region import CandidateRegion, RegionFeatures

__all__ = [
    "CandidateRegion",
    "RegionFeatures",
    "RegionDecision",
    "compute_confidence",
    "connected_components",
    "decide_region",
    "detect_regions",
]

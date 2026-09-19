"""Confidence Engine: deterministic decision for candidate regions.

Combines multiple RegionFeatures into a single confidence score.
High confidence → deterministic decision (no Jev call).
Low confidence → deterministic decision (keep).
Medium confidence → send to Jev for semantic review.
"""
from __future__ import annotations

from dataclasses import dataclass

from edgewise_core.segmentation.region import RegionFeatures


@dataclass(frozen=True, slots=True)
class RegionDecision:
    """Deterministic verdict for a candidate region."""
    decision: str  # "REMOVE", "KEEP", "UNCERTAIN"
    confidence: float  # 0..1, how sure we are
    reason: str


# Weights for each feature (sum to 1.0)
WEIGHTS = {
    "enclosure": 0.25,
    "background_similarity": 0.25,
    "background_reachability": 0.20,
    "narrow_gap_score": 0.15,
    "isolation_score": 0.10,
    "foreground_neighbors": 0.05,
}

# Thresholds for three-tier decision
HIGH_CONFIDENCE = 0.75   # above this → REMOVE
LOW_CONFIDENCE = 0.35    # below this → KEEP
# between → UNCERTAIN (send to Jev)


def compute_confidence(features: RegionFeatures) -> float:
    """Weighted sum of features → 0..1 confidence that region is background."""
    score = (
        WEIGHTS["enclosure"] * features.enclosure
        + WEIGHTS["background_similarity"] * features.background_similarity
        + WEIGHTS["background_reachability"] * (1.0 - features.background_reachability)
        + WEIGHTS["narrow_gap_score"] * features.narrow_gap_score
        + WEIGHTS["isolation_score"] * features.isolation_score
        + WEIGHTS["foreground_neighbors"] * min(features.foreground_neighbors / 10.0, 1.0)
    )
    return float(score)


def decide_region(features: RegionFeatures) -> RegionDecision:
    """Three-tier deterministic decision for a candidate region."""
    confidence = compute_confidence(features)

    if confidence >= HIGH_CONFIDENCE:
        return RegionDecision(
            decision="REMOVE",
            confidence=confidence,
            reason=f"high confidence ({confidence:.2f}): enclosure={features.enclosure:.2f}, bg_sim={features.background_similarity:.2f}",
        )
    elif confidence <= LOW_CONFIDENCE:
        return RegionDecision(
            decision="KEEP",
            confidence=confidence,
            reason=f"low confidence ({confidence:.2f}): likely subject detail",
        )
    else:
        return RegionDecision(
            decision="UNCERTAIN",
            confidence=confidence,
            reason=f"medium confidence ({confidence:.2f}): needs Jev review",
        )

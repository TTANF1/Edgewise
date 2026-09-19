"""Edgewise Layer 2 — semantic review.

The pipeline only depends on `SemanticReviewer`; concrete backends are
adapters behind it. Swapping Jev for another model means adding one
adapter here — the graphics layer never changes.
"""

from edgewise_semantic.jev.adapter import JevReviewer

__all__ = ["JevReviewer"]

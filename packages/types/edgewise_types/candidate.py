"""Data contract between the graphics layer and the semantic layer."""
from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EdgeCandidate:
    """An edge pixel worth a second look.

    Produced by the deterministic rule engine (Layer 1) and consumed by a
    semantic reviewer (Layer 2).
    """

    y: int
    x: int
    edge_rgb: RGB
    interior_rgb: RGB
    reasons: tuple[str, ...] = field(default_factory=tuple)
    bright_diff: float = 0.0
    color_dist: float = 0.0
    force: bool = False  # rules are already certain — no semantic review needed

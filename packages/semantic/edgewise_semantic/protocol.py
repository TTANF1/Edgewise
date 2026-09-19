"""Semantic layer contract."""
from __future__ import annotations

from typing import Protocol, Sequence

from edgewise_types.candidate import EdgeCandidate, RGB


class SemanticReviewer(Protocol):
    """Reviews candidate edge pixels and scores wall-bleed probability.

    Implementations never mutate the candidates; they return one
    probability per *color* so repeated pixels share a verdict.
    """

    def review(self, candidates: Sequence[EdgeCandidate]) -> dict[RGB, float]:
        """Return {edge_rgb: contamination probability in [0, 1]}.

        Colors absent from the result are treated as UNCERTAIN by the
        pipeline (see DecontaminationParams.jev_uncertain_floor).
        """
        ...

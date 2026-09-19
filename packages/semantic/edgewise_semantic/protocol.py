"""Semantic layer contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from edgewise_types.candidate import EdgeCandidate, RGB


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Jev's verdict on a decontaminated frame."""
    quality_score: float       # 0 = terrible, 1 = perfect
    needs_more_iterations: bool  # True if halo still visible
    issues: list[str]          # e.g. ["halo on head", "gap between monitor and wall"]


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

    def confirm_background_color(
        self,
        candidates: list[RGB],
        interior_sample: RGB,
    ) -> RGB | None:
        """Identify the true background wall color from corner samples.

        Returns the confirmed wall_rgb, or None if the model cannot decide.
        """
        ...

    def evaluate_quality(
        self,
        frame_path: str,
        wall_rgb: RGB,
    ) -> QualityReport:
        """Judge whether a decontaminated frame is clean enough."""
        ...

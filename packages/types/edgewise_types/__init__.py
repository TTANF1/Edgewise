"""Shared contracts for the edgewise pipeline.

This package is a dependency-free leaf: both the graphics layer
(`edgewise-core`) and the semantic layer (`edgewise-semantic`) import
from here, and neither ever imports the other.
"""

from edgewise_types.candidate import EdgeCandidate, RGB
from edgewise_types.decision import Decision, decision_from_probability
from edgewise_types.params import DecontaminationParams

__all__ = [
    "Decision",
    "DecontaminationParams",
    "EdgeCandidate",
    "RGB",
    "decision_from_probability",
]

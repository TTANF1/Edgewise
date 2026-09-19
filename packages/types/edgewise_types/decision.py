"""Semantic decision vocabulary shared across layers.

Layer 2 (semantic models) reduces each candidate edge pixel to one of
three decisions, so downstream layers never depend on which model made
the call.
"""
from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    """Verdict for a candidate edge pixel after semantic review."""

    KEEP = "keep"            # subject color — leave untouched
    REMOVE = "remove"        # background bleed — pull toward interior
    UNCERTAIN = "uncertain"  # model is not confident either way


def decision_from_probability(
    probability: float,
    remove_threshold: float,
    uncertain_floor: float,
) -> Decision:
    """Map a model's contamination probability to a decision.

    `probability` is the model's estimate that the pixel is wall/background
    bleed. Above `remove_threshold` the verdict is REMOVE; below
    `uncertain_floor` it is KEEP; anything in between is UNCERTAIN.
    """
    if probability > remove_threshold:
        return Decision.REMOVE
    if probability < uncertain_floor:
        return Decision.KEEP
    return Decision.UNCERTAIN

"""Decision mapping tests (edgewise_types).

Run directly:  python packages/types/tests/test_decision.py
Or via pytest: uv run pytest packages/types/tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgewise_types.decision import Decision, decision_from_probability


def test_decision_remove_above_threshold():
    assert decision_from_probability(0.41, 0.40, 0.30) is Decision.REMOVE
    assert decision_from_probability(0.99, 0.40, 0.30) is Decision.REMOVE


def test_decision_keep_below_floor():
    assert decision_from_probability(0.29, 0.40, 0.30) is Decision.KEEP
    assert decision_from_probability(0.0, 0.40, 0.30) is Decision.KEEP


def test_decision_uncertain_band():
    assert decision_from_probability(0.35, 0.40, 0.30) is Decision.UNCERTAIN
    assert decision_from_probability(0.30, 0.40, 0.30) is Decision.UNCERTAIN


if __name__ == "__main__":
    test_decision_remove_above_threshold()
    test_decision_keep_below_floor()
    test_decision_uncertain_band()
    print("test_decision: ok")

"""Prompt construction for the Jev model (TypeSafe System One).

Kept separate from the adapter so prompts can be tuned and tested without
an API round-trip, and so a different model can ship its own prompt.
"""
from __future__ import annotations

from edgewise_types.candidate import RGB


def _lum(r: float, g: float, b: float) -> int:
    # Local copy of the Rec.601 luminance — the semantic package stays
    # free of graphics-layer imports by design.
    return int(0.299 * r + 0.587 * g + 0.114 * b)


def build_questions(
    colors: list[tuple[RGB, RGB]],
    wall_rgb: RGB,
    scene_hint: str,
) -> dict[str, dict]:
    """Build one TypeSafe `noul` question per unique edge color.

    `colors` is a list of (edge_rgb, interior_rgb) pairs, already
    deduplicated by the adapter.
    """
    questions: dict[str, dict] = {}
    for i, (edge_rgb, interior_rgb) in enumerate(colors):
        r, g, b = edge_rgb
        ir, ig, ib = interior_rgb
        questions[f"c{i}"] = {
            "type": "noul",
            "instructions": (
                f"Pixel-art scene: {scene_hint}. Wall is light grayish-blue RGB{wall_rgb}. "
                f"Edge pixel RGB({r},{g},{b}) (brightness {_lum(r, g, b)}) borders interior "
                f"pixel RGB({ir},{ig},{ib}) (brightness {_lum(ir, ig, ib)}). "
                f"The interior is the actual subject color (dark leaf, clothing, or object). "
                f"Is the edge pixel contaminated by wall background bleed — i.e., washed out, "
                f"lightened, or grayed toward the wall — such that it should be darkened and "
                f"pulled toward the interior color?"
            ),
        }
    return questions


def build_wall_questions(
    candidates: list[RGB],
    interior_sample: RGB,
    scene_hint: str,
) -> dict[str, dict]:
    """Ask Jev which candidate color is the actual background wall.

    `candidates` are corner-sampled colors; `interior_sample` is a known
    subject pixel for contrast.
    """
    ir, ig, ib = interior_sample
    questions: dict[str, dict] = {}
    for i, (r, g, b) in enumerate(candidates):
        questions[f"w{i}"] = {
            "type": "noul",
            "instructions": (
                f"Pixel-art scene: {scene_hint}. We sampled several colors from the image "
                f"corners to identify the background wall. Candidate RGB({r},{g},{b}) "
                f"(brightness {_lum(r, g, b)}). Known subject interior is RGB({ir},{ig},{ib}) "
                f"(brightness {_lum(ir, ig, ib)}). Is this candidate the actual flat background "
                f"wall color (not a subject pixel, not shadow, not gradient noise)?"
            ),
        }
    return questions

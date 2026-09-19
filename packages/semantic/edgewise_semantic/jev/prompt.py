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

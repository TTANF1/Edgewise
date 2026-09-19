"""Color math: RGB<->HSL, luminance, distance, blending.

Everything here is deterministic and depends only on numpy.
"""
from __future__ import annotations

import numpy as np

from edgewise_types.candidate import RGB


def rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert one RGB pixel (0-255) to (hue, saturation, lightness)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h * 60.0, s, l


def luminance(r: float, g: float, b: float) -> float:
    """Perceived luminance (Rec. 601 weights), 0-255 scale."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def color_distance(rgb_a: RGB, rgb_b: RGB) -> float:
    """Euclidean distance in RGB space."""
    return float(np.linalg.norm(np.asarray(rgb_a, dtype=float) - np.asarray(rgb_b, dtype=float)))


def blend_toward(edge: RGB, interior: RGB, t: float) -> RGB:
    """Linear pull of `edge` toward `interior` by factor t in [0, 1].

    Channel values are truncated with int(), matching the original tool.
    """
    return tuple(int(e * (1 - t) + i * t) for e, i in zip(edge, interior))  # type: ignore[return-value]


def wall_contamination_alphas(edge: RGB, interior: RGB, wall: RGB) -> list[float]:
    """Per-channel reconstruction of `alpha` in edge = alpha*interior + (1-alpha)*wall.

    Channels whose interior/wall difference is too small are skipped
    (the equation is ill-conditioned there).
    """
    alphas: list[float] = []
    for ec, ic, wc in zip(edge, interior, wall):
        denom = wc - ic
        if abs(denom) > 5:
            alphas.append(float(np.clip((wc - ec) / denom, 0.0, 1.0)))
    return alphas

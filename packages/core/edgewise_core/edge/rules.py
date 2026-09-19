"""Deterministic contamination rules (Layer 1, no API).

The rules turn raw edge/interior color pairs into EdgeCandidate objects.
Candidates with `force=True` are considered certain and skip semantic
review; the rest go to Layer 2.
"""
from __future__ import annotations

import numpy as np

from edgewise_core.pixel.color import (
    color_distance,
    luminance,
    rgb_to_hsl,
    wall_contamination_alphas,
)
from edgewise_types.candidate import EdgeCandidate, RGB
from edgewise_types.params import DecontaminationParams


def classify_edge_pixel(
    y: int,
    x: int,
    edge_rgb: RGB,
    interior_rgb: RGB,
    params: DecontaminationParams,
) -> EdgeCandidate | None:
    """Run the three wall-bleed rules on one pixel.

    Returns None when the pixel looks clean (no rule fires).
    """
    dr, dg, db = (e - i for e, i in zip(edge_rgb, interior_rgb))
    if (dr * dr + dg * dg + db * db) ** 0.5 < params.min_color_dist:
        return None

    _, es, _ = rgb_to_hsl(*edge_rgb)
    _, is_, _ = rgb_to_hsl(*interior_rgb)
    bright_diff = luminance(*edge_rgb) - luminance(*interior_rgb)

    reasons: list[str] = []
    force = False

    # Rule 1: edge significantly lighter than interior -> wall bleeds in
    if bright_diff > params.bright_diff_threshold:
        reasons.append(f"brighter(+{int(bright_diff)})")
        if es < is_ - params.rule1_sat_delta and is_ > params.rule1_sat_min:
            force = True

    # Rule 2: edge desaturated vs interior -> washed out
    if es < is_ - params.rule2_sat_delta and is_ > params.rule2_sat_min:
        reasons.append(f"desaturated({is_:.2f}->{es:.2f})")
        if bright_diff > params.rule2_force_bright_diff:
            force = True

    # Rule 3: linear wall-contamination model
    alphas = wall_contamination_alphas(edge_rgb, interior_rgb, params.wall_rgb)
    if alphas:
        wall_pct = 1.0 - float(np.mean(alphas))
        if wall_pct > params.wall_pct_reason:
            reasons.append(f"wall={wall_pct:.0%}")
            if wall_pct > params.wall_pct_force:
                force = True

    if not reasons:
        return None

    return EdgeCandidate(
        y=y,
        x=x,
        edge_rgb=edge_rgb,
        interior_rgb=interior_rgb,
        reasons=tuple(reasons),
        bright_diff=float(bright_diff),
        color_dist=float(color_distance(edge_rgb, interior_rgb)),
        force=force,
    )


def flag_edge_pixels(
    edge: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    params: DecontaminationParams,
) -> list[EdgeCandidate]:
    """Run the rules over every edge pixel.

    `iy`/`ix` map each pixel to its nearest interior pixel (see
    `nearest_interior_indices`).
    """
    candidates: list[EdgeCandidate] = []
    for y, x in np.argwhere(edge):
        cand = classify_edge_pixel(
            int(y),
            int(x),
            (int(r[y, x]), int(g[y, x]), int(b[y, x])),
            (
                int(r[iy[y, x], ix[y, x]]),
                int(g[iy[y, x], ix[y, x]]),
                int(b[iy[y, x], ix[y, x]]),
            ),
            params,
        )
        if cand is not None:
            candidates.append(cand)
    return candidates

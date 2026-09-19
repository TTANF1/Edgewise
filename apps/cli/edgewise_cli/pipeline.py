"""Cutout decontamination pipeline.

Orchestrates the three layers for one frame:

    Layer 1 (deterministic)  -> masks, candidates
    Layer 2 (semantic)       -> contamination probability per color
    Layer 1 (deterministic)  -> blend map, final pixels
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from edgewise_core.edge.detect import edge_and_interior_masks, nearest_interior_indices
from edgewise_core.edge.rules import flag_edge_pixels
from edgewise_core.mask.background import remove_background
from edgewise_core.pixel.color import blend_toward
from edgewise_semantic.protocol import SemanticReviewer
from edgewise_types.candidate import EdgeCandidate, RGB
from edgewise_types.decision import Decision, decision_from_probability
from edgewise_types.params import DecontaminationParams

Masks = dict[str, np.ndarray]
Analysis = tuple[Image.Image, list[EdgeCandidate], Masks]


def analyze_frame(img_path: str, params: DecontaminationParams) -> Analysis:
    """Layer 1: masks, nearest interior, and rule-based candidates. No API."""
    opened = Image.open(img_path)
    if params.background_tolerance is not None:
        # Opaque wall-backed frame: flood-fill the wall to transparency first.
        rgb = np.asarray(opened.convert("RGB")).astype(np.float64)
        rgba = remove_background(rgb, params.wall_rgb, params.background_tolerance)
        im = Image.fromarray(rgba.astype(np.uint8))
    else:
        im = opened.convert("RGBA")
    arr = np.asarray(im).astype(np.float64)
    r, g, b, alpha = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    _, _, edge, interior = edge_and_interior_masks(alpha, params.alpha_threshold)
    iy, ix = nearest_interior_indices(interior)

    candidates = flag_edge_pixels(edge, r, g, b, iy, ix, params)

    masks: Masks = {"edge": edge, "iy": iy, "ix": ix}
    return im, candidates, masks


def build_blend_map(
    edge: np.ndarray,
    candidates: list[EdgeCandidate],
    reviews: dict[RGB, float],
    params: DecontaminationParams,
) -> dict[tuple[int, int], float]:
    """Decide the pull strength for every edge pixel.

    - every edge pixel            : base_blend
    - rule-confirmed (force)      : force_blend
    - semantic REMOVE             : base_blend + scaled extra (0..jev_max_extra_blend)
    """
    blend: dict[tuple[int, int], float] = {
        (int(y), int(x)): params.base_blend for y, x in np.argwhere(edge)
    }
    for c in candidates:
        key = (c.y, c.x)
        if c.force:
            blend[key] = params.force_blend
            continue
        probability = reviews.get(c.edge_rgb, params.jev_uncertain_floor)
        if probability > params.jev_threshold:
            extra = (
                (probability - params.jev_threshold)
                / (1.0 - params.jev_threshold)
                * params.jev_max_extra_blend
            )
            blend[key] = max(blend[key], extra)
    return blend


def apply_decontamination(
    arr: np.ndarray,
    blend: dict[tuple[int, int], float],
    iy: np.ndarray,
    ix: np.ndarray,
    params: DecontaminationParams,
) -> np.ndarray:
    """Pull edge RGB toward the nearest interior pixel and shrink alpha."""
    result = arr.copy()
    for (y, x), b in blend.items():
        edge_rgb = (int(result[y, x, 0]), int(result[y, x, 1]), int(result[y, x, 2]))
        interior_rgb = (
            int(result[iy[y, x], ix[y, x], 0]),
            int(result[iy[y, x], ix[y, x], 1]),
            int(result[iy[y, x], ix[y, x], 2]),
        )
        nr, ng, nb = blend_toward(edge_rgb, interior_rgb, b)
        result[y, x, 0], result[y, x, 1], result[y, x, 2] = nr, ng, nb
        result[y, x, 3] = int(result[y, x, 3] * (1.0 - b * params.alpha_decay))
    return result


def decontaminate(
    img_path: str,
    out_path: str,
    params: DecontaminationParams,
    reviewer: SemanticReviewer | None = None,
    skip_jev: bool = False,
) -> dict[str, Any]:
    """Full pipeline for one frame. Returns stats for reporting."""
    im, candidates, masks = analyze_frame(img_path, params)
    edge = masks["edge"]
    print(
        f"  edge: {int(edge.sum())}, flagged: {len(candidates)}, "
        f"force: {sum(1 for c in candidates if c.force)}"
    )

    reviews: dict[RGB, float] = {}
    if candidates and not skip_jev:
        if reviewer is None:
            raise ValueError("a reviewer is required unless --skip-jev is set")
        reviews = reviewer.review(candidates)
        confirmed = sum(
            1
            for p in reviews.values()
            if decision_from_probability(p, params.jev_threshold, params.jev_uncertain_floor)
            is Decision.REMOVE
        )
        print(f"  Jev confirmed: {confirmed} colors")
        for rgb, p in sorted(reviews.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    RGB{rgb} -> {p:.2f}")

    blend = build_blend_map(edge, candidates, reviews, params)
    arr = np.asarray(im).astype(np.float64)
    result = apply_decontamination(arr, blend, masks["iy"], masks["ix"], params)
    Image.fromarray(result.astype(np.uint8)).save(out_path)
    print(f"  tightened {len(blend)} edge px")
    return {
        "edge_px": int(edge.sum()),
        "flagged": len(candidates),
        "force": sum(1 for c in candidates if c.force),
        "reviewed": len(reviews),
    }

"""Cutout decontamination pipeline.

Orchestrates the three layers for one frame:

    Layer 1 (deterministic)  -> masks, candidates
    Layer 2 (semantic)       -> contamination probability per color
    Layer 1 (deterministic)  -> blend map, final pixels
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from edgewise_core.edge.detect import edge_and_interior_masks, nearest_interior_indices
from edgewise_core.edge.rules import flag_edge_pixels
from edgewise_core.mask.background import remove_background
from edgewise_core.pixel.analyze import analyze_input
from edgewise_core.pixel.color import blend_toward
from edgewise_semantic.protocol import SemanticReviewer
from edgewise_types.candidate import EdgeCandidate, RGB
from edgewise_types.decision import Decision, decision_from_probability
from edgewise_types.params import DecontaminationParams

Masks = dict[str, np.ndarray]
Analysis = tuple[Image.Image, list[EdgeCandidate], Masks]


def analyze_frame(
    img_path: str,
    params: DecontaminationParams,
    reviewer: SemanticReviewer | None = None,
    skip_jev: bool = False,
) -> Analysis:
    """Layer 1: Step 0 input analysis + masks, candidates. No API."""
    opened = Image.open(img_path)
    rgba_init = np.asarray(opened.convert("RGBA")).astype(np.float64)

    # Step 0: auto-detect frame properties (opaque? wall color?)
    info = analyze_input(rgba_init)
    if info.needs_background_removal and params.background_tolerance is None:
        # Auto-enable background removal with detected wall color
        params = replace(
            params,
            background_tolerance=info.suggested_tolerance,
            wall_rgb=info.wall_rgb,
        )
        print(f"  [auto] opaque frame detected, wall_rgb={info.wall_rgb}, tolerance={info.suggested_tolerance:.0f}")

    # Step 0b: Jev wall-color confirmation when auto-detection is uncertain
    if (
        reviewer is not None
        and not skip_jev
        and info.needs_background_removal
        and info.wall_std > 15.0  # corner colors disagree → ask Jev
    ):
        from edgewise_core.pixel.analyze import _sample_border_colors
        samples = _sample_border_colors(rgba_init[:, :, :3])
        sorted_by_lum = sorted(
            [tuple(v) for v in samples],
            key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2],
        )
        candidates = sorted_by_lum[::max(1, len(sorted_by_lum) // 3)][:3]
        h, w = rgba_init.shape[:2]
        center = rgba_init[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5, :3].mean(axis=(0, 1))
        interior_sample = tuple(int(v) for v in center)

        confirmed = reviewer.confirm_background_color(candidates, interior_sample)
        if confirmed:
            print(f"  [jev] wall_rgb confirmed: {info.wall_rgb} -> {confirmed}")
            params = replace(params, wall_rgb=confirmed)
        else:
            print(f"  [jev] wall_rgb uncertain, keeping {info.wall_rgb}")

    if params.background_tolerance is not None:
        # Opaque wall-backed frame: flood-fill the wall to transparency first.
        rgb = rgba_init[:, :, :3]
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
    im, candidates, masks = analyze_frame(img_path, params, reviewer, skip_jev)
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

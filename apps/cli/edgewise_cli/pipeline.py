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

        # Region Analyzer: detect trapped background regions and decide
        # which ones to remove (replaces simple pixel-distance threshold).
        from edgewise_core.segmentation.analyzer import detect_regions
        from edgewise_core.mask.trapped import detect_trapped_background

        trapped = detect_trapped_background(rgba, params.wall_rgb)
        if trapped.any():
            alpha_px = rgba[:, :, 3]
            rgb_px = rgba[:, :, :3]
            regions = detect_regions(trapped, alpha_px, rgb_px, params.wall_rgb)
            removed_regions = 0
            removed_px = 0
            for region in regions:
                f = region.features
                # Deterministic decision: high enclosure + high bg similarity
                # + low reachability = definitely trapped background gap
                is_trapped_gap = (
                    f.enclosure > 0.7
                    and f.background_similarity > 0.7
                    and f.background_reachability < 0.2
                )
                if is_trapped_gap:
                    for y, x in region.pixels:
                        rgba[int(y), int(x), 3] = int(rgba[int(y), int(x), 3] * 0.5)
                    removed_regions += 1
                    removed_px += region.area
            print(f"  regions: {len(regions)}, trapped gaps removed: {removed_regions} ({removed_px} px)")

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
    - rule-confirmed (force)      : force_blend, unless Jev says subject detail
    - semantic REMOVE             : base_blend + scaled extra (0..jev_max_extra_blend)
    """
    blend: dict[tuple[int, int], float] = {
        (int(y), int(x)): params.base_blend for y, x in np.argwhere(edge)
    }
    for c in candidates:
        key = (c.y, c.x)
        probability = reviews.get(c.edge_rgb)
        if c.force:
            # Jev can downgrade force pixels that are actually subject detail.
            if probability is not None and probability < params.jev_uncertain_floor:
                blend[key] = params.base_blend  # protect: don't over-pull
            else:
                blend[key] = params.force_blend
            continue
        if probability is not None and probability > params.jev_threshold:
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
    iterations: int = 3,
) -> dict[str, Any]:
    """Full pipeline for one frame, iterated. Returns stats for reporting.

    Each iteration feeds the previous output back as input — residual halo
    pixels that were semi-transparent last round become edge pixels this
    round and get tightened further. Jev quality check decides when to stop.
    """
    current_path = img_path
    last_edge_px = -1
    quality_history: list[float] = []

    for it in range(iterations):
        im, candidates, masks = analyze_frame(current_path, params, reviewer, skip_jev)
        edge = masks["edge"]
        edge_px = int(edge.sum())
        print(
            f"  [iter {it+1}/{iterations}] edge: {edge_px}, flagged: {len(candidates)}, "
            f"force: {sum(1 for c in candidates if c.force)}"
        )

        # Convergence check: edge px barely changed → stop early
        if it > 0 and abs(edge_px - last_edge_px) < max(5, edge_px * 0.03):
            print(f"  converged at iter {it+1} (edge px stable)")
            reviews: dict[RGB, float] = {}
            blend = build_blend_map(edge, candidates, reviews, params)
            arr = np.asarray(im).astype(np.float64)
            result = apply_decontamination(arr, blend, masks["iy"], masks["ix"], params)
            Image.fromarray(result.astype(np.uint8)).save(out_path)
            print(f"  tightened {len(blend)} edge px")
            break
        last_edge_px = edge_px

        reviews: dict[RGB, float] = {}
        if candidates and not skip_jev:
            if reviewer is None:
                raise ValueError("a reviewer is required unless --skip-jev is set")
            reviews = reviewer.review(candidates)

        blend = build_blend_map(edge, candidates, reviews, params)
        arr = np.asarray(im).astype(np.float64)
        result = apply_decontamination(arr, blend, masks["iy"], masks["ix"], params)

        if it == iterations - 1:
            Image.fromarray(result.astype(np.uint8)).save(out_path)
            current_path = out_path
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(result.astype(np.uint8)).save(tmp.name)
            current_path = tmp.name

        print(f"  tightened {len(blend)} edge px")

        # Jev quality evaluation: stop early if clean enough
        if reviewer is not None and not skip_jev and it < iterations - 1:
            report = reviewer.evaluate_quality(current_path, params.wall_rgb)
            quality_history.append(report.quality_score)
            print(f"  [jev] quality={report.quality_score:.2f}, issues: {report.issues or 'none'}")
            if not report.needs_more_iterations:
                print(f"  Jev says clean enough at iter {it+1}")
                break

    return {
        "edge_px": edge_px,
        "iterations": it + 1,
        "quality_history": quality_history,
    }

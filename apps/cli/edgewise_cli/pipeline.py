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
from edgewise_semantic.protocol import SemanticReviewer
from edgewise_types.candidate import EdgeCandidate, RGB
from edgewise_types.params import DecontaminationParams

Masks = dict[str, Any]
Analysis = tuple[Image.Image, list[EdgeCandidate], Masks]


def analyze_frame(
    img_path: str | Image.Image,
    params: DecontaminationParams,
    reviewer: SemanticReviewer | None = None,
    skip_jev: bool = False,
    *,
    preprocess: bool = True,
) -> Analysis:
    """Analyze masks/candidates, optionally preparing the background once."""
    if isinstance(img_path, Image.Image):
        opened = img_path.convert("RGBA")
    else:
        with Image.open(img_path) as source:
            opened = source.convert("RGBA")
    rgba_init = np.asarray(opened).astype(np.float64)

    # Step 0: auto-detect frame properties (opaque? wall color?)
    info = analyze_input(rgba_init)
    if preprocess and info.needs_background_removal and params.background_tolerance is None:
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
        and preprocess
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
        center = rgba_init[max(0, h // 2 - 5):h // 2 + 5, max(0, w // 2 - 5):w // 2 + 5, :3].mean(axis=(0, 1))
        interior_sample = tuple(int(v) for v in center)

        confirmed = reviewer.confirm_background_color(candidates, interior_sample)
        if confirmed:
            print(f"  [jev] wall_rgb confirmed: {info.wall_rgb} -> {confirmed}")
            params = replace(params, wall_rgb=confirmed)
        else:
            print(f"  [jev] wall_rgb uncertain, keeping {info.wall_rgb}")

    if preprocess and params.background_tolerance is not None:
        # Opaque wall-backed frame: flood-fill the wall to transparency first.
        rgb = rgba_init[:, :, :3]
        rgba = remove_background(rgb, params.wall_rgb, params.background_tolerance)
        rgba[:, :, 3] = np.minimum(rgba[:, :, 3], rgba_init[:, :, 3])

        # Region Analyzer + Confidence Engine: three-tier decision per region
        # (REMOVE / KEEP / UNCERTAIN). Uncertain regions are retained until
        # a region-review backend is implemented.
        from edgewise_core.segmentation.analyzer import detect_regions
        from edgewise_core.segmentation.confidence import decide_region
        from edgewise_core.mask.trapped import detect_trapped_background

        trapped = detect_trapped_background(rgba, params.wall_rgb)
        if trapped.any():
            alpha_px = rgba[:, :, 3]
            rgb_px = rgba[:, :, :3]
            regions = detect_regions(trapped, alpha_px, rgb_px, params.wall_rgb)
            removed_regions = 0
            removed_px = 0
            uncertain_regions = 0
            for region in regions:
                decision = decide_region(region.features)
                if decision.decision == "REMOVE":
                    for y, x in region.pixels:
                        rgba[int(y), int(x), 3] = int(rgba[int(y), int(x), 3] * 0.5)
                    removed_regions += 1
                    removed_px += region.area
                elif decision.decision == "UNCERTAIN":
                    uncertain_regions += 1
            print(f"  regions: {len(regions)}, REMOVE: {removed_regions} ({removed_px} px), UNCERTAIN: {uncertain_regions}")

        im = Image.fromarray(rgba.astype(np.uint8))
    else:
        im = opened.convert("RGBA")
    arr = np.asarray(im).astype(np.float64)
    r, g, b, alpha = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    _, opaque, edge, interior = edge_and_interior_masks(alpha, params.alpha_threshold)
    iy, ix = nearest_interior_indices(interior, opaque)
    yy, xx = np.indices(edge.shape)
    edge &= (iy != yy) | (ix != xx)

    candidates = flag_edge_pixels(edge, r, g, b, iy, ix, params)

    # Boundary Normal Sampling: per-pixel local background score
    from edgewise_core.edge.normal_sampling import boundary_evidence
    evidence = boundary_evidence(arr, edge, interior)

    masks: Masks = {"edge": edge, "iy": iy, "ix": ix, "boundary_evidence": evidence, "params": params}
    return im, candidates, masks


def build_blend_map(
    edge: np.ndarray,
    candidates: list[EdgeCandidate],
    reviews: dict[RGB, float],
    params: DecontaminationParams,
    boundary_evidence: dict | None = None,
) -> dict[tuple[int, int], float]:
    """Decide the pull strength for every edge pixel.

    - unflagged edge pixels       : unchanged
    - rule-confirmed (force)      : force_blend, unless Jev says subject detail
    - semantic REMOVE             : base_blend + scaled extra (0..jev_max_extra_blend)
    - boundary local_bg_score    : boost blend for pixels closer to background side
    """
    blend: dict[tuple[int, int], float] = {}
    protected = set()
    for c in candidates:
        key = (c.y, c.x)
        if not edge[key]:
            continue
        probability = reviews.get(c.edge_rgb)
        if probability is not None and probability < params.jev_uncertain_floor:
            protected.add(key)
            continue
        blend[key] = params.base_blend
        if c.force:
            blend[key] = params.force_blend
            continue
        if probability is not None and probability > params.jev_threshold:
            extra = (
                (probability - params.jev_threshold)
                / (1.0 - params.jev_threshold)
                * params.jev_max_extra_blend
            )
            blend[key] = min(1.0, params.base_blend + extra)

    # Boundary Normal Sampling boost: pixels closer to background side
    # get extra pull toward interior. Only applies to non-force pixels
    # (force pixels already get force_blend).
    if boundary_evidence:
        force_keys = {(c.y, c.x) for c in candidates if c.force}
        for (y, x), ev in boundary_evidence.items():
            key = (int(y), int(x))
            if key in blend and key not in force_keys and key not in protected:
                score = ev.get("local_bg_score", 0.5)
                if score > 0.75:  # very clearly background-side only
                    boost = (score - 0.75) * 0.3
                    blend[key] = min(1.0, blend[key] + boost)

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
    if blend:
        coords = np.array(list(blend), dtype=int)
        y, x = coords[:, 0], coords[:, 1]
        strength = np.array(list(blend.values()))
        # All references come from the same immutable input, independent of
        # iteration order. Vectorization avoids a Python loop per pixel.
        result[y, x, :3] = np.trunc(
            arr[y, x, :3] * (1-strength[:, None])
            + arr[iy[y, x], ix[y, x], :3] * strength[:, None]
        )
        result[y, x, 3] = np.trunc(arr[y, x, 3] * (1-strength*params.alpha_decay))
    return result


def decontaminate(
    img_path: str,
    out_path: str,
    params: DecontaminationParams,
    reviewer: SemanticReviewer | None = None,
    skip_jev: bool = False,
    iterations: int = 3,
) -> dict[str, Any]:
    """Process background once, then refine in memory until pixels stabilize.

    Intermediate files are only needed by the optional quality reviewer.
    The final output is always saved, including an early quality stop.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with Image.open(img_path) as source:
        current = source.convert("RGBA")
    quality_history: list[float] = []
    change_history: list[int] = []
    stop_reason = "iteration_limit"
    from contextlib import nullcontext
    temporary = (TemporaryDirectory(prefix="edgewise-", dir=Path(out_path).resolve().parent)
                 if reviewer is not None and not skip_jev else nullcontext(None))
    with temporary as temp_dir:
        for it in range(iterations):
            im, candidates, masks = analyze_frame(current, params, reviewer, skip_jev,
                                                  preprocess=it == 0)
            params = masks["params"]
            edge = masks["edge"]
            edge_px = int(edge.sum())
            print(f"  [iter {it+1}/{iterations}] edge: {edge_px}, flagged: {len(candidates)}, "
                  f"force: {sum(c.force for c in candidates)}")
            reviews: dict[RGB, float] = {}
            if candidates and not skip_jev:
                if reviewer is None:
                    raise ValueError("a reviewer is required unless --skip-jev is set")
                reviews = reviewer.review(candidates)
            blend = build_blend_map(edge, candidates, reviews, params,
                                    masks.get("boundary_evidence"))
            arr = np.asarray(im).astype(np.float64)
            result = apply_decontamination(arr, blend, masks["iy"], masks["ix"], params)
            changed = int(np.any(result != arr, axis=2).sum())
            change_history.append(changed)
            current = Image.fromarray(result.astype(np.uint8))
            print(f"  changed {changed} pixels")
            # Background removal and trapped-region handling only happen once.
            params = replace(params, background_tolerance=None)
            if changed == 0:
                stop_reason = "pixels_stable"
                break
            if reviewer is not None and not skip_jev and it < iterations-1:
                quality_path = str(Path(temp_dir) / "quality.png")
                current.save(quality_path)
                report = reviewer.evaluate_quality(quality_path, params.wall_rgb)
                quality_history.append(report.quality_score)
                if not report.needs_more_iterations:
                    stop_reason = "quality_accepted"
                    break
        current.save(out_path)
    return {
        "edge_px": edge_px,
        "iterations": it+1,
        "quality_history": quality_history,
        "changed_pixels": change_history,
        "stop_reason": stop_reason,
        "wall_rgb": params.wall_rgb,
    }

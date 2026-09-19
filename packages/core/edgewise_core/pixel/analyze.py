"""Input analysis: automatically detect frame properties before processing.

Runs before the pipeline to answer:
  - Is this frame opaque (wall-backed) or already transparent?
  - What is the actual wall/background color?
  - How complex is the edge situation?

All deterministic — no API, no models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgewise_types.candidate import RGB


@dataclass(frozen=True, slots=True)
class InputAnalysis:
    """What Step 0 learned about the input frame."""

    is_opaque: bool          # True if alpha is all 255 (no transparency)
    wall_rgb: RGB            # sampled background color from corners/edges
    wall_color: RGB         # alias for wall_rgb
    wall_std: float          # color std across border samples (uniformity)
    estimated_edge_px: int   # rough count of opaque pixels adjacent to... well, opaque
    needs_background_removal: bool  # should we run flood-fill?
    suggested_tolerance: float  # recommended background_tolerance


def _sample_border_colors(rgb: np.ndarray, margin: int = 8) -> np.ndarray:
    """Sample pixels from the four corners (small patches, robust to subject intrusion)."""
    h, w = rgb.shape[:2]
    corners = [
        rgb[0:margin, 0:margin],          # top-left
        rgb[0:margin, w - margin:w],       # top-right
        rgb[h - margin:h, 0:margin],       # bottom-left
        rgb[h - margin:h, w - margin:w],   # bottom-right
    ]
    return np.vstack([c.reshape(-1, 3) for c in corners])


def analyze_input(arr: np.ndarray) -> InputAnalysis:
    """Analyze a frame to decide how to process it.

    `arr` is HxWx4 (RGBA) float64.
    """
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # 1. Is the frame fully opaque?
    is_opaque = bool((alpha >= 255).all())

    # 2. Sample corner colors to detect wall/background (median = robust to outliers)
    samples = _sample_border_colors(rgb)
    wall_rgb = tuple(int(v) for v in np.median(samples, axis=0))
    wall_std = float(samples.std(axis=0).mean())

    # 3. Estimate edge complexity (rough): count opaque pixels that are
    #    adjacent to a different color (simple gradient-based estimate)
    gray = rgb.mean(axis=2)
    gradient_x = np.abs(np.diff(gray, axis=1))
    gradient_y = np.abs(np.diff(gray, axis=0))
    strong_edges = ((gradient_x > 20).any(axis=0)).sum() + ((gradient_y > 20).any(axis=0)).sum()

    # 4. Decide if background removal is needed
    needs_bg = is_opaque  # opaque frame = wall-backed, needs flood-fill

    # 5. Suggest tolerance based on wall uniformity
    #    Uniform wall → tight tolerance works; noisy wall → wider tolerance
    if wall_std < 10:
        suggested_tolerance = 20.0
    elif wall_std < 25:
        suggested_tolerance = 30.0
    else:
        suggested_tolerance = 45.0

    return InputAnalysis(
        is_opaque=is_opaque,
        wall_rgb=wall_rgb,
        wall_color=wall_rgb,
        wall_std=wall_std,
        estimated_edge_px=int(strong_edges),
        needs_background_removal=needs_bg,
        suggested_tolerance=suggested_tolerance,
    )

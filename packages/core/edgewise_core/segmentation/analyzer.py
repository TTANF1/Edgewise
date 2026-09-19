"""Region Analyzer: group suspicious pixels into CandidateRegions.

Input: a binary mask of suspicious pixels (edge + trapped background).
Output: list of CandidateRegion with computed features.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from edgewise_core.segmentation.components import connected_components
from edgewise_core.segmentation.region import CandidateRegion, RegionFeatures
from edgewise_types.candidate import RGB


def _boundary_of(mask: np.ndarray) -> np.ndarray:
    """Return coordinates of pixels on the boundary of a binary mask."""
    eroded = ndimage.binary_erosion(mask)
    boundary = mask & ~eroded
    return np.argwhere(boundary)


def detect_regions(
    suspicious_mask: np.ndarray,
    alpha: np.ndarray,
    rgb: np.ndarray,
    wall_rgb: RGB,
    wall_threshold: float = 25.0,
) -> list[CandidateRegion]:
    """Group suspicious pixels into regions and compute features.

    Parameters
    ----------
    suspicious_mask : HxW bool
        Pixels that might be background residue (edge + trapped).
    alpha : HxW float
        Alpha channel (255 = opaque, 0 = transparent).
    rgb : HxWx3 float
        RGB channels.
    wall_rgb : tuple
        Known background wall color.
    wall_threshold : float
        Color distance below which a pixel counts as "wall-like".
    """
    labels, n = connected_components(suspicious_mask)
    if n == 0:
        return []

    opaque = alpha >= 128
    h, w = alpha.shape
    wall = np.array(wall_rgb, dtype=np.float64)

    # Precompute background reachability: pixels connected to image border
    # (this is the "external background" mask from flood-fill)
    # Pixels that are NOT reachable from border = trapped/enclosed
    external_bg = _reachable_from_border(opaque)
    # A region's reachability = fraction of its pixels that touch external bg
    # (0 = fully enclosed, 1 = fully connected to outside)

    regions: list[CandidateRegion] = []
    for label_id in range(1, n + 1):
        region_mask = labels == label_id
        pixels = np.argwhere(region_mask)
        area = len(pixels)
        if area == 0:
            continue

        ys, xs = pixels[:, 0], pixels[:, 1]
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        width = x1 - x0 + 1
        height = y1 - y0 + 1

        # Boundary pixels
        boundary = _boundary_of(region_mask)

        # Feature: foreground neighbors (opaque pixels adjacent to region)
        dilated = ndimage.binary_dilation(region_mask, iterations=1)
        foreground_neighbors = int((dilated & opaque & ~region_mask).sum())

        # Feature: background similarity (mean color distance to wall)
        region_rgb = rgb[region_mask]
        dist_to_wall = np.sqrt(((region_rgb - wall) ** 2).sum(axis=1))
        background_similarity = float(np.clip(1.0 - dist_to_wall.mean() / 100.0, 0, 1))

        # Feature: enclosure (fraction of region boundary adjacent to foreground)
        if len(boundary) > 0:
            boundary_opaque_count = 0
            for by, bx in boundary:
                # Check 4 neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = int(by) + dy, int(bx) + dx
                    if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and not region_mask[ny, nx]:
                        boundary_opaque_count += 1
                        break
            enclosure = float(boundary_opaque_count) / len(boundary)
        else:
            enclosure = 0.0

        # Feature: border distance (min distance to image edge)
        border_distances = [y0, x0, h - 1 - y1, w - 1 - x1]
        border_distance = float(min(border_distances))

        # Feature: edge distance (distance to nearest opaque pixel)
        if foreground_neighbors > 0:
            edge_distance = 0.0  # adjacent to foreground
        else:
            edge_distance = 1.0

        # Feature: local contrast
        region_brightness = float(region_rgb.mean())
        # Sample surrounding opaque pixels
        dilated_rgb = rgb[dilated & opaque & ~region_mask]
        if len(dilated_rgb) > 0:
            surrounding_brightness = float(dilated_rgb.mean())
            local_contrast = abs(region_brightness - surrounding_brightness) / 255.0
        else:
            local_contrast = 0.0

        # Feature: background reachability (fraction of pixels touching external bg)
        reachable = (region_mask & external_bg).sum()
        background_reachability = float(reachable) / area if area > 0 else 0.0

        # Feature: narrow gap score
        # A region is likely a narrow gap if: small area + thin shape + enclosed
        narrow_gap_score = 0.0
        if width <= 3 or height <= 3:
            narrow_gap_score = enclosure * background_similarity * 0.5 + 0.5
        elif area <= 16 and enclosure > 0.7:
            narrow_gap_score = enclosure * background_similarity
        narrow_gap_score = float(np.clip(narrow_gap_score, 0, 1))

        # Feature: isolation score (small + wall-like + not reachable)
        isolation_score = 0.0
        if area <= 8 and background_reachability < 0.1:
            isolation_score = background_similarity * enclosure
        isolation_score = float(np.clip(isolation_score, 0, 1))

        features = RegionFeatures(
            area=area,
            width=width,
            height=height,
            aspect_ratio=width / max(height, 1),
            border_distance=border_distance,
            foreground_neighbors=foreground_neighbors,
            background_similarity=background_similarity,
            enclosure=enclosure,
            edge_distance=edge_distance,
            local_contrast=local_contrast,
            background_reachability=background_reachability,
            narrow_gap_score=narrow_gap_score,
            isolation_score=isolation_score,
        )

        regions.append(CandidateRegion(
            id=label_id,
            pixels=pixels,
            area=area,
            bbox=(y0, x0, y1, x1),
            boundary_pixels=boundary,
            features=features,
        ))

    return regions


def _reachable_from_border(opaque: np.ndarray) -> np.ndarray:
    """Return mask of pixels connected to image border (external background).

    Note: this treats transparent pixels as background. For the cutout
    pipeline, opaque pixels are the subject; transparent = background.
    """
    # Background = NOT opaque (alpha < 128)
    background = ~opaque
    # Label background components
    labels, n = ndimage.label(background)
    if n == 0:
        return np.zeros_like(opaque, dtype=bool)
    # Find which label touches the border
    h, w = opaque.shape
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)
    # Return mask of background pixels connected to border
    reachable = np.isin(labels, list(border_labels))
    return reachable

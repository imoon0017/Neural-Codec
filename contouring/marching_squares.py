"""Iso-contour extraction at iso-level 0.5 on a cSDF grid.

Uses ``skimage.measure.find_contours`` (compiled C, marching-squares with
bilinear interpolation) instead of a hand-rolled Python loop.

Coordinate convention
─────────────────────
Pixel (row, col) has its centre at:
    x_nm = origin_x_nm + (col + 0.5) * grid_res_nm_per_px
    y_nm = origin_y_nm + (row + 0.5) * grid_res_nm_per_px
Row 0 = y_min (bottom of patch), matching the rasterizer convention.

skimage returns contour points as (row, col) array-index coordinates where the
centre of pixel [i, j] is at (i, j).  This module converts to the (x, y) pixel
space used by the rest of the contouring pipeline, where x = col + 0.5 and
y = row + 0.5, so that ``pixels_to_nm`` in ``pwcl_fit`` applies without any
extra offset.

Winding order
─────────────
``find_contours`` produces contours in a consistent traversal order but does not
guarantee CCW/CW convention.  ``fix_winding()`` corrects this after the fact.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from skimage.measure import find_contours as _skimage_find_contours

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Iso-level: SDF=0 boundary — must match cSDF normalisation convention.
ISO_LEVEL: float = 0.5


def extract_isocontours(
    csdf: "NDArray[np.float32]",
    iso: float = ISO_LEVEL,
) -> list[list[tuple[float, float]]]:
    """Extract closed iso-contours from a 2D cSDF grid.

    Delegates to ``skimage.measure.find_contours`` (compiled marching-squares
    with bilinear interpolation).  Contour points are returned in (x, y)
    pixel space with the pixel-centre offset (+0.5) already applied, so they
    are ready for ``pixels_to_nm`` without further adjustment.

    Args:
        csdf: float32 [H, W] array (row 0 = y_min).
        iso:  Iso-level for extraction (default 0.5).

    Returns:
        List of closed contours.  Each contour is a list of (x, y) points
        where x = col-direction and y = row-direction, both in pixel units
        with the 0.5 centre offset applied.  Winding order is *not*
        corrected here — call ``fix_winding()`` afterwards.
    """
    if csdf.ndim != 2:
        raise ValueError(f"csdf must be 2D, got shape {csdf.shape}")

    H, W = csdf.shape
    if H < 2 or W < 2:
        return []

    # skimage returns a list of (N, 2) float arrays, each row is [row, col].
    raw = _skimage_find_contours(csdf, level=iso)

    result: list[list[tuple[float, float]]] = []
    for arr in raw:
        if len(arr) < 3:
            continue
        # Convert (row, col) → (x=col+0.5, y=row+0.5) pixel space.
        # The +0.5 shifts from skimage's array-index centre convention to
        # the pixel-centre convention used by pixels_to_nm.
        pts = [(float(col) + 0.5, float(row) + 0.5) for row, col in arr]
        result.append(pts)

    return result


# ─── Winding-order correction ─────────────────────────────────────────────────


def _signed_area(pts: list[tuple[float, float]]) -> float:
    """Compute signed area of a closed polygon via the shoelace formula."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def fix_winding(
    contours: list[list[tuple[float, float]]],
) -> list[tuple[list[tuple[float, float]], bool]]:
    """Classify contours as outer/hole and enforce winding convention.

    Convention:
        outer boundary → CCW (positive signed area, is_hole=False)
        interior hole  → CW  (negative signed area, is_hole=True)

    Holes are identified as contours whose bounding box is fully contained
    within another contour's bounding box.

    Args:
        contours: Raw contours from ``extract_isocontours`` in pixel space.

    Returns:
        List of (points, is_hole) tuples with corrected winding.
    """
    if not contours:
        return []

    bboxes: list[tuple[float, float, float, float]] = []
    areas: list[float] = []
    for pts in contours:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        areas.append(_signed_area(pts))

    n = len(contours)
    is_hole = [False] * n
    for i in range(n):
        xi0, yi0, xi1, yi1 = bboxes[i]
        for j in range(n):
            if i == j:
                continue
            xj0, yj0, xj1, yj1 = bboxes[j]
            if xj0 <= xi0 and yj0 <= yi0 and xi1 <= xj1 and yi1 <= yj1:
                is_hole[i] = True
                break

    result: list[tuple[list[tuple[float, float]], bool]] = []
    for i, pts in enumerate(contours):
        hole = is_hole[i]
        area = areas[i]
        pts_out = list(pts)
        if hole and area > 0:
            pts_out.reverse()
        elif not hole and area < 0:
            pts_out.reverse()
        result.append((pts_out, hole))

    return result

"""Convert pixel-space polyline contours to physical PwclContour objects.

This module handles two concerns:
1. **Coordinate transform** — pixel (x, y) → physical nm using the patch origin
   and grid resolution, matching the rasterizer convention exactly.
2. **Segment fitting** — wrap each edge as a LINE PwclSegment (MVP).

The result is a list of PwclContour objects ready to be written to OASIS via
``klayout.db`` or passed back to the rasterizer for round-trip validation.

Physical coordinate convention (must match rasterizer)
───────────────────────────────────────────────────────
Pixel centre (col, row) maps to:
    x_nm = origin_x_nm + (col + 0.5) * grid_res_nm_per_px   [already embedded in
    y_nm = origin_y_nm + (row + 0.5) * grid_res_nm_per_px    the marching-squares pts]

Marching-squares points already carry the +0.5 pixel-centre offset (see
``_cell_edge_point``), so the transform here is just:
    x_nm = origin_x_nm + x_px * grid_res_nm_per_px
    y_nm = origin_y_nm + y_px * grid_res_nm_per_px
"""

from __future__ import annotations

import logging
import math

from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType

log = logging.getLogger(__name__)


def pixels_to_nm(
    pts_px: list[tuple[float, float]],
    origin_x_nm: float,
    origin_y_nm: float,
    grid_res_nm_per_px: float,
) -> list[tuple[float, float]]:
    """Convert pixel-space points to physical nanometres.

    Args:
        pts_px: Points in pixel space (x=col, y=row) with sub-pixel interpolation.
        origin_x_nm: x-coordinate of the patch bottom-left corner (nm).
        origin_y_nm: y-coordinate of the patch bottom-left corner (nm).
        grid_res_nm_per_px: Physical size of one pixel (nm/px).

    Returns:
        List of (x_nm, y_nm) tuples.
    """
    return [
        (
            origin_x_nm + px * grid_res_nm_per_px,
            origin_y_nm + py * grid_res_nm_per_px,
        )
        for px, py in pts_px
    ]


def fit_line_contour(
    pts_nm: list[tuple[float, float]],
    is_hole: bool,
    min_segment_nm: float = 0.0,
) -> PwclContour:
    """Fit a closed polyline as a LINE-segment PwclContour.

    Each consecutive pair of points becomes one LINE segment.  The last point
    connects back to the first to close the contour.

    Args:
        pts_nm: Closed polyline vertices in nanometres.  The last vertex
            should *not* repeat the first (the closing edge is implicit).
        is_hole: True if this is an interior hole contour.
        min_segment_nm: Skip degenerate edges shorter than this threshold (nm).
            Default 0.0 keeps all edges.

    Returns:
        PwclContour with LINE segments.
    """
    n = len(pts_nm)
    if n < 3:
        log.warning("Contour has fewer than 3 vertices (%d) — skipping.", n)
        return PwclContour(segments=[], is_hole=is_hole)

    segments: list[PwclSegment] = []
    for i in range(n):
        p0 = pts_nm[i]
        p1 = pts_nm[(i + 1) % n]
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length < min_segment_nm:
            continue
        segments.append(PwclSegment(type=SegmentType.LINE, pts=[p0, p1]))

    return PwclContour(segments=segments, is_hole=is_hole)


def polylines_to_pwcl(
    contours_px: list[tuple[list[tuple[float, float]], bool]],
    origin_x_nm: float,
    origin_y_nm: float,
    grid_res_nm_per_px: float,
    min_segment_nm: float = 0.0,
) -> list[PwclContour]:
    """Convert pixel-space polylines (with winding flags) to PwclContour objects.

    This is the main entry point for the fitting stage.  It chains coordinate
    conversion and LINE-segment fitting.

    Args:
        contours_px: List of (points_px, is_hole) from ``fix_winding()``.
        origin_x_nm: Patch bottom-left x (nm).
        origin_y_nm: Patch bottom-left y (nm).
        grid_res_nm_per_px: nm per pixel.
        min_segment_nm: Discard degenerate edges shorter than this (nm).

    Returns:
        List of PwclContour objects in physical nm coordinates.
    """
    result: list[PwclContour] = []
    for pts_px, is_hole in contours_px:
        pts_nm = pixels_to_nm(pts_px, origin_x_nm, origin_y_nm, grid_res_nm_per_px)
        contour = fit_line_contour(pts_nm, is_hole=is_hole, min_segment_nm=min_segment_nm)
        if contour.segments:
            result.append(contour)
    return result

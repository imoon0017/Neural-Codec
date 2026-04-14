"""Public API for the contouring module: cSDF → PWCL.

The full pipeline is:
    1. Marching squares at iso=0.5  →  pixel-space closed polylines
    2. Winding-order correction     →  CCW outer / CW hole classification
    3. Coordinate transform + fit   →  PwclContour objects in physical nm

Usage
─────
    from contouring.contour import csdf_to_contours

    contours = csdf_to_contours(
        csdf,
        origin_x_nm=0.0,
        origin_y_nm=0.0,
        grid_res_nm_per_px=1.0,
    )
    # contours: list[PwclContour] — LINE segments in nm

The default output uses only LINE segments.  This is sufficient for IC mask
geometry that was rasterized from straight-edge polygons, and for round-trip
validation at any resolution where features span ≥ 2 pixels in the truncation
band.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from contouring.marching_squares import ISO_LEVEL, extract_isocontours, fix_winding
from contouring.pwcl_fit import polylines_to_pwcl
from csdf.csdf_utils import PwclContour

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


def csdf_to_contours(
    csdf: "NDArray[np.float32]",
    origin_x_nm: float,
    origin_y_nm: float,
    grid_res_nm_per_px: float,
    iso: float = ISO_LEVEL,
    min_segment_nm: float = 0.0,
) -> list[PwclContour]:
    """Convert a cSDF patch to PWCL contours.

    Runs the full marching-squares → winding-fix → coordinate-transform →
    LINE-fitting pipeline in one call.

    Args:
        csdf: float32 [H, W] cSDF array (row 0 = y_min, values in [0, 1]).
        origin_x_nm: x-coordinate of the patch bottom-left corner (nm).
        origin_y_nm: y-coordinate of the patch bottom-left corner (nm).
        grid_res_nm_per_px: Physical size of one pixel (nm/px).
        iso: Iso-level for contour extraction (default 0.5 = SDF boundary).
            Must be 0.5 for correct marching-squares reconstruction.
        min_segment_nm: Skip degenerate edges shorter than this (nm).

    Returns:
        List of PwclContour objects with LINE segments in physical nm.
        Outer boundaries are CCW (is_hole=False); interior holes are CW
        (is_hole=True).

    Raises:
        ValueError: If ``csdf`` is not a 2D array or ``iso`` is not in (0, 1).
    """
    if csdf.ndim != 2:
        raise ValueError(f"csdf must be 2D float32 array, got ndim={csdf.ndim}")
    if not (0.0 < iso < 1.0):
        raise ValueError(f"iso must be in (0, 1), got {iso}")

    # Step 1: extract pixel-space iso-contours
    raw_contours = extract_isocontours(csdf, iso=iso)
    if not raw_contours:
        log.debug("No iso-contours found at iso=%.3f", iso)
        return []

    # Step 2: fix winding order and classify outer/hole
    winding_fixed = fix_winding(raw_contours)

    # Step 3: convert to physical nm and fit as LINE segments
    pwcl = polylines_to_pwcl(
        winding_fixed,
        origin_x_nm=origin_x_nm,
        origin_y_nm=origin_y_nm,
        grid_res_nm_per_px=grid_res_nm_per_px,
        min_segment_nm=min_segment_nm,
    )

    log.debug(
        "csdf_to_contours: %d raw → %d winding-fixed → %d PWCL contours",
        len(raw_contours),
        len(winding_fixed),
        len(pwcl),
    )
    return pwcl

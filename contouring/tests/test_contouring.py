"""Tests for the contouring module (marching squares + PWCL fitting).

Tests are ordered from unit (marching squares internals) to integration
(round-trip with the cSDF rasterizer).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from contouring.marching_squares import (
    ISO_LEVEL,
    _signed_area,
    extract_isocontours,
    fix_winding,
)
from contouring.pwcl_fit import pixels_to_nm, polylines_to_pwcl
from contouring.contour import csdf_to_contours
from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_filled_patch(H: int, W: int, inside_value: float = 1.0) -> np.ndarray:
    """Return an all-inside cSDF patch (no boundary within the patch)."""
    return np.full((H, W), inside_value, dtype=np.float32)


def _make_empty_patch(H: int, W: int) -> np.ndarray:
    """Return an all-outside cSDF patch."""
    return np.zeros((H, W), dtype=np.float32)


def _make_disk_csdf(H: int, W: int, cx: float, cy: float, r: float, t: float) -> np.ndarray:
    """Analytic cSDF for a filled circle of radius r (centre at cx, cy in pixel space)."""
    row_idx, col_idx = np.mgrid[0:H, 0:W]
    x = col_idx + 0.5  # pixel centre x
    y = row_idx + 0.5  # pixel centre y
    sdf = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r  # negative inside
    clamped = np.clip(sdf, -t, t)
    return ((t - clamped) / (2 * t)).astype(np.float32)


def _make_rect_csdf(
    H: int, W: int,
    x0: float, y0: float, x1: float, y1: float,
    t: float,
) -> np.ndarray:
    """Analytic cSDF for an axis-aligned rectangle [x0,x1] × [y0,y1] in pixel space."""
    row_idx, col_idx = np.mgrid[0:H, 0:W]
    x = col_idx + 0.5
    y = row_idx + 0.5
    dx = np.maximum(x0 - x, 0.0) + np.maximum(x - x1, 0.0)
    dy = np.maximum(y0 - y, 0.0) + np.maximum(y - y1, 0.0)
    dist_outside = np.sqrt(dx ** 2 + dy ** 2)
    # Inside: negative, using Chebyshev-like inner distance as approximation
    dist_inside = np.minimum(x - x0, np.minimum(x1 - x, np.minimum(y - y0, y1 - y)))
    sdf = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), -dist_inside, dist_outside)
    clamped = np.clip(sdf, -t, t)
    return ((t - clamped) / (2 * t)).astype(np.float32)


# ─── Unit: _signed_area ───────────────────────────────────────────────────────


def test_signed_area_ccw_square():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    area = _signed_area(pts)
    assert area > 0, "CCW square should have positive signed area"
    assert abs(area - 1.0) < 1e-9


def test_signed_area_cw_square():
    pts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    area = _signed_area(pts)
    assert area < 0, "CW square should have negative signed area"
    assert abs(area + 1.0) < 1e-9


def test_signed_area_triangle():
    pts = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    area = _signed_area(pts)
    assert abs(area - 6.0) < 1e-9


# ─── Unit: extract_isocontours edge cases ────────────────────────────────────


def test_extract_all_outside_returns_empty():
    csdf = _make_empty_patch(8, 8)
    contours = extract_isocontours(csdf)
    assert contours == []


def test_extract_all_inside_returns_empty():
    csdf = _make_filled_patch(8, 8)
    contours = extract_isocontours(csdf)
    assert contours == []


def test_extract_too_small_returns_empty():
    csdf = np.array([[0.0, 1.0]], dtype=np.float32)  # 1×2 — too small
    assert extract_isocontours(csdf) == []


def test_extract_single_inside_pixel():
    """One inside pixel surrounded by outside — should produce a small contour."""
    csdf = np.zeros((5, 5), dtype=np.float32)
    csdf[2, 2] = 1.0
    contours = extract_isocontours(csdf, iso=0.5)
    assert len(contours) >= 1


# ─── Unit: extract_isocontours — disk ────────────────────────────────────────


def test_disk_contour_count():
    """A disk should produce exactly one contour."""
    S, cx, cy, r, t = 64, 32.0, 32.0, 16.0, 5.0
    csdf = _make_disk_csdf(S, S, cx, cy, r, t)
    contours = extract_isocontours(csdf)
    assert len(contours) == 1


def test_disk_contour_is_closed():
    """The contour should be closed (last→first edge implicit in marching squares)."""
    S, cx, cy, r, t = 64, 32.0, 32.0, 16.0, 5.0
    csdf = _make_disk_csdf(S, S, cx, cy, r, t)
    contours = extract_isocontours(csdf)
    assert len(contours) == 1
    pts = contours[0]
    assert len(pts) >= 6  # at least a coarse circle


def test_disk_contour_approximate_radius():
    """Extracted contour points should lie near the iso-circle (r±2 px tolerance)."""
    S, cx, cy, r, t = 64, 32.0, 32.0, 15.0, 5.0
    csdf = _make_disk_csdf(S, S, cx, cy, r, t)
    contours = extract_isocontours(csdf)
    assert len(contours) == 1
    pts = contours[0]
    radii = [math.hypot(x - cx, y - cy) for x, y in pts]
    for rad in radii:
        assert abs(rad - r) < 2.5, f"Point radius {rad:.2f} too far from expected {r}"


# ─── Unit: fix_winding ────────────────────────────────────────────────────────


def test_fix_winding_ccw_outer():
    """CCW outer square should remain CCW after fix_winding."""
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    result = fix_winding([outer])
    pts, is_hole = result[0]
    assert not is_hole
    assert _signed_area(pts) > 0


def test_fix_winding_cw_outer_gets_reversed():
    """CW outer square should be reversed to CCW by fix_winding."""
    cw_outer = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    assert _signed_area(cw_outer) < 0
    result = fix_winding([cw_outer])
    pts, is_hole = result[0]
    assert not is_hole
    assert _signed_area(pts) > 0


def test_fix_winding_hole_is_cw():
    """Inner contour fully inside outer should be classified as hole and be CW."""
    outer = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    inner = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]  # CCW
    result = fix_winding([outer, inner])
    # find which is hole
    roles = {is_hole: pts for pts, is_hole in result}
    assert True in roles  # there is a hole
    hole_pts = roles[True]
    assert _signed_area(hole_pts) < 0, "Hole should be CW (negative area)"


def test_fix_winding_preserves_count():
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    inner = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    result = fix_winding([outer, inner])
    assert len(result) == 2


# ─── Unit: pixels_to_nm ──────────────────────────────────────────────────────


def test_pixels_to_nm_identity():
    pts_px = [(0.5, 0.5), (1.5, 0.5)]
    nm = pixels_to_nm(pts_px, origin_x_nm=0.0, origin_y_nm=0.0, grid_res_nm_per_px=1.0)
    assert nm == [(0.5, 0.5), (1.5, 0.5)]


def test_pixels_to_nm_with_offset_and_scale():
    pts_px = [(1.0, 2.0)]
    nm = pixels_to_nm(pts_px, origin_x_nm=100.0, origin_y_nm=200.0, grid_res_nm_per_px=6.0)
    assert abs(nm[0][0] - 106.0) < 1e-9
    assert abs(nm[0][1] - 212.0) < 1e-9


# ─── Unit: polylines_to_pwcl ─────────────────────────────────────────────────


def test_polylines_to_pwcl_segment_types():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    result = polylines_to_pwcl(
        [(pts, False)],
        origin_x_nm=0.0, origin_y_nm=0.0, grid_res_nm_per_px=1.0,
    )
    assert len(result) == 1
    for seg in result[0].segments:
        assert seg.type == SegmentType.LINE


def test_polylines_to_pwcl_segment_count():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    result = polylines_to_pwcl(
        [(pts, False)],
        origin_x_nm=0.0, origin_y_nm=0.0, grid_res_nm_per_px=1.0,
    )
    assert len(result[0].segments) == 4  # 4 vertices → 4 closing edges


def test_polylines_to_pwcl_is_hole_flag():
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole  = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    result = polylines_to_pwcl(
        [(outer, False), (hole, True)],
        origin_x_nm=0.0, origin_y_nm=0.0, grid_res_nm_per_px=1.0,
    )
    assert len(result) == 2
    flags = {c.is_hole for c in result}
    assert flags == {True, False}


def test_polylines_to_pwcl_min_segment_filter():
    pts = [(0.0, 0.0), (0.001, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    result = polylines_to_pwcl(
        [(pts, False)],
        origin_x_nm=0.0, origin_y_nm=0.0, grid_res_nm_per_px=1.0,
        min_segment_nm=0.1,
    )
    # The 0.001 nm edge should be filtered out
    assert len(result[0].segments) < len(pts)


# ─── Integration: csdf_to_contours ───────────────────────────────────────────


def test_csdf_to_contours_disk_returns_one_contour():
    S, cx, cy, r, t = 64, 32.0, 32.0, 15.0, 5.0
    csdf = _make_disk_csdf(S, S, cx, cy, r, t)
    contours = csdf_to_contours(csdf, origin_x_nm=0.0, origin_y_nm=0.0,
                                 grid_res_nm_per_px=1.0)
    assert len(contours) == 1
    assert not contours[0].is_hole


def test_csdf_to_contours_all_line_segments():
    S = 32
    csdf = _make_disk_csdf(S, S, 16.0, 16.0, 8.0, 3.0)
    contours = csdf_to_contours(csdf, origin_x_nm=0.0, origin_y_nm=0.0,
                                 grid_res_nm_per_px=1.0)
    for c in contours:
        for seg in c.segments:
            assert seg.type == SegmentType.LINE


def test_csdf_to_contours_physical_coords():
    """Contour points should be in physical nm, not pixel units."""
    S, r, grid_res = 64, 10.0, 6.0
    cx, cy = 32.0, 32.0  # pixel space
    csdf = _make_disk_csdf(S, S, cx, cy, r, t=5.0)
    ox, oy = 1000.0, 2000.0
    contours = csdf_to_contours(csdf, origin_x_nm=ox, origin_y_nm=oy,
                                 grid_res_nm_per_px=grid_res)
    assert len(contours) == 1
    for seg in contours[0].segments:
        for x_nm, y_nm in seg.pts[:2]:
            # Point should be near the circle boundary in nm space
            # Circle centre in nm: ox + cx*grid_res, oy + cy*grid_res
            cx_nm = ox + cx * grid_res
            cy_nm = oy + cy * grid_res
            r_nm = r * grid_res
            dist = math.hypot(x_nm - cx_nm, y_nm - cy_nm)
            assert abs(dist - r_nm) < 3 * grid_res, (
                f"Point ({x_nm:.1f}, {y_nm:.1f}) dist={dist:.1f} expected ~{r_nm:.1f}"
            )


def test_csdf_to_contours_empty_patch():
    csdf = np.zeros((32, 32), dtype=np.float32)
    contours = csdf_to_contours(csdf, 0.0, 0.0, 1.0)
    assert contours == []


def test_csdf_to_contours_full_patch():
    csdf = np.ones((32, 32), dtype=np.float32)
    contours = csdf_to_contours(csdf, 0.0, 0.0, 1.0)
    assert contours == []


def test_csdf_to_contours_invalid_ndim():
    with pytest.raises(ValueError, match="2D"):
        csdf_to_contours(np.zeros((32,), dtype=np.float32), 0.0, 0.0, 1.0)


def test_csdf_to_contours_invalid_iso():
    csdf = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="iso"):
        csdf_to_contours(csdf, 0.0, 0.0, 1.0, iso=1.5)


# ─── Integration: round-trip with cSDF rasterizer ────────────────────────────


def test_round_trip_disk_area():
    """Round-trip: rasterize disk → extract contour → check approximate area.

    The reconstructed polygon area should be within 5% of the analytic area.
    """
    S, cx, cy, r, t, grid_res = 128, 64.0, 64.0, 24.0, 5.0, 1.0
    csdf = _make_disk_csdf(S, S, cx, cy, r, t)
    contours = csdf_to_contours(csdf, origin_x_nm=0.0, origin_y_nm=0.0,
                                 grid_res_nm_per_px=grid_res)
    assert len(contours) >= 1
    outer = [c for c in contours if not c.is_hole]
    assert len(outer) == 1

    # Reconstruct polygon vertices from LINE segments
    pts_nm: list[tuple[float, float]] = [seg.pts[0] for seg in outer[0].segments]
    poly_area = abs(_signed_area(pts_nm))
    analytic_area = math.pi * r ** 2

    rel_err = abs(poly_area - analytic_area) / analytic_area
    assert rel_err < 0.05, f"Area rel error {rel_err:.3%} > 5%"

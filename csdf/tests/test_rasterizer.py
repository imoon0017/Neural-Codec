"""Unit tests for the cSDF CPU rasterizer.

Run with:  pytest csdf/tests/ -v

Tests cover:
  - derive_patch_size_px correctness and edge cases
  - cSDF value convention: 1.0 inside, 0.0 outside, 0.5 on boundary
  - Square polygon (LINE segments only)
  - Circular arc segment (ARC)
  - Quadratic Bézier segment (BEZIER2)
  - Cubic Bézier segment (BEZIER3)
  - Polygon with a hole (nested contours)
  - Output shape, dtype, and value range

Tests that require the compiled C++ extension are skipped automatically
when the extension is not available.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from csdf.csdf_utils import (
    PwclContour,
    PwclSegment,
    SegmentType,
    derive_patch_size_px,
    rasterize_patch,
)

# ─── Skip marker ──────────────────────────────────────────────────────────────

try:
    from csdf import csdf_ext  # noqa: F401

    _EXT_AVAILABLE = True
except ImportError:
    _EXT_AVAILABLE = False

requires_ext = pytest.mark.skipif(
    not _EXT_AVAILABLE,
    reason="csdf_ext C++ extension not compiled",
)

# ─── Test helpers ─────────────────────────────────────────────────────────────

# Grid / truncation parameters used across all geometric tests.
_GRID_RES = 1.0  # nm/px — 1:1 so distances in nm == distances in px
_TRUNC_PX = 5.0
_S = 64  # patch size in pixels


def _square_contour(
    cx_nm: float, cy_nm: float, half_nm: float, is_hole: bool = False
) -> PwclContour:
    """Axis-aligned square contour with four LINE segments.

    Vertices are listed CCW (outer) or CW (hole) as required by the spec.
    """
    if not is_hole:
        # CCW: bottom-left → bottom-right → top-right → top-left
        corners = [
            (cx_nm - half_nm, cy_nm - half_nm),
            (cx_nm + half_nm, cy_nm - half_nm),
            (cx_nm + half_nm, cy_nm + half_nm),
            (cx_nm - half_nm, cy_nm + half_nm),
        ]
    else:
        # CW (hole): reverse order
        corners = [
            (cx_nm - half_nm, cy_nm - half_nm),
            (cx_nm - half_nm, cy_nm + half_nm),
            (cx_nm + half_nm, cy_nm + half_nm),
            (cx_nm + half_nm, cy_nm - half_nm),
        ]
    segments = [
        PwclSegment(SegmentType.LINE, [corners[i], corners[(i + 1) % 4]])
        for i in range(4)
    ]
    return PwclContour(segments=segments, is_hole=is_hole)


# ─────────────────────────────────────────────────────────────────────────────
# derive_patch_size_px
# ─────────────────────────────────────────────────────────────────────────────


class TestDerivePatchSizePx:
    def test_exact_multiple(self) -> None:
        assert derive_patch_size_px(64.0, 1.0, 8) == 64

    def test_rounds_up_to_multiple(self) -> None:
        # ceil(65 / 1) = 65, rounds up to 72 (next multiple of 8)
        assert derive_patch_size_px(65.0, 1.0, 8) == 72

    def test_non_integer_nm_per_px(self) -> None:
        # ceil(100.0 / 1.5) = 67, rounds up to 68 (next multiple of 4)
        s = derive_patch_size_px(100.0, 1.5, 4)
        assert s % 4 == 0
        assert s >= math.ceil(100.0 / 1.5)

    def test_compaction_ratio_1(self) -> None:
        # c=1 → no rounding needed, just ceil
        assert derive_patch_size_px(33.7, 1.0, 1) == 34

    def test_invalid_compaction_ratio(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            derive_patch_size_px(64.0, 1.0, 3)

    def test_result_divisible_by_c(self) -> None:
        for c in (1, 2, 4, 8, 16):
            s = derive_patch_size_px(100.0, 0.7, c)
            assert s % c == 0, f"S={s} not divisible by c={c}"


# ─────────────────────────────────────────────────────────────────────────────
# rasterize_patch — output properties
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestOutputProperties:
    def test_shape_and_dtype(self) -> None:
        contour = _square_contour(32.0, 32.0, 16.0)
        patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )
        assert patch.shape == (_S, _S)
        assert patch.dtype == np.float32

    def test_values_in_unit_interval(self) -> None:
        contour = _square_contour(32.0, 32.0, 16.0)
        patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )
        assert float(patch.min()) >= 0.0 - 1e-5
        assert float(patch.max()) <= 1.0 + 1e-5

    def test_cuda_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            rasterize_patch([], 0.0, 0.0, 8, 1.0, 3.0, device="cuda")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError):
            rasterize_patch([], 0.0, 0.0, 8, 1.0, 3.0, device="tpu")


# ─────────────────────────────────────────────────────────────────────────────
# cSDF value convention — LINE segments
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestValueConventionLine:
    """Verify 1.0 inside / 0.5 boundary / 0.0 outside for a square polygon."""

    # Place a 20×20 nm square in a 64-px patch.  Pixel (col, row) maps to
    # physical (col+0.5, row+0.5) nm with grid_res=1 nm/px and origin=0.
    # Centre is at (32, 32); half-side = 10 nm.

    @pytest.fixture(autouse=True)
    def _rasterize(self) -> None:
        cx, half = 32.0, 10.0
        contour = _square_contour(cx, cx, half)
        self.patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )
        self.cx = cx
        self.half = half

    def test_deep_interior_is_one(self) -> None:
        # Centre pixel is deep inside (≥ truncation_px from any edge)
        col = row = 32  # centre ≈ (32.5, 32.5) nm — 9.5 nm from nearest edge
        assert self.patch[row, col] == pytest.approx(1.0, abs=1e-4)

    def test_deep_exterior_is_zero(self) -> None:
        # Corner pixel is far outside
        assert self.patch[0, 0] == pytest.approx(0.0, abs=1e-4)

    def test_boundary_pixel_near_half(self) -> None:
        # Pixel whose centre is just outside the bottom edge: row=21
        # Bottom edge is at y=22 nm; pixel row=21 has centre y=21.5 nm → 0.5 nm outside.
        # Raw SDF = +0.5 nm; csdf = (t − clamped) / 2t = (5 − 0.5) / 10 = 0.45
        row = 21  # y_centre = 21.5 nm, bottom edge at 22.0 nm
        col = 32  # well inside in x
        expected_sdf_nm = 22.0 - 21.5  # = 0.5 nm (positive → outside)
        # Correct formula: (t − clamp(sdf)) / (2t)
        expected = (_TRUNC_PX - clamp_val(expected_sdf_nm, _TRUNC_PX)) / (2 * _TRUNC_PX)
        assert self.patch[row, col] == pytest.approx(expected, abs=0.01)

    def test_symmetry_left_right(self) -> None:
        # Patch should be symmetric left/right around x=32.
        np.testing.assert_allclose(
            self.patch[:, :32], self.patch[:, 32:][:, ::-1], atol=1e-4
        )

    def test_symmetry_top_bottom(self) -> None:
        np.testing.assert_allclose(
            self.patch[:32, :], self.patch[32:, :][::-1, :], atol=1e-4
        )


def clamp_val(sdf_nm: float, trunc_px: float) -> float:
    """Clamp SDF to [−trunc, +trunc] with grid_res=1 so nm==px."""
    return max(-trunc_px, min(trunc_px, sdf_nm))


# ─────────────────────────────────────────────────────────────────────────────
# ARC segment
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestArcSegment:
    """Rasterize a filled unit circle (CCW arc) and check its cSDF."""

    # Circle: centre (32, 32) nm, radius 12 nm.
    # Use four 90° CCW arcs, each defined by start, on-arc-mid, end.

    @pytest.fixture(autouse=True)
    def _rasterize(self) -> None:
        cx, cy, r = 32.0, 32.0, 12.0
        # Four quarter-arc segments going CCW:
        #   (r,0) → (0,r) → (−r,0) → (0,−r) → (r,0)  (relative to centre)
        def pt(angle_deg: float) -> tuple[float, float]:
            th = math.radians(angle_deg)
            return (cx + r * math.cos(th), cy + r * math.sin(th))

        def arc_seg(a0: float, a1: float) -> PwclSegment:
            mid_angle = (a0 + a1) / 2.0
            return PwclSegment(
                SegmentType.ARC,
                [pt(a0), pt(mid_angle), pt(a1)],
            )

        segments = [arc_seg(0, 90), arc_seg(90, 180), arc_seg(180, 270), arc_seg(270, 360)]
        contour = PwclContour(segments=segments, is_hole=False)
        self.patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )
        self.cx, self.cy, self.r = cx, cy, r

    def test_centre_is_inside(self) -> None:
        # Pixel at (32, 32) — centre of circle, well inside
        assert self.patch[32, 32] == pytest.approx(1.0, abs=1e-3)

    def test_far_exterior_is_zero(self) -> None:
        assert self.patch[0, 0] == pytest.approx(0.0, abs=1e-3)

    def test_point_on_boundary(self) -> None:
        # Pixel at (32+r, 32) → centre (44.5, 32.5) nm, circle boundary at x=44 nm.
        # Pixel is ~0.5 nm outside the circle.
        # Raw SDF ≈ +0.5 nm; csdf = (t − clamp(sdf)) / 2t = (5 − 0.5) / 10 = 0.45
        col, row = 44, 32
        dist_to_edge_nm = 44.5 - (self.cx + self.r)  # ≈ 0.5 nm (positive → outside)
        expected = (_TRUNC_PX - clamp_val(dist_to_edge_nm, _TRUNC_PX)) / (2 * _TRUNC_PX)
        assert self.patch[row, col] == pytest.approx(expected, abs=0.05)

    def test_radial_symmetry(self) -> None:
        # The circle patch should be approximately 4-fold symmetric.
        p = self.patch
        s = _S
        np.testing.assert_allclose(p[:s//2, :], p[s//2:, :][::-1, :], atol=0.05)
        np.testing.assert_allclose(p[:, :s//2], p[:, s//2:][:, ::-1], atol=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# BEZIER2 segment
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestBezier2Segment:
    """Rasterize a simple quadratic-Bézier closed shape."""

    @pytest.fixture(autouse=True)
    def _rasterize(self) -> None:
        # A "lens" shape: two quadratic Béziers forming a closed CCW contour.
        # Bottom arc: from (22, 32) to (42, 32) bulging down to (32, 22).
        # Top arc:    from (42, 32) to (22, 32) bulging up to (32, 42).
        segs = [
            PwclSegment(SegmentType.BEZIER2, [(22.0, 32.0), (32.0, 22.0), (42.0, 32.0)]),
            PwclSegment(SegmentType.BEZIER2, [(42.0, 32.0), (32.0, 42.0), (22.0, 32.0)]),
        ]
        contour = PwclContour(segments=segs, is_hole=False)
        self.patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )

    def test_centre_is_inside(self) -> None:
        # The centre (32.5, 32.5) is inside the lens. Distance to the nearest
        # Bézier boundary is ~4.8 nm (< truncation=5), so csdf < 1.0 but > 0.9.
        assert self.patch[32, 32] >= 0.9

    def test_far_corner_is_outside(self) -> None:
        assert self.patch[0, 0] == pytest.approx(0.0, abs=1e-3)

    def test_values_in_range(self) -> None:
        assert float(self.patch.min()) >= -1e-5
        assert float(self.patch.max()) <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# BEZIER3 segment
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestBezier3Segment:
    """Rasterize a closed contour built from two cubic Bézier curves."""

    @pytest.fixture(autouse=True)
    def _rasterize(self) -> None:
        # Two cubic Béziers forming a rounded-rectangle-like shape.
        # Bottom half: (16,32)→(20,16),(44,16),(48,32)
        # Top half:    (48,32)→(44,48),(20,48),(16,32)
        segs = [
            PwclSegment(
                SegmentType.BEZIER3,
                [(16.0, 32.0), (20.0, 16.0), (44.0, 16.0), (48.0, 32.0)],
            ),
            PwclSegment(
                SegmentType.BEZIER3,
                [(48.0, 32.0), (44.0, 48.0), (20.0, 48.0), (16.0, 32.0)],
            ),
        ]
        contour = PwclContour(segments=segs, is_hole=False)
        self.patch = rasterize_patch(
            [contour], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )

    def test_centre_is_inside(self) -> None:
        assert self.patch[32, 32] == pytest.approx(1.0, abs=0.05)

    def test_far_corner_is_outside(self) -> None:
        assert self.patch[0, 0] == pytest.approx(0.0, abs=1e-3)

    def test_values_in_range(self) -> None:
        assert float(self.patch.min()) >= -1e-5
        assert float(self.patch.max()) <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Polygon with a hole
# ─────────────────────────────────────────────────────────────────────────────


@requires_ext
class TestPolygonWithHole:
    """Outer square + inner square hole: hole region should be treated as outside."""

    @pytest.fixture(autouse=True)
    def _rasterize(self) -> None:
        outer = _square_contour(32.0, 32.0, 16.0, is_hole=False)  # 32×32 nm
        hole  = _square_contour(32.0, 32.0,  6.0, is_hole=True)   # 12×12 nm hole
        self.patch = rasterize_patch(
            [outer, hole], 0.0, 0.0, _S, _GRID_RES, _TRUNC_PX
        )

    def test_annular_region_is_inside(self) -> None:
        # Use row=20 → y_centre=20.5 nm.  Outer bottom at y=16 (dist=4.5 nm), hole
        # bottom at y=26 (dist=5.5 nm > truncation, clamped to t).  Both edges are
        # within the annular ring, so winding=+1 and SDF = −4.5 nm.
        # csdf = (5 + 4.5) / 10 = 0.95  (≥ 0.9 well within the inside band).
        assert self.patch[20, 32] >= 0.9

    def test_hole_centre_is_outside(self) -> None:
        # Centre pixel is inside the hole → should look like outside (csdf ≈ 0.5+ due to
        # distance to hole boundary, but winding = 0 → positive SDF).
        # Distance from (32.5, 32.5) to hole edge ≈ 5.5 nm < truncation → not quite 0.
        # Key assertion: csdf < 0.5 (outside convention).
        assert self.patch[32, 32] < 0.5

    def test_exterior_is_zero(self) -> None:
        assert self.patch[0, 0] == pytest.approx(0.0, abs=1e-3)

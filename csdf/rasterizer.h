// csdf/rasterizer.h
// Shared interface for the cSDF rasterizer, declaring types and functions used
// by both the CPU (OpenMP) and GPU (CUDA) implementation paths.
//
// Coordinate conventions
// ──────────────────────
//  • All geometry coordinates are in physical nanometres (nm).
//  • Pixel (col, row) has its centre at:
//      x_nm = origin_x_nm + (col + 0.5) * grid_res_nm_per_px
//      y_nm = origin_y_nm + (row + 0.5) * grid_res_nm_per_px
//  • Row 0 = bottom of the patch (y_min).
//
// cSDF formula (three-step pipeline)
// ───────────────────────────────────
//  1. sdf_nm  = signed Euclidean distance to nearest PWCL edge  (negative inside)
//  2. clamped = clamp(sdf_nm, −t_nm, +t_nm)   where t_nm = truncation_px * grid_res
//  3. csdf    = (clamped + t_nm) / (2 * t_nm)  ∈ [0, 1]
//
//  Value map: 1.0 = fully inside, 0.0 = fully outside, 0.5 = on boundary.

#pragma once

#include <cstdint>
#include <vector>

namespace csdf {

// ─── PWCL segment types ───────────────────────────────────────────────────────

enum class SegmentType : int32_t {
    LINE    = 0,  // 2 control points: pts[0]=start, pts[1]=end
    ARC     = 1,  // 3 control points: pts[0]=start, pts[1]=on-arc-mid, pts[2]=end
    BEZIER2 = 2,  // 3 control points: pts[0]=p0, pts[1]=ctrl, pts[2]=p2  (quadratic)
    BEZIER3 = 3,  // 4 control points: pts[0]=p0, pts[1]=c1, pts[2]=c2, pts[3]=p3 (cubic)
};

// Physical 2-D point (nm).
struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

// One directed edge segment in a PWCL contour.
// Unused entries in pts[] are zero-initialised.
struct PwclSegment {
    SegmentType type = SegmentType::LINE;
    Point2D     pts[4]{};
};

// A single closed PWCL contour.
// Winding convention (enforced by ingest.py):
//   is_hole == false → outer boundary (CCW, positive area)
//   is_hole == true  → interior hole  (CW,  negative area)
struct PwclContour {
    std::vector<PwclSegment> segments;
    bool is_hole = false;
};

// ─── CPU rasterizer ───────────────────────────────────────────────────────────

// Rasterize a set of PWCL contours into a float32 cSDF patch on the CPU
// using OpenMP parallelism.
//
// Parameters
// ──────────
//  contours           – closed PWCL contours in nm (outer CCW + optional CW holes)
//  origin_x_nm        – x-coordinate of patch bottom-left corner (nm)
//  origin_y_nm        – y-coordinate of patch bottom-left corner (nm)
//  patch_size_px      – side length S of the square output patch
//  grid_res_nm_per_px – physical size of one pixel (nm/px)
//  truncation_px      – SDF truncation half-width in pixel units
//  out                – caller-allocated float[S * S] output array
//                       indexed as  out[row * S + col],  row 0 = bottom
void rasterize_csdf_cpu(
    const std::vector<PwclContour>& contours,
    double  origin_x_nm,
    double  origin_y_nm,
    int     patch_size_px,
    double  grid_res_nm_per_px,
    double  truncation_px,
    float*  out
);

// Rasterize all marker regions into a single canvas in one call.
// Eliminates per-marker Python→C++ call overhead for batch workloads.
//
// all_contours        – flat list of all contours across all markers
// marker_c_start/end  – [start, end) contour index slice per marker
// marker_ox/oy        – bottom-left origin (nm) per marker
// marker_sizes        – patch side length (px) per marker
// n_markers           – number of markers
// canvas_x0/y0_nm     – canvas bottom-left origin (nm)
// canvas_H/W          – canvas dimensions (px)
// out                 – caller-allocated float[H * W] canvas, row 0 = bottom
void rasterize_csdf_canvas(
    const std::vector<PwclContour>& all_contours,
    const int*    marker_c_start,
    const int*    marker_c_end,
    const double* marker_ox,
    const double* marker_oy,
    const int*    marker_sizes,
    int     n_markers,
    double  canvas_x0_nm,
    double  canvas_y0_nm,
    double  grid_res_nm_per_px,
    double  truncation_px,
    int     canvas_H,
    int     canvas_W,
    float*  out
);

}  // namespace csdf

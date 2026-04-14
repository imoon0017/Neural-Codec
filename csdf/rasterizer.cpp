// csdf/rasterizer.cpp
// CPU rasterizer path (OpenMP) + pybind11 bindings.
//
// Algorithm overview
// ──────────────────
//  For every pixel centre p in the S×S patch:
//    1. Compute the unsigned Euclidean distance (nm) from p to the nearest
//       point on any PWCL segment (line / circular-arc / Bézier).
//    2. Determine the sign via the non-zero winding-number rule:
//         winding ≠ 0  →  inside  (sdf = −dist)
//         winding = 0  →  outside (sdf = +dist)
//    3. Truncate to [−t_nm, +t_nm] and normalise to [0, 1].
//
// Segment support
// ───────────────
//  LINE    – exact analytic closest point
//  ARC     – 3-point circular arc; exact analytic closest point
//  BEZIER2 – quadratic Bézier; analytic ray–curve intersections,
//             golden-section refinement for closest point
//  BEZIER3 – cubic Bézier; monotone-interval bisection for winding,
//             golden-section refinement for closest point
//
// Winding number
// ──────────────
//  Casts a ray in the +x direction from each query point.  Every upward
//  crossing to the right of the point contributes +1; every downward
//  crossing contributes −1.  Works correctly for outer CCW contours and
//  CW holes without special-casing them.

#include "rasterizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace csdf {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// ── 2-D cross product of (a) and (b) ─────────────────────────────────────────
inline double cross2(double ax, double ay, double bx, double by) noexcept {
    return ax * by - ay * bx;
}

// ── Clamp a value to [lo, hi] ─────────────────────────────────────────────────
inline double clamp(double v, double lo, double hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ─────────────────────────────────────────────────────────────────────────────
// LINE segment
// ─────────────────────────────────────────────────────────────────────────────

// Unsigned distance from p to the finite line segment [a, b].
double dist_line_segment(Point2D p, Point2D a, Point2D b) noexcept {
    const double dx = b.x - a.x, dy = b.y - a.y;
    const double len2 = dx * dx + dy * dy;
    if (len2 < 1e-24) return std::hypot(p.x - a.x, p.y - a.y);
    const double t = clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / len2, 0.0, 1.0);
    return std::hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

// Winding-number contribution of the line segment [a→b] at query point q.
// Uses a horizontal +x ray from q.  Returns −1, 0, or +1.
int winding_line(Point2D q, Point2D a, Point2D b) noexcept {
    const bool upward   = (a.y <= q.y && b.y > q.y);
    const bool downward = (a.y > q.y  && b.y <= q.y);
    if (!upward && !downward) return 0;
    // x-coordinate of the y = q.y intersection along a→b
    const double x_int = a.x + (q.y - a.y) * (b.x - a.x) / (b.y - a.y);
    if (x_int <= q.x) return 0;
    return upward ? +1 : -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Circular ARC  (3-point representation: start, on-arc-mid, end)
// ─────────────────────────────────────────────────────────────────────────────

struct ArcParams {
    double cx, cy, r;   // circumcircle centre and radius
    double th0, th2;    // atan2 angles of start and end points
    bool   ccw;         // true ↔ arc goes counter-clockwise
    bool   degenerate;  // collinear points → fall back to line
};

// Derive ArcParams from the three control points.
ArcParams arc_from_3pts(Point2D p0, Point2D pm, Point2D p2) noexcept {
    const double ax = p0.x, ay = p0.y;
    const double bx = pm.x, by = pm.y;
    const double cx = p2.x, cy = p2.y;
    const double D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (std::fabs(D) < 1e-14) return {0, 0, 1e18, 0, 0, true, true};
    const double ma = ax * ax + ay * ay;
    const double mb = bx * bx + by * by;
    const double mc = cx * cx + cy * cy;
    ArcParams arc;
    arc.cx  = (ma * (by - cy) + mb * (cy - ay) + mc * (ay - by)) / D;
    arc.cy  = (ma * (cx - bx) + mb * (ax - cx) + mc * (bx - ax)) / D;
    arc.r   = std::hypot(ax - arc.cx, ay - arc.cy);
    arc.th0 = std::atan2(ay - arc.cy, ax - arc.cx);
    arc.th2 = std::atan2(cy - arc.cy, cx - arc.cx);
    // CCW ↔ p0→pm→p2 is a counter-clockwise turn (cross product > 0)
    arc.ccw = cross2(bx - ax, by - ay, cx - ax, cy - ay) > 0.0;
    arc.degenerate = false;
    return arc;
}

// Is angle th (any value) contained within the arc [th0 → th2]
// travelling in direction ccw?
bool angle_in_arc(double th0, double th2, bool ccw, double th) noexcept {
    if (ccw) {
        // CCW: normalise th2 and th into [th0, th0 + 2π)
        const double span = std::fmod(th2 - th0 + 4.0 * M_PI, 2.0 * M_PI);
        const double t    = std::fmod(th  - th0 + 4.0 * M_PI, 2.0 * M_PI);
        return t <= span + 1e-10;
    } else {
        // CW from th0 → th2  ≡  CCW from th2 → th0
        const double span = std::fmod(th0 - th2 + 4.0 * M_PI, 2.0 * M_PI);
        const double t    = std::fmod(th  - th2 + 4.0 * M_PI, 2.0 * M_PI);
        return t <= span + 1e-10;
    }
}

// Unsigned distance from p to the circular arc.
double dist_arc(Point2D p, const ArcParams& arc) noexcept {
    const double d_center = std::hypot(p.x - arc.cx, p.y - arc.cy);
    const double th = (d_center > 1e-15)
        ? std::atan2(p.y - arc.cy, p.x - arc.cx)
        : arc.th0;

    if (angle_in_arc(arc.th0, arc.th2, arc.ccw, th)) {
        return std::fabs(d_center - arc.r);
    }
    // Nearest is one of the two endpoints.
    const double d0 = std::hypot(p.x - (arc.cx + arc.r * std::cos(arc.th0)),
                                 p.y - (arc.cy + arc.r * std::sin(arc.th0)));
    const double d2 = std::hypot(p.x - (arc.cx + arc.r * std::cos(arc.th2)),
                                 p.y - (arc.cy + arc.r * std::sin(arc.th2)));
    return std::min(d0, d2);
}

// Winding-number contribution of an arc at query point q (+x ray).
int winding_arc(Point2D q, const ArcParams& arc) noexcept {
    const double sin_val = clamp((q.y - arc.cy) / arc.r, -1.0, 1.0);
    if (std::fabs((q.y - arc.cy) / arc.r) > 1.0 + 1e-9) return 0;
    int wind = 0;
    // Two solutions: th_a ∈ [−π/2, π/2] and its supplement th_b = π − th_a.
    const double th_a = std::asin(sin_val);
    const double th_b = M_PI - th_a;
    for (double th : {th_a, th_b}) {
        if (!angle_in_arc(arc.th0, arc.th2, arc.ccw, th)) continue;
        const double x_int = arc.cx + arc.r * std::cos(th);
        if (x_int <= q.x) continue;
        // y-component of the arc tangent: CCW → (−sin θ, cos θ); CW → (sin θ, −cos θ)
        const double ty = arc.ccw ? std::cos(th) : -std::cos(th);
        if      (ty > 1e-12)  wind += 1;
        else if (ty < -1e-12) wind -= 1;
        // ty ≈ 0: tangential grazing — skip
    }
    return wind;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bézier curves
// ─────────────────────────────────────────────────────────────────────────────

// ── Quadratic Bézier ─────────────────────────────────────────────────────────

inline std::pair<double, double>
bezier2_eval(Point2D p0, Point2D p1, Point2D p2, double t) noexcept {
    const double s = 1.0 - t;
    return {s*s*p0.x + 2*s*t*p1.x + t*t*p2.x,
            s*s*p0.y + 2*s*t*p1.y + t*t*p2.y};
}

// Find the t ∈ [0,1] that minimises distance from p to a parametric curve
// `eval_fn` using golden-section search within an initial bracket [lo, hi].
template <typename EvalFn>
double golden_section_min_dist(Point2D p, EvalFn eval_fn,
                               double lo, double hi) noexcept {
    constexpr double phi = 0.6180339887498949; // (√5 − 1) / 2
    for (int i = 0; i < 25; ++i) {
        const double m1 = hi - phi * (hi - lo);
        const double m2 = lo + phi * (hi - lo);
        auto [x1, y1] = eval_fn(m1);
        auto [x2, y2] = eval_fn(m2);
        const double d1 = (p.x-x1)*(p.x-x1) + (p.y-y1)*(p.y-y1);
        const double d2 = (p.x-x2)*(p.x-x2) + (p.y-y2)*(p.y-y2);
        if (d1 < d2) hi = m2; else lo = m1;
    }
    const double tm = (lo + hi) / 2.0;
    auto [xf, yf] = eval_fn(tm);
    return std::hypot(p.x - xf, p.y - yf);
}

// Sample a curve at n_samples uniform steps, return the distance from p.
// Returns {best_distance, best_t, bracket_lo, bracket_hi}.
template <typename EvalFn>
std::array<double, 4>
curve_sample(Point2D p, EvalFn eval_fn, int n_samples) noexcept {
    double best_d2 = std::numeric_limits<double>::max();
    double best_t  = 0.0;
    for (int i = 0; i <= n_samples; ++i) {
        const double t = static_cast<double>(i) / n_samples;
        auto [x, y] = eval_fn(t);
        const double d2 = (p.x-x)*(p.x-x) + (p.y-y)*(p.y-y);
        if (d2 < best_d2) { best_d2 = d2; best_t = t; }
    }
    const double step = 1.0 / n_samples;
    return {std::sqrt(best_d2), best_t,
            std::max(0.0, best_t - step),
            std::min(1.0, best_t + step)};
}

// Unsigned distance from p to a quadratic Bézier curve.
double dist_bezier2(Point2D p, Point2D p0, Point2D p1, Point2D p2) noexcept {
    auto fn = [&](double t) { return bezier2_eval(p0, p1, p2, t); };
    auto [d, bt, lo, hi] = curve_sample(p, fn, 16);
    (void)d;
    return golden_section_min_dist(p, fn, lo, hi);
}

// Winding-number contribution of a quadratic Bézier at q (+x ray).
// Solves the quadratic B_y(t) = q.y analytically.
int winding_bezier2(Point2D q, Point2D p0, Point2D p1, Point2D p2) noexcept {
    const double a = p0.y - 2*p1.y + p2.y;
    const double b = 2*(p1.y - p0.y);
    const double c = p0.y - q.y;

    int wind = 0;
    auto process_t = [&](double t) {
        if (t < -1e-9 || t > 1.0 + 1e-9) return;
        t = clamp(t, 0.0, 1.0);
        auto [bx, by_] = bezier2_eval(p0, p1, p2, t);
        (void)by_;
        if (bx <= q.x) return;
        // dy = B'_y(t) = 2(p1−p0).y + 2(p0−2p1+p2).y * t
        const double dy = b + 2*a*t;
        if      (dy > 1e-12)  wind += 1;
        else if (dy < -1e-12) wind -= 1;
    };

    if (std::fabs(a) < 1e-12) {
        if (std::fabs(b) > 1e-12) process_t(-c / b);
    } else {
        const double disc = b*b - 4*a*c;
        if (disc >= 0) {
            const double sq = std::sqrt(disc);
            process_t((-b - sq) / (2*a));
            process_t((-b + sq) / (2*a));
        }
    }
    return wind;
}

// ── Cubic Bézier ─────────────────────────────────────────────────────────────

inline std::pair<double, double>
bezier3_eval(Point2D p0, Point2D p1, Point2D p2, Point2D p3, double t) noexcept {
    const double s = 1.0 - t;
    return {s*s*s*p0.x + 3*s*s*t*p1.x + 3*s*t*t*p2.x + t*t*t*p3.x,
            s*s*s*p0.y + 3*s*s*t*p1.y + 3*s*t*t*p2.y + t*t*t*p3.y};
}

// Unsigned distance from p to a cubic Bézier curve.
double dist_bezier3(Point2D p, Point2D p0, Point2D p1,
                    Point2D p2, Point2D p3) noexcept {
    auto fn = [&](double t) { return bezier3_eval(p0, p1, p2, p3, t); };
    auto [d, bt, lo, hi] = curve_sample(p, fn, 16);
    (void)d;
    return golden_section_min_dist(p, fn, lo, hi);
}

// Winding-number contribution of a cubic Bézier at q (+x ray).
// Finds t ∈ [0,1] where B_y(t) = q.y using monotone-interval bisection.
int winding_bezier3(Point2D q, Point2D p0, Point2D p1,
                    Point2D p2, Point2D p3) noexcept {
    // Coefficients of B_y(t) = Ay*t³ + By*t² + Cy*t + Dy + q.y (shifted)
    const double Ay = -p0.y + 3*p1.y - 3*p2.y + p3.y;
    const double By =  3*p0.y - 6*p1.y + 3*p2.y;
    const double Cy = -3*p0.y + 3*p1.y;
    // Dy = p0.y  (B_y(0) = p0.y)

    // Critical points of B_y: solve B'_y(t) = 3Ay*t² + 2By*t + Cy = 0
    std::vector<double> breaks = {0.0, 1.0};
    if (std::fabs(3*Ay) > 1e-12) {
        const double disc = 4*By*By - 12*Ay*Cy;
        if (disc >= 0) {
            const double sq = std::sqrt(disc);
            const double t1 = (-2*By - sq) / (6*Ay);
            const double t2 = (-2*By + sq) / (6*Ay);
            if (t1 > 1e-9 && t1 < 1.0 - 1e-9) breaks.push_back(t1);
            if (t2 > 1e-9 && t2 < 1.0 - 1e-9) breaks.push_back(t2);
        }
    } else if (std::fabs(2*By) > 1e-12) {
        const double t1 = -Cy / (2*By);
        if (t1 > 1e-9 && t1 < 1.0 - 1e-9) breaks.push_back(t1);
    }
    std::sort(breaks.begin(), breaks.end());

    auto eval_y = [&](double t) -> double {
        const double s = 1-t;
        return s*s*s*p0.y + 3*s*s*t*p1.y + 3*s*t*t*p2.y + t*t*t*p3.y;
    };
    auto eval_x = [&](double t) -> double {
        const double s = 1-t;
        return s*s*s*p0.x + 3*s*s*t*p1.x + 3*s*t*t*p2.x + t*t*t*p3.x;
    };
    auto eval_dy = [&](double t) -> double {
        return 3*Ay*t*t + 2*By*t + Cy;
    };

    int wind = 0;
    auto process_t = [&](double t) {
        const double x_int = eval_x(t);
        if (x_int <= q.x) return;
        const double dy = eval_dy(t);
        if      (dy > 1e-12)  wind += 1;
        else if (dy < -1e-12) wind -= 1;
    };

    for (std::size_t i = 0; i + 1 < breaks.size(); ++i) {
        const double ta = breaks[i], tb = breaks[i+1];
        const double fa = eval_y(ta) - q.y;
        const double fb = eval_y(tb) - q.y;
        if (fa * fb > 0) continue;      // no root in this monotone interval
        if (std::fabs(fa) < 1e-12) { process_t(ta); continue; }
        if (std::fabs(fb) < 1e-12) continue;  // handled when i increments
        // Bisect to find the unique root
        double lo = ta, hi = tb, flo = fa;
        for (int iter = 0; iter < 20; ++iter) {
            const double mid = (lo + hi) / 2.0;
            const double fm  = eval_y(mid) - q.y;
            if (std::fabs(fm) < 1e-12) { lo = hi = mid; break; }
            if (fm * flo < 0) { hi = mid; }
            else              { lo = mid; flo = fm; }
        }
        process_t((lo + hi) / 2.0);
    }
    return wind;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-segment dispatch
// ─────────────────────────────────────────────────────────────────────────────

double segment_dist(Point2D p, const PwclSegment& seg) noexcept {
    switch (seg.type) {
    case SegmentType::LINE:
        return dist_line_segment(p, seg.pts[0], seg.pts[1]);
    case SegmentType::ARC: {
        const ArcParams arc = arc_from_3pts(seg.pts[0], seg.pts[1], seg.pts[2]);
        if (arc.degenerate)
            return dist_line_segment(p, seg.pts[0], seg.pts[2]);
        return dist_arc(p, arc);
    }
    case SegmentType::BEZIER2:
        return dist_bezier2(p, seg.pts[0], seg.pts[1], seg.pts[2]);
    case SegmentType::BEZIER3:
        return dist_bezier3(p, seg.pts[0], seg.pts[1], seg.pts[2], seg.pts[3]);
    }
    return std::numeric_limits<double>::max();
}

int segment_winding(Point2D q, const PwclSegment& seg) noexcept {
    switch (seg.type) {
    case SegmentType::LINE:
        return winding_line(q, seg.pts[0], seg.pts[1]);
    case SegmentType::ARC: {
        const ArcParams arc = arc_from_3pts(seg.pts[0], seg.pts[1], seg.pts[2]);
        if (arc.degenerate)
            return winding_line(q, seg.pts[0], seg.pts[2]);
        return winding_arc(q, arc);
    }
    case SegmentType::BEZIER2:
        return winding_bezier2(q, seg.pts[0], seg.pts[1], seg.pts[2]);
    case SegmentType::BEZIER3:
        return winding_bezier3(q, seg.pts[0], seg.pts[1], seg.pts[2], seg.pts[3]);
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// SDF at a single point
// ─────────────────────────────────────────────────────────────────────────────

double compute_sdf_nm(Point2D p, const std::vector<PwclContour>& contours) noexcept {
    double min_dist = std::numeric_limits<double>::max();
    int    winding  = 0;
    for (const auto& contour : contours) {
        for (const auto& seg : contour.segments) {
            const double d = segment_dist(p, seg);
            if (d < min_dist) min_dist = d;
            winding += segment_winding(p, seg);
        }
    }
    // Non-zero winding → inside → negative SDF
    return (winding != 0) ? -min_dist : +min_dist;
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment bounding box
// ─────────────────────────────────────────────────────────────────────────────

struct SegBBox { double xmin, ymin, xmax, ymax; };

SegBBox segment_bbox(const PwclSegment& seg) noexcept {
    double xmin, xmax, ymin, ymax;
    switch (seg.type) {
    case SegmentType::LINE:
        xmin = std::min(seg.pts[0].x, seg.pts[1].x);
        xmax = std::max(seg.pts[0].x, seg.pts[1].x);
        ymin = std::min(seg.pts[0].y, seg.pts[1].y);
        ymax = std::max(seg.pts[0].y, seg.pts[1].y);
        break;
    case SegmentType::ARC: {
        const ArcParams arc = arc_from_3pts(seg.pts[0], seg.pts[1], seg.pts[2]);
        if (arc.degenerate) {
            xmin = std::min(seg.pts[0].x, seg.pts[2].x); xmax = std::max(seg.pts[0].x, seg.pts[2].x);
            ymin = std::min(seg.pts[0].y, seg.pts[2].y); ymax = std::max(seg.pts[0].y, seg.pts[2].y);
        } else {
            xmin = arc.cx - arc.r; xmax = arc.cx + arc.r;
            ymin = arc.cy - arc.r; ymax = arc.cy + arc.r;
        }
        break;
    }
    case SegmentType::BEZIER2:
        xmin = std::min({seg.pts[0].x, seg.pts[1].x, seg.pts[2].x});
        xmax = std::max({seg.pts[0].x, seg.pts[1].x, seg.pts[2].x});
        ymin = std::min({seg.pts[0].y, seg.pts[1].y, seg.pts[2].y});
        ymax = std::max({seg.pts[0].y, seg.pts[1].y, seg.pts[2].y});
        break;
    case SegmentType::BEZIER3:
        xmin = std::min({seg.pts[0].x, seg.pts[1].x, seg.pts[2].x, seg.pts[3].x});
        xmax = std::max({seg.pts[0].x, seg.pts[1].x, seg.pts[2].x, seg.pts[3].x});
        ymin = std::min({seg.pts[0].y, seg.pts[1].y, seg.pts[2].y, seg.pts[3].y});
        ymax = std::max({seg.pts[0].y, seg.pts[1].y, seg.pts[2].y, seg.pts[3].y});
        break;
    default:
        xmin = xmax = ymin = ymax = 0.0;
    }
    return {xmin, ymin, xmax, ymax};
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial grid — maps nm space to cells, each cell stores nearby segment indices.
// Used for distance queries: a pixel only needs to check segments in its cell.
// Cells are pre-expanded by trunc_nm so every segment within truncation distance
// is guaranteed to appear in the pixel's cell.
// ─────────────────────────────────────────────────────────────────────────────

struct SpatialGrid {
    double x0 = 0, y0 = 0, cell_nm = 1;
    int nx = 0, ny = 0;
    std::vector<std::vector<int>> cells;

    void build(const std::vector<SegBBox>& bboxes, double trunc_nm,
               double gx0, double gy0, double width_nm, double height_nm) {
        x0 = gx0; y0 = gy0;
        cell_nm = std::max(trunc_nm, 1e-6);
        nx = static_cast<int>(std::ceil(width_nm  / cell_nm)) + 2;
        ny = static_cast<int>(std::ceil(height_nm / cell_nm)) + 2;
        cells.assign(static_cast<std::size_t>(nx) * ny, {});
        for (int i = 0; i < static_cast<int>(bboxes.size()); ++i) {
            const auto& b = bboxes[i];
            int cx0 = std::max(0,      (int)std::floor((b.xmin - trunc_nm - x0) / cell_nm));
            int cy0 = std::max(0,      (int)std::floor((b.ymin - trunc_nm - y0) / cell_nm));
            int cx1 = std::min(nx - 1, (int)std::ceil( (b.xmax + trunc_nm - x0) / cell_nm));
            int cy1 = std::min(ny - 1, (int)std::ceil( (b.ymax + trunc_nm - y0) / cell_nm));
            for (int cy = cy0; cy <= cy1; ++cy)
                for (int cx = cx0; cx <= cx1; ++cx)
                    cells[static_cast<std::size_t>(cy) * nx + cx].push_back(i);
        }
    }

    const std::vector<int>& at(int cx, int cy) const noexcept {
        static const std::vector<int> empty;
        if (cx < 0 || cy < 0 || cx >= nx || cy >= ny) return empty;
        return cells[static_cast<std::size_t>(cy) * nx + cx];
    }

    int cell_x(double x_nm) const noexcept { return (int)std::floor((x_nm - x0) / cell_nm); }
    int cell_y(double y_nm) const noexcept { return (int)std::floor((y_nm - y0) / cell_nm); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Scanline crossing helper — collects (x_nm, sign) crossing pairs for a row.
// Used to precompute winding numbers for an entire row at once (O(N_segs/row)
// instead of O(N_segs) per pixel).
// ─────────────────────────────────────────────────────────────────────────────

void segment_crossings(const PwclSegment& seg, double y_nm,
                       std::vector<std::pair<double, int>>& out) noexcept {
    switch (seg.type) {
    case SegmentType::LINE: {
        const Point2D a = seg.pts[0], b = seg.pts[1];
        const bool up = (a.y <= y_nm && b.y >  y_nm);
        const bool dn = (a.y >  y_nm && b.y <= y_nm);
        if (!up && !dn) return;
        out.push_back({a.x + (y_nm - a.y) * (b.x - a.x) / (b.y - a.y), up ? +1 : -1});
        break;
    }
    case SegmentType::ARC: {
        const ArcParams arc = arc_from_3pts(seg.pts[0], seg.pts[1], seg.pts[2]);
        if (arc.degenerate) {
            const Point2D a = seg.pts[0], b = seg.pts[2];
            const bool up = (a.y <= y_nm && b.y > y_nm);
            const bool dn = (a.y >  y_nm && b.y <= y_nm);
            if (!up && !dn) return;
            out.push_back({a.x + (y_nm - a.y) * (b.x - a.x) / (b.y - a.y), up ? +1 : -1});
            return;
        }
        if (std::fabs((y_nm - arc.cy) / arc.r) > 1.0 + 1e-9) return;
        const double sv = clamp((y_nm - arc.cy) / arc.r, -1.0, 1.0);
        for (double th : {std::asin(sv), M_PI - std::asin(sv)}) {
            if (!angle_in_arc(arc.th0, arc.th2, arc.ccw, th)) continue;
            const double ty = arc.ccw ? std::cos(th) : -std::cos(th);
            if (std::fabs(ty) < 1e-12) continue;
            out.push_back({arc.cx + arc.r * std::cos(th), ty > 0 ? +1 : -1});
        }
        break;
    }
    case SegmentType::BEZIER2: {
        const Point2D &p0 = seg.pts[0], &p1 = seg.pts[1], &p2 = seg.pts[2];
        const double a = p0.y - 2*p1.y + p2.y, bv = 2*(p1.y - p0.y), c = p0.y - y_nm;
        auto proc2 = [&](double t) {
            if (t < -1e-9 || t > 1.0 + 1e-9) return;
            t = clamp(t, 0.0, 1.0);
            const double s = 1-t;
            const double dy = bv + 2*a*t;
            if (std::fabs(dy) < 1e-12) return;
            out.push_back({s*s*p0.x + 2*s*t*p1.x + t*t*p2.x, dy > 0 ? +1 : -1});
        };
        if (std::fabs(a) < 1e-12) { if (std::fabs(bv) > 1e-12) proc2(-c / bv); }
        else {
            const double disc = bv*bv - 4*a*c;
            if (disc >= 0) { const double sq = std::sqrt(disc); proc2((-bv-sq)/(2*a)); proc2((-bv+sq)/(2*a)); }
        }
        break;
    }
    case SegmentType::BEZIER3: {
        const Point2D &p0=seg.pts[0], &p1=seg.pts[1], &p2=seg.pts[2], &p3=seg.pts[3];
        const double Ay=-p0.y+3*p1.y-3*p2.y+p3.y, By=3*p0.y-6*p1.y+3*p2.y, Cy=-3*p0.y+3*p1.y;
        std::vector<double> brk = {0.0, 1.0};
        if (std::fabs(3*Ay) > 1e-12) {
            const double disc = 4*By*By - 12*Ay*Cy;
            if (disc >= 0) { const double sq=std::sqrt(disc);
                double t1=(-2*By-sq)/(6*Ay), t2=(-2*By+sq)/(6*Ay);
                if (t1>1e-9 && t1<1-1e-9) brk.push_back(t1);
                if (t2>1e-9 && t2<1-1e-9) brk.push_back(t2); }
        } else if (std::fabs(2*By) > 1e-12) {
            double t1=-Cy/(2*By); if (t1>1e-9 && t1<1-1e-9) brk.push_back(t1);
        }
        std::sort(brk.begin(), brk.end());
        auto ey=[&](double t){const double s=1-t; return s*s*s*p0.y+3*s*s*t*p1.y+3*s*t*t*p2.y+t*t*t*p3.y;};
        auto ex=[&](double t){const double s=1-t; return s*s*s*p0.x+3*s*s*t*p1.x+3*s*t*t*p2.x+t*t*t*p3.x;};
        auto dy=[&](double t){ return 3*Ay*t*t+2*By*t+Cy; };
        for (std::size_t i=0; i+1<brk.size(); ++i) {
            double ta=brk[i], tb=brk[i+1], fa=ey(ta)-y_nm, fb=ey(tb)-y_nm;
            if (fa*fb > 0) continue;
            if (std::fabs(fa) < 1e-12) { if (std::fabs(dy(ta))>1e-12) out.push_back({ex(ta), dy(ta)>0?+1:-1}); continue; }
            if (std::fabs(fb) < 1e-12) continue;
            double lo=ta, hi=tb, flo=fa;
            for (int it=0; it<20; ++it) {
                double mid=(lo+hi)/2.0, fm=ey(mid)-y_nm;
                if (std::fabs(fm)<1e-12){lo=hi=mid;break;}
                if (fm*flo<0) hi=mid; else {lo=mid; flo=fm;}
            }
            double tr=(lo+hi)/2.0;
            if (std::fabs(dy(tr))>1e-12) out.push_back({ex(tr), dy(tr)>0?+1:-1});
        }
        break;
    }
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public CPU rasterizer
// ─────────────────────────────────────────────────────────────────────────────

void rasterize_csdf_cpu(
    const std::vector<PwclContour>& contours,
    double  origin_x_nm,
    double  origin_y_nm,
    int     patch_size_px,
    double  grid_res_nm_per_px,
    double  truncation_px,
    float*  out)
{
    const double t_nm    = truncation_px * grid_res_nm_per_px;
    const double inv_2t  = 1.0 / (2.0 * t_nm);

#pragma omp parallel for schedule(dynamic, 4)
    for (int row = 0; row < patch_size_px; ++row) {
        const double y_nm = origin_y_nm + (row + 0.5) * grid_res_nm_per_px;
        for (int col = 0; col < patch_size_px; ++col) {
            const double x_nm = origin_x_nm + (col + 0.5) * grid_res_nm_per_px;
            const double sdf  = compute_sdf_nm({x_nm, y_nm}, contours);
            const double clamped = clamp(sdf, -t_nm, t_nm);
            // 1.0 = inside (SDF ≤ −t), 0.0 = outside (SDF ≥ +t), 0.5 = boundary.
            // SDF is negative inside, so we invert: csdf = (t − clamped) / (2t).
            out[row * patch_size_px + col] =
                static_cast<float>((t_nm - clamped) * inv_2t);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch canvas rasterizer
// ─────────────────────────────────────────────────────────────────────────────

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
    float*  out)
{
    const double t_nm   = truncation_px * grid_res_nm_per_px;
    const double inv_2t = 1.0 / (2.0 * t_nm);

    for (int m = 0; m < n_markers; ++m) {
        const int    c0   = marker_c_start[m];
        const int    c1   = marker_c_end[m];
        const double ox   = marker_ox[m];
        const double oy   = marker_oy[m];
        const int    S    = marker_sizes[m];
        const int    col0 = static_cast<int>(std::round((ox - canvas_x0_nm) / grid_res_nm_per_px));
        const int    row0 = static_cast<int>(std::round((oy - canvas_y0_nm) / grid_res_nm_per_px));

        // ── Flatten segments for this marker ─────────────────────────────────
        std::vector<PwclSegment> segs;
        std::vector<SegBBox>     bboxes;
        for (int c = c0; c < c1; ++c)
            for (const auto& seg : all_contours[c].segments) {
                segs.push_back(seg);
                bboxes.push_back(segment_bbox(seg));
            }
        if (segs.empty()) continue;

        // ── Spatial grid for distance queries ─────────────────────────────────
        // Each cell stores indices of segments whose AABB (expanded by t_nm)
        // overlaps the cell.  A pixel only queries its own cell → O(N_local).
        SpatialGrid grid;
        grid.build(bboxes, t_nm, ox, oy,
                   S * grid_res_nm_per_px, S * grid_res_nm_per_px);

        // ── Scanline winding precomputation (parallel, y-culled) ─────────────
        // For each row we fire horizontal +x rays and accumulate crossings into
        // a difference array → prefix-sum → per-pixel winding number.
        //
        // Optimisations vs. per-pixel winding:
        //   • Y-cull: skip segments whose bbox y-range doesn't include this row.
        //     IC polygon edges are short relative to the full marker height, so
        //     typically only ~1-5% of segments are relevant per row.
        //   • Parallel: each row is independent → #pragma omp parallel for.
        //     Thread-local delta/crossings buffers avoid false sharing.
        std::vector<int> winding_map(static_cast<std::size_t>(S) * S, 0);
        {
            const int N_segs = static_cast<int>(segs.size());
#pragma omp parallel for schedule(dynamic, 8)
            for (int row = 0; row < S; ++row) {
                const double y_nm = oy + (row + 0.5) * grid_res_nm_per_px;

                // Thread-local buffers (stack-allocated or small heap).
                std::vector<int>                    delta(S + 1, 0);
                std::vector<std::pair<double, int>> crossings;

                for (int si = 0; si < N_segs; ++si) {
                    // Y-cull: bbox already accounts for arc/bezier extent.
                    const auto& b = bboxes[si];
                    if (y_nm < b.ymin - 1e-9 || y_nm > b.ymax + 1e-9) continue;
                    segment_crossings(segs[si], y_nm, crossings);
                }

                for (auto [x_int, sign] : crossings) {
                    const double x_col = (x_int - ox) / grid_res_nm_per_px;
                    int col_thresh = static_cast<int>(std::ceil(x_col - 0.5));
                    col_thresh = std::max(0, std::min(S, col_thresh));
                    delta[0]          += sign;
                    delta[col_thresh] -= sign;
                }

                int w = 0;
                for (int col = 0; col < S; ++col) {
                    w += delta[col];
                    winding_map[static_cast<std::size_t>(row) * S + col] = w;
                }
            }
        }

        // ── Per-pixel rasterization (distance via grid, winding from map) ─────
#pragma omp parallel for schedule(dynamic, 8)
        for (int row = 0; row < S; ++row) {
            const int cr = row0 + row;
            if (cr < 0 || cr >= canvas_H) continue;
            const double y_nm = oy + (row + 0.5) * grid_res_nm_per_px;
            const int gcy = grid.cell_y(y_nm);

            for (int col = 0; col < S; ++col) {
                const int cc = col0 + col;
                if (cc < 0 || cc >= canvas_W) continue;
                const double x_nm = ox + (col + 0.5) * grid_res_nm_per_px;
                const int gcx = grid.cell_x(x_nm);

                // Distance: only query segments in this grid cell.
                // Bbox early-out: if the segment's AABB is farther than the
                // current best distance, skip the expensive curve computation.
                double min_dist = t_nm + 1.0;  // default: beyond truncation
                for (int si : grid.at(gcx, gcy)) {
                    const auto& b = bboxes[si];
                    const double bdx = std::max(0.0, std::max(b.xmin - x_nm, x_nm - b.xmax));
                    const double bdy = std::max(0.0, std::max(b.ymin - y_nm, y_nm - b.ymax));
                    if (bdx * bdx + bdy * bdy >= min_dist * min_dist) continue;
                    const double d = segment_dist({x_nm, y_nm}, segs[si]);
                    if (d < min_dist) min_dist = d;
                }

                const int    winding = winding_map[static_cast<std::size_t>(row) * S + col];
                const double sdf     = (winding != 0) ? -min_dist : +min_dist;
                const double clamped = clamp(sdf, -t_nm, t_nm);
                out[cr * canvas_W + cc] =
                    static_cast<float>((t_nm - clamped) * inv_2t);
            }
        }
    }
}

}  // namespace csdf

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 module  (csdf_ext)
// ─────────────────────────────────────────────────────────────────────────────

namespace py = pybind11;

PYBIND11_MODULE(csdf_ext, m) {
    m.doc() = "cSDF rasterizer C++ extension — CPU/OpenMP path";

    py::enum_<csdf::SegmentType>(m, "SegmentType")
        .value("LINE",    csdf::SegmentType::LINE)
        .value("ARC",     csdf::SegmentType::ARC)
        .value("BEZIER2", csdf::SegmentType::BEZIER2)
        .value("BEZIER3", csdf::SegmentType::BEZIER3)
        .export_values();

    // Numpy-based rasterize entry point.
    //
    // Parameters (all numpy arrays must be C-contiguous):
    //   seg_types   – int32[N]       segment type per segment
    //   seg_pts     – float64[N,4,2] control points in nm (padded with 0)
    //   contour_ids – int32[N]       0-based contour index for each segment
    //   is_hole     – bool[C]        is_hole flag per contour
    //   origin_x_nm, origin_y_nm – patch bottom-left corner (nm)
    //   patch_size_px             – S (side length of output square)
    //   grid_res_nm_per_px        – nm per pixel
    //   truncation_px             – SDF truncation half-width (pixels)
    //
    // Returns float32[S, S] cSDF patch (row 0 = y_min).
    m.def(
        "rasterize_csdf_cpu",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> seg_types,
           py::array_t<double,  py::array::c_style | py::array::forcecast> seg_pts,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> contour_ids,
           py::array_t<bool,    py::array::c_style | py::array::forcecast> is_hole,
           double origin_x_nm,
           double origin_y_nm,
           int    patch_size_px,
           double grid_res_nm_per_px,
           double truncation_px) -> py::array_t<float>
        {
            const auto  t_buf  = seg_types.request();
            const auto  p_buf  = seg_pts.request();
            const auto  id_buf = contour_ids.request();
            const auto  h_buf  = is_hole.request();

            const int N = static_cast<int>(t_buf.shape[0]);
            const int C = static_cast<int>(h_buf.shape[0]);

            const auto* t_ptr  = static_cast<const int32_t*>(t_buf.ptr);
            const auto* p_ptr  = static_cast<const double*>(p_buf.ptr);
            const auto* id_ptr = static_cast<const int32_t*>(id_buf.ptr);
            const auto* h_ptr  = static_cast<const bool*>(h_buf.ptr);

            // Build contour vector.
            std::vector<csdf::PwclContour> contours(C);
            for (int c = 0; c < C; ++c)
                contours[c].is_hole = h_ptr[c];

            for (int i = 0; i < N; ++i) {
                csdf::PwclSegment seg;
                seg.type = static_cast<csdf::SegmentType>(t_ptr[i]);
                for (int k = 0; k < 4; ++k) {
                    seg.pts[k].x = p_ptr[(i * 4 + k) * 2 + 0];
                    seg.pts[k].y = p_ptr[(i * 4 + k) * 2 + 1];
                }
                const int c = id_ptr[i];
                if (c >= 0 && c < C)
                    contours[c].segments.push_back(seg);
            }

            // Allocate output (must hold GIL).
            py::array_t<float> out({patch_size_px, patch_size_px});
            float* out_ptr = static_cast<float*>(out.request().ptr);

            // Release GIL for the pure-C++ computation so multiple Python
            // threads can rasterize patches concurrently.
            {
                py::gil_scoped_release release;
                csdf::rasterize_csdf_cpu(
                    contours,
                    origin_x_nm, origin_y_nm,
                    patch_size_px,
                    grid_res_nm_per_px,
                    truncation_px,
                    out_ptr);
            }
            return out;
        },
        py::arg("seg_types"),
        py::arg("seg_pts"),
        py::arg("contour_ids"),
        py::arg("is_hole"),
        py::arg("origin_x_nm"),
        py::arg("origin_y_nm"),
        py::arg("patch_size_px"),
        py::arg("grid_res_nm_per_px"),
        py::arg("truncation_px"),
        R"pbdoc(
Rasterize PWCL contours into a float32 cSDF patch of shape [S, S].

Segment types
    0 = LINE    pts[0..1] in nm
    1 = ARC     pts[0]=start, pts[1]=on-arc-mid, pts[2]=end  (nm)
    2 = BEZIER2 pts[0]=p0, pts[1]=ctrl, pts[2]=p2  (nm)
    3 = BEZIER3 pts[0..3] = p0,c1,c2,p3  (nm)

cSDF value convention
    1.0  fully inside  (SDF ≤ −t)
    0.5  on boundary   (SDF = 0)
    0.0  fully outside (SDF ≥ +t)

Row 0 of the output corresponds to y_min (bottom of the marker box).
)pbdoc");

    // ── Batch canvas rasterizer ───────────────────────────────────────────────
    m.def(
        "rasterize_csdf_canvas",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> seg_types,
           py::array_t<double,  py::array::c_style | py::array::forcecast> seg_pts,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> contour_ids,
           py::array_t<bool,    py::array::c_style | py::array::forcecast> is_hole,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> marker_c_start,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> marker_c_end,
           py::array_t<double,  py::array::c_style | py::array::forcecast> marker_ox,
           py::array_t<double,  py::array::c_style | py::array::forcecast> marker_oy,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> marker_sizes,
           double canvas_x0_nm,
           double canvas_y0_nm,
           int    canvas_H,
           int    canvas_W,
           double grid_res_nm_per_px,
           double truncation_px) -> py::array_t<float>
        {
            // Extract raw pointers (GIL held).
            const int N = static_cast<int>(seg_types.request().shape[0]);
            const int C = static_cast<int>(is_hole.request().shape[0]);
            const int M = static_cast<int>(marker_c_start.request().shape[0]);

            const auto* t_ptr   = seg_types.data();
            const auto* p_ptr   = seg_pts.data();
            const auto* id_ptr  = contour_ids.data();
            const auto* h_ptr   = is_hole.data();
            const auto* cs_ptr  = marker_c_start.data();
            const auto* ce_ptr  = marker_c_end.data();
            const auto* ox_ptr  = marker_ox.data();
            const auto* oy_ptr  = marker_oy.data();
            const auto* sz_ptr  = marker_sizes.data();

            // Build contour vector.
            std::vector<csdf::PwclContour> contours(C);
            for (int c = 0; c < C; ++c)
                contours[c].is_hole = h_ptr[c];
            for (int i = 0; i < N; ++i) {
                csdf::PwclSegment seg;
                seg.type = static_cast<csdf::SegmentType>(t_ptr[i]);
                for (int k = 0; k < 4; ++k) {
                    seg.pts[k].x = p_ptr[(i * 4 + k) * 2 + 0];
                    seg.pts[k].y = p_ptr[(i * 4 + k) * 2 + 1];
                }
                const int c = id_ptr[i];
                if (c >= 0 && c < C)
                    contours[c].segments.push_back(seg);
            }

            // Allocate canvas output (GIL held).
            py::array_t<float> out({canvas_H, canvas_W});
            float* out_ptr = out.mutable_data();
            std::fill(out_ptr, out_ptr + canvas_H * canvas_W, 0.0f);

            // Release GIL and rasterize all markers in one C++ call.
            {
                py::gil_scoped_release release;
                csdf::rasterize_csdf_canvas(
                    contours,
                    cs_ptr, ce_ptr, ox_ptr, oy_ptr, sz_ptr, M,
                    canvas_x0_nm, canvas_y0_nm,
                    grid_res_nm_per_px, truncation_px,
                    canvas_H, canvas_W,
                    out_ptr);
            }
            return out;
        },
        py::arg("seg_types"), py::arg("seg_pts"), py::arg("contour_ids"),
        py::arg("is_hole"),
        py::arg("marker_c_start"), py::arg("marker_c_end"),
        py::arg("marker_ox"), py::arg("marker_oy"), py::arg("marker_sizes"),
        py::arg("canvas_x0_nm"), py::arg("canvas_y0_nm"),
        py::arg("canvas_H"), py::arg("canvas_W"),
        py::arg("grid_res_nm_per_px"), py::arg("truncation_px"),
        "Batch-rasterize all marker regions into a single float32 [H, W] canvas.");
}

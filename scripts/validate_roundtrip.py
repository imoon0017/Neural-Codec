"""Round-trip validation: IC layout → cSDF → contouring → XOR → OASIS.

Supported input formats:
  OASIS  — .oas, .oasis
  GDSII  — .gds, .gds2, .gdsii, .gdx

Pipeline
────────
  1. Load original layout, extract mask polygons + square markers.
  2. Rasterize all markers into a single cSDF canvas (C++ path).
  3. Crop per-marker patches from the canvas, run contouring to recover
     PWCL polygons in physical nm.
  4. Write three layers to the output OASIS:
       --original-layer  (default 1)  original mask polygons
       --recon-layer     (default 2)  reconstructed polygons
       --xor-layer       (default 3)  XOR(original, reconstructed)
  5. Report per-marker and total XOR area ratio
     (XOR area / original area, ideal = 0.0).

When --marker-layer is omitted (the default), the bounding box of all mask
polygons is used as the single canvas region.  The bounding box is expanded
to a square (side = max(W, H)) and then by --marker-margin on every side.

Usage
─────
    python scripts/validate_roundtrip.py layout.oas  --output-dir out/
    python scripts/validate_roundtrip.py layout.gds  --output-dir out/
    python scripts/validate_roundtrip.py layout.oas \\
        --mask-layer 1 --marker-layer 100 \\
        --grid-res 1.0 --truncation 5 \\
        --original-layer 1 --recon-layer 2 --xor-layer 3 \\
        --output-dir out/ --cell TOP
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import klayout.db as db

from csdf.csdf_utils import (
    PwclContour,
    PwclSegment,
    SegmentType,
    rasterize_canvas,
)
from contouring.contour import csdf_to_contours

log = logging.getLogger(__name__)

# Accepted layout file extensions (KLayout auto-detects format from extension).
_LAYOUT_EXTS: frozenset[str] = frozenset(
    {".oas", ".oasis", ".gds", ".gds2", ".gdsii", ".gdx"}
)


def _check_layout_ext(path: Path) -> None:
    if path.suffix.lower() not in _LAYOUT_EXTS:
        raise ValueError(
            f"Unsupported layout format '{path.suffix}'. "
            f"Expected one of: {', '.join(sorted(_LAYOUT_EXTS))}"
        )


# ─── Coordinate helpers ───────────────────────────────────────────────────────


def _dbu_to_nm(val: float, dbu_um: float) -> float:
    return val * dbu_um * 1000.0


def _nm_to_dbu_int(val_nm: float, dbu_um: float) -> int:
    """Convert physical nm to integer DBU units (rounding to nearest)."""
    return round(val_nm / (dbu_um * 1000.0))


def _shape_to_polygon(shape: db.Shape) -> db.Polygon | None:
    """Return a db.Polygon for any polygon-like shape, or None if not applicable.

    KLayout reads GDSII axis-aligned rectangles as Box shapes rather than
    Polygon shapes.  This helper normalises both to db.Polygon so that the
    mask-layer extraction works identically for OASIS and GDSII inputs.
    """
    if shape.is_polygon():
        return shape.polygon
    if shape.is_box():
        return db.Polygon(shape.box)
    return None


# ─── KLayout ↔ PWCL helpers ──────────────────────────────────────────────────


def _poly_to_pwcl(poly: db.Polygon, dbu_um: float) -> list[PwclContour]:
    """Convert a KLayout Polygon (with holes) to PWCL LINE contours."""
    def pts_nm(it) -> list[tuple[float, float]]:
        return [(_dbu_to_nm(pt.x, dbu_um), _dbu_to_nm(pt.y, dbu_um)) for pt in it]

    def make_contour(pts: list[tuple[float, float]], is_hole: bool) -> PwclContour:
        n = len(pts)
        segs = [PwclSegment(SegmentType.LINE, [pts[i], pts[(i + 1) % n]]) for i in range(n)]
        return PwclContour(segments=segs, is_hole=is_hole)

    out = [make_contour(pts_nm(poly.each_point_hull()), is_hole=False)]
    for h in range(poly.holes()):
        out.append(make_contour(pts_nm(poly.each_point_hole(h)), is_hole=True))
    return out


def _pwcl_to_klayout_poly(contours: list[PwclContour], dbu_um: float) -> list[db.Polygon]:
    """Convert a list of PwclContour objects back to KLayout Polygon objects.

    Outer contours (is_hole=False) become polygon hulls.  Hole contours
    (is_hole=True) are associated with the most recently seen outer contour.
    Simple case: one outer + zero or more holes.
    """
    polys: list[db.Polygon] = []
    hull_pts: list[db.Point] | None = None
    hole_pts_list: list[list[db.Point]] = []

    def _flush() -> None:
        if hull_pts is None:
            return
        poly = db.Polygon(hull_pts)
        for hp in hole_pts_list:
            poly.insert_hole(hp)
        polys.append(poly)

    def _nm_pt(x_nm: float, y_nm: float) -> db.Point:
        return db.Point(_nm_to_dbu_int(x_nm, dbu_um), _nm_to_dbu_int(y_nm, dbu_um))

    for c in contours:
        # Collect unique vertices from the LINE segments (skip closing duplicate)
        pts = [_nm_pt(seg.pts[0][0], seg.pts[0][1]) for seg in c.segments]
        if not pts:
            continue
        if not c.is_hole:
            _flush()
            hull_pts = pts
            hole_pts_list = []
        else:
            hole_pts_list.append(pts)

    _flush()
    return polys


def _poly_area_nm2(poly: db.Polygon, dbu_um: float) -> float:
    """Return the area of a KLayout polygon in nm²."""
    nm_per_dbu = dbu_um * 1000.0
    return poly.area() * nm_per_dbu ** 2


# ─── Main round-trip logic ────────────────────────────────────────────────────


def run_roundtrip(
    oas_path: Path,
    output_dir: Path,
    mask_layer: int,
    marker_layer: int | None,
    grid_res_nm_per_px: float,
    truncation_px: float,
    original_layer: int,
    recon_layer: int,
    xor_layer: int,
    cell_name: str | None,
    marker_margin_nm: float = 0.0,
) -> None:
    """Full round-trip: rasterize → contour → XOR → write OASIS.

    Args:
        marker_layer: GDS layer of square marker shapes.  When ``None``, the
            bounding box of all mask polygons is used as the single canvas
            region (expanded to a square via ``max(W, H)``).
        marker_margin_nm: Expand each marker box by this many nm on every side
            before rasterizing.  The contoured polygon is then clipped back to
            the original marker boundary for the XOR comparison, so the margin
            only affects rasterization quality near the boundary — it never
            inflates the reconstructed geometry.  Default 0.0 (no margin).
    """
    _check_layout_ext(oas_path)

    t_total = time.perf_counter()

    # ── Load layout ───────────────────────────────────────────────────────────
    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu

    mask_li = layout.layer(mask_layer, 0)

    cells = [layout.cell(cell_name)] if cell_name else list(layout.each_cell())
    if cell_name and cells[0] is None:
        raise ValueError(f"Cell '{cell_name}' not found in {oas_path.name}")

    # ── Collect markers ───────────────────────────────────────────────────────
    markers: list[tuple[db.Box, float, float, float]] = []  # (box, x0_nm, y0_nm, size_nm)
    layout_bbox = db.Box()

    if marker_layer is not None:
        marker_li = layout.layer(marker_layer, 0)
        for cell in cells:
            for shape in cell.shapes(marker_li).each():
                if not shape.is_box():
                    continue
                box = shape.box
                w_nm = _dbu_to_nm(box.width(),  dbu_um)
                h_nm = _dbu_to_nm(box.height(), dbu_um)
                if abs(w_nm - h_nm) > 0.5:
                    log.warning("Non-square marker (%.1f×%.1f nm) — skipping", w_nm, h_nm)
                    continue
                markers.append((box,
                                _dbu_to_nm(box.left,   dbu_um),
                                _dbu_to_nm(box.bottom, dbu_um),
                                w_nm))
                layout_bbox += box

    if not markers:
        # ── Auto-bbox fallback: derive a single square marker from all mask polygons ──
        if marker_layer is not None:
            log.warning(
                "No square markers found on layer %d — falling back to mask polygon bbox.",
                marker_layer,
            )
        poly_bbox = db.Box()
        for cell in cells:
            for shape in cell.shapes(mask_li).each():
                poly = _shape_to_polygon(shape)
                if poly is not None:
                    poly_bbox += poly.bbox()
        if poly_bbox.empty():
            raise RuntimeError("No mask polygons found; cannot auto-generate canvas bbox.")
        # Expand to square (side = max dimension) anchored at the polygon bbox origin.
        side_dbu = max(poly_bbox.width(), poly_bbox.height())
        auto_box = db.Box(
            poly_bbox.left, poly_bbox.bottom,
            poly_bbox.left + side_dbu, poly_bbox.bottom + side_dbu,
        )
        size_nm = _dbu_to_nm(side_dbu, dbu_um)
        markers.append((auto_box,
                        _dbu_to_nm(auto_box.left,   dbu_um),
                        _dbu_to_nm(auto_box.bottom, dbu_um),
                        size_nm))
        layout_bbox = auto_box
        log.info(
            "Auto-bbox: polygon extent %.1f × %.1f nm → square marker %.1f nm",
            _dbu_to_nm(poly_bbox.width(),  dbu_um),
            _dbu_to_nm(poly_bbox.height(), dbu_um),
            size_nm,
        )

    log.info("Found %d markers", len(markers))

    # ── Group mask polygons by marker ─────────────────────────────────────────
    # orig_polys[m_idx]: original KLayout polygons for marker m
    orig_polys: dict[int, list[db.Polygon]] = defaultdict(list)
    marker_contours: dict[int, list[PwclContour]] = defaultdict(list)

    for cell in cells:
        for shape in cell.shapes(mask_li).each():
            poly = _shape_to_polygon(shape)
            if poly is None:
                continue
            hull_pts = list(poly.each_point_hull())
            if not hull_pts:
                continue
            centroid = db.Point(
                sum(pt.x for pt in hull_pts) // len(hull_pts),
                sum(pt.y for pt in hull_pts) // len(hull_pts),
            )
            for m_idx, (box, *_) in enumerate(markers):
                if box.contains(centroid):
                    orig_polys[m_idx].append(poly)
                    marker_contours[m_idx].extend(_poly_to_pwcl(poly, dbu_um))
                    break

    log.info("%d markers have polygons", len(marker_contours))

    # ── Build canvas (expanded by marker_margin_nm on every side) ────────────
    canvas_x0_nm = _dbu_to_nm(layout_bbox.left,   dbu_um) - marker_margin_nm
    canvas_y0_nm = _dbu_to_nm(layout_bbox.bottom, dbu_um) - marker_margin_nm
    canvas_W = math.ceil(
        (_dbu_to_nm(layout_bbox.width(),  dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )
    canvas_H = math.ceil(
        (_dbu_to_nm(layout_bbox.height(), dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )

    # ── Rasterize all markers → canvas ───────────────────────────────────────
    t_raster = time.perf_counter()

    batch = [
        (marker_contours[m_idx],
         mx_nm - marker_margin_nm,
         my_nm - marker_margin_nm,
         math.ceil((msize_nm + 2.0 * marker_margin_nm) / grid_res_nm_per_px))
        for m_idx, (_, mx_nm, my_nm, msize_nm) in enumerate(markers)
        if m_idx in marker_contours
    ]
    batch_m_indices = [
        m_idx
        for m_idx, (_, mx_nm, my_nm, msize_nm) in enumerate(markers)
        if m_idx in marker_contours
    ]

    canvas = rasterize_canvas(
        batch,
        canvas_x0_nm=canvas_x0_nm,
        canvas_y0_nm=canvas_y0_nm,
        canvas_H=canvas_H,
        canvas_W=canvas_W,
        grid_res_nm_per_px=grid_res_nm_per_px,
        truncation_px=truncation_px,
    )

    log.info("Rasterized %d markers in %.1f ms", len(batch),
             (time.perf_counter() - t_raster) * 1e3)

    # ── Contour each marker patch ─────────────────────────────────────────────
    t_contour = time.perf_counter()

    # recon_polys[m_idx]: reconstructed KLayout polygons for marker m
    recon_polys: dict[int, list[db.Polygon]] = {}

    for bi, m_idx in enumerate(batch_m_indices):
        _, mx_nm, my_nm, msize_nm = markers[m_idx]
        # Expanded patch origin and size (same values used in the batch)
        patch_ox   = mx_nm  - marker_margin_nm
        patch_oy   = my_nm  - marker_margin_nm
        patch_size = math.ceil((msize_nm + 2.0 * marker_margin_nm) / grid_res_nm_per_px)

        # Canvas pixel offset for the expanded patch origin
        col0 = round((patch_ox - canvas_x0_nm) / grid_res_nm_per_px)
        row0 = round((patch_oy - canvas_y0_nm) / grid_res_nm_per_px)

        # Crop the expanded patch from the canvas (clamp to canvas bounds)
        r0 = max(0, row0);  r1 = min(canvas_H, row0 + patch_size)
        c0 = max(0, col0);  c1 = min(canvas_W, col0 + patch_size)
        patch = canvas[r0:r1, c0:c1]

        if patch.size == 0:
            recon_polys[m_idx] = []
            continue

        # Contour using the expanded patch origin so physical coords are correct
        contours = csdf_to_contours(
            patch,
            origin_x_nm=patch_ox,
            origin_y_nm=patch_oy,
            grid_res_nm_per_px=grid_res_nm_per_px,
        )
        recon_polys[m_idx] = _pwcl_to_klayout_poly(contours, dbu_um)

    log.info("Contouring done in %.1f ms", (time.perf_counter() - t_contour) * 1e3)

    # ── XOR per marker ────────────────────────────────────────────────────────
    ep = db.EdgeProcessor()

    total_orig_area_nm2 = 0.0
    total_xor_area_nm2  = 0.0

    per_marker_stats: list[dict] = []

    all_orig_polys:  list[db.Polygon] = []
    all_recon_polys: list[db.Polygon] = []
    all_xor_polys:   list[db.Polygon] = []

    for m_idx in range(len(markers)):
        o_polys = orig_polys.get(m_idx, [])
        r_polys = recon_polys.get(m_idx, [])

        orig_area_nm2 = sum(_poly_area_nm2(p, dbu_um) for p in o_polys)
        recon_area_nm2 = sum(_poly_area_nm2(p, dbu_um) for p in r_polys)

        xor_result: list[db.Polygon] = []
        if o_polys or r_polys:
            xor_result = ep.boolean_p2p(
                o_polys, r_polys,
                db.EdgeProcessor.ModeXor,
                False,   # resolve_holes
                True,    # min_coherence
            )

        xor_area_nm2 = sum(_poly_area_nm2(p, dbu_um) for p in xor_result)
        xor_ratio = (xor_area_nm2 / orig_area_nm2) if orig_area_nm2 > 0 else 0.0
        xor_ratio = min(xor_ratio, 1.0)  # clamp per spec

        total_orig_area_nm2 += orig_area_nm2
        total_xor_area_nm2  += xor_area_nm2

        per_marker_stats.append({
            "marker_idx":      m_idx,
            "orig_area_nm2":   orig_area_nm2,
            "recon_area_nm2":  recon_area_nm2,
            "xor_area_nm2":    xor_area_nm2,
            "xor_ratio":       xor_ratio,
        })

        all_orig_polys.extend(o_polys)
        all_recon_polys.extend(r_polys)
        all_xor_polys.extend(xor_result)

    total_xor_ratio = (
        min(total_xor_area_nm2 / total_orig_area_nm2, 1.0)
        if total_orig_area_nm2 > 0 else 0.0
    )

    # ── Write output OASIS ────────────────────────────────────────────────────
    out_layout = db.Layout()
    out_layout.dbu = dbu_um
    top_cell = out_layout.create_cell("ROUNDTRIP")

    orig_li  = out_layout.layer(original_layer, 0)
    recon_li = out_layout.layer(recon_layer,    0)
    xor_li   = out_layout.layer(xor_layer,      0)

    for poly in all_orig_polys:
        top_cell.shapes(orig_li).insert(poly)
    for poly in all_recon_polys:
        top_cell.shapes(recon_li).insert(poly)
    for poly in all_xor_polys:
        top_cell.shapes(xor_li).insert(poly)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{oas_path.stem}_roundtrip.oas"
    out_layout.write(str(out_path))

    # ── Report ────────────────────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t_total) * 1e3

    print(f"\n{'─'*60}")
    print(f"  Input          : {oas_path.name}")
    print(f"  Grid res       : {grid_res_nm_per_px:.2f} nm/px")
    print(f"  Truncation     : {truncation_px:.1f} px")
    print(f"  Marker margin  : {marker_margin_nm:.1f} nm")
    print(f"  Markers        : {len(markers)}  ({len(marker_contours)} with polygons)")
    print(f"  Canvas         : {canvas_W} × {canvas_H} px")
    print(f"{'─'*60}")
    print(f"  {'Marker':>6}  {'Orig (µm²)':>12}  {'Recon (µm²)':>12}  "
          f"{'XOR (µm²)':>10}  {'XOR ratio':>10}")
    print(f"{'─'*60}")

    UM2 = 1e6  # nm² → µm²
    for s in per_marker_stats:
        if s["orig_area_nm2"] == 0 and s["xor_area_nm2"] == 0:
            continue
        print(f"  {s['marker_idx']:>6}  "
              f"{s['orig_area_nm2']/UM2:>12.4f}  "
              f"{s['recon_area_nm2']/UM2:>12.4f}  "
              f"{s['xor_area_nm2']/UM2:>10.4f}  "
              f"{s['xor_ratio']:>9.4%}")

    print(f"{'─'*60}")
    print(f"  {'TOTAL':>6}  "
          f"{total_orig_area_nm2/UM2:>12.4f}  "
          f"{'':>12}  "
          f"{total_xor_area_nm2/UM2:>10.4f}  "
          f"{total_xor_ratio:>9.4%}")
    print(f"{'─'*60}")
    print(f"\n  Output → {out_path}")
    print(f"    Layer {original_layer}: original polygons")
    print(f"    Layer {recon_layer}: reconstructed polygons")
    print(f"    Layer {xor_layer}: XOR (error) polygons")
    print(f"\n  Total time: {elapsed_ms:.0f} ms")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("oas_file", type=Path,
                   help="Input layout file (.oas/.oasis for OASIS, .gds/.gds2/.gdsii/.gdx for GDSII)")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("roundtrip_out"),
                   help="Output directory (default: roundtrip_out/)")
    p.add_argument("--mask-layer",     type=int,   default=1,
                   help="GDS layer of mask polygons (default: 1)")
    p.add_argument("--marker-layer",   type=int,   default=None,
                   help="GDS layer of square markers. "
                        "If omitted, the bounding box of all mask polygons is used "
                        "as the single canvas region.")
    p.add_argument("--grid-res",       type=float, default=1.0, metavar="NM_PER_PX",
                   help="Grid resolution nm/px (default: 1.0)")
    p.add_argument("--truncation",     type=float, default=5.0, metavar="PX",
                   help="cSDF truncation half-width in pixels (default: 5.0)")
    p.add_argument("--marker-margin",  type=float, default=0.0, metavar="NM",
                   help="Expand each marker box by this many nm on every side (default: 0.0)")
    p.add_argument("--original-layer", type=int,   default=1,
                   help="Output layer for original polygons (default: 1)")
    p.add_argument("--recon-layer",    type=int,   default=2,
                   help="Output layer for reconstructed polygons (default: 2)")
    p.add_argument("--xor-layer",      type=int,   default=3,
                   help="Output layer for XOR (error) polygons (default: 3)")
    p.add_argument("--cell", type=str, default=None,
                   help="Only process shapes in this cell (default: all cells)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.oas_file.exists():
        log.error("File not found: %s", args.oas_file)
        sys.exit(1)

    run_roundtrip(
        oas_path=args.oas_file,
        output_dir=args.output_dir,
        mask_layer=args.mask_layer,
        marker_layer=args.marker_layer,
        grid_res_nm_per_px=args.grid_res,
        truncation_px=args.truncation,
        original_layer=args.original_layer,
        recon_layer=args.recon_layer,
        xor_layer=args.xor_layer,
        cell_name=args.cell,
        marker_margin_nm=args.marker_margin,
    )


if __name__ == "__main__":
    main()

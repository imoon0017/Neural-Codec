"""Rasterize an IC layout (OASIS or GDSII) to a single 2D cSDF image.

Each square marker shape defines one rasterization region. Mask polygons are
converted to PWCL contours and rasterized via the cSDF pipeline. All marker
patches are composited into one float32 [H, W] image covering the full marker
bounding box. Pixels outside all markers are 0.0 (exterior).

When --marker-layer is omitted (the default), the bounding box of all mask
polygons is used as the single rasterization region.  The bounding box is
expanded to a square (side = max(W, H)) so the patch API constraints are met,
then further enlarged by --marker-margin on every side if specified.

Supported input formats:
  OASIS  — .oas, .oasis
  GDSII  — .gds, .gds2, .gdsii, .gdx

Outputs:
  <stem>.npy   — float32 [H, W] cSDF (1.0=inside, 0.5=boundary, 0.0=outside)
  <stem>.png   — heatmap with iso=0.5 boundary overlay (optional, --png)

Usage:
    python scripts/rasterize_oas.py layout.oas --output-dir out/
    python scripts/rasterize_oas.py layout.gds --output-dir out/
    python scripts/rasterize_oas.py layout.oas \\
        --mask-layer 1 --marker-layer 100 \\
        --grid-res 6 --truncation 5 \\
        --output-dir out/ --png
    python scripts/rasterize_oas.py layout.oas --cell TOP --output-dir out/
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


# ─── KLayout helpers ──────────────────────────────────────────────────────────


def _dbu_to_nm(val: float, dbu_um: float) -> float:
    return val * dbu_um * 1000.0


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


def _extract_contours(poly: db.Polygon, dbu_um: float) -> list[PwclContour]:
    """Convert a KLayout polygon (with holes) to PWCL LINE contours."""
    def pts_nm(it) -> list[tuple[float, float]]:
        return [(_dbu_to_nm(pt.x, dbu_um), _dbu_to_nm(pt.y, dbu_um)) for pt in it]

    def make_contour(pts: list[tuple[float, float]], is_hole: bool) -> PwclContour:
        n = len(pts)
        segs = [PwclSegment(SegmentType.LINE, [pts[i], pts[(i + 1) % n]]) for i in range(n)]
        return PwclContour(segments=segs, is_hole=is_hole)

    contours = [make_contour(pts_nm(poly.each_point_hull()), is_hole=False)]
    for h in range(poly.holes()):
        contours.append(make_contour(pts_nm(poly.each_point_hole(h)), is_hole=True))
    return contours


# ─── Layout → single 2D cSDF ─────────────────────────────────────────────────


def rasterize_layout(
    oas_path: Path,
    mask_layer: int,
    marker_layer: int | None,
    grid_res_nm_per_px: float,
    truncation_px: float,
    cell_name: str | None,
    marker_margin_nm: float = 0.0,
) -> np.ndarray:
    """Rasterize all marker regions into one float32 [H, W] cSDF image.

    Args:
        oas_path: Input layout file (.oas/.oasis for OASIS, .gds/.gds2/.gdsii/.gdx
            for GDSII).  KLayout auto-detects the format from the extension.
        marker_layer: GDS layer of square marker shapes.  When ``None``, the
            bounding box of all mask polygons is used as the single rasterization
            region (expanded to a square via ``max(W, H)``).
        marker_margin_nm: Expand each marker box by this many nm on every side
            before rasterizing.  Useful for stable cSDF values near marker
            boundaries — the extra context pixels prevent the cSDF from being
            artificially truncated at the patch edge.  The canvas is expanded
            by the same margin so no pixels are lost.  Default 0.0 (no margin).
    """
    _check_layout_ext(oas_path)

    t0 = time.perf_counter()

    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu

    mask_li = layout.layer(mask_layer, 0)

    cells = [layout.cell(cell_name)] if cell_name else list(layout.each_cell())

    # ── Collect markers ───────────────────────────────────────────────────────
    markers: list[tuple[db.Box, float, float, float]] = []
    layout_bbox = db.Box()

    if marker_layer is not None:
        marker_li = layout.layer(marker_layer, 0)
        for cell in cells:
            if cell is None:
                raise ValueError(f"Cell '{cell_name}' not found in {oas_path.name}")
            for shape in cell.shapes(marker_li).each():
                if not shape.is_box():
                    continue
                box = shape.box
                w_nm = _dbu_to_nm(box.width(),  dbu_um)
                h_nm = _dbu_to_nm(box.height(), dbu_um)
                if abs(w_nm - h_nm) > 0.5:
                    log.warning("Non-square marker (%.1f × %.1f nm) — skipping", w_nm, h_nm)
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
            if cell is None:
                raise ValueError(f"Cell '{cell_name}' not found in {oas_path.name}")
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

    # ── Pre-extract contours and group by marker ───────────────────────────────
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
                    marker_contours[m_idx].extend(_extract_contours(poly, dbu_um))
                    break

    log.info("Loaded %d markers, %d with polygons  (%.1f ms)",
             len(markers), len(marker_contours),
             (time.perf_counter() - t0) * 1e3)

    # ── Build canvas (expanded by marker_margin_nm on every side) ────────────
    canvas_x0_nm = _dbu_to_nm(layout_bbox.left,   dbu_um) - marker_margin_nm
    canvas_y0_nm = _dbu_to_nm(layout_bbox.bottom, dbu_um) - marker_margin_nm
    canvas_W = math.ceil(
        (_dbu_to_nm(layout_bbox.width(),  dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )
    canvas_H = math.ceil(
        (_dbu_to_nm(layout_bbox.height(), dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )

    if canvas_W * canvas_H > 200_000_000:
        raise RuntimeError(
            f"Canvas {canvas_W}×{canvas_H} = {canvas_W*canvas_H:,} px too large. "
            "Increase --grid-res."
        )

    log.info("Canvas: %d × %d px  (margin %.1f nm)", canvas_W, canvas_H, marker_margin_nm)

    # ── Batch rasterize — single C++ call for all markers ────────────────────
    t1 = time.perf_counter()

    batch = [
        (marker_contours[m_idx],
         mx_nm - marker_margin_nm,
         my_nm - marker_margin_nm,
         math.ceil((msize_nm + 2.0 * marker_margin_nm) / grid_res_nm_per_px))
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

    log.info("Rasterized %d markers in %.1f ms  (total %.1f ms)",
             len(batch),
             (time.perf_counter() - t1) * 1e3,
             (time.perf_counter() - t0) * 1e3)

    return canvas


# ─── I/O ─────────────────────────────────────────────────────────────────────


def save_npy(image: np.ndarray, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{stem}.npy"
    np.save(out, image)
    return out


def save_png(image: np.ndarray, output_dir: Path, stem: str) -> Path:
    import matplotlib.pyplot as plt

    aspect = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(12, max(4, 12 * aspect)))
    im = ax.imshow(image, vmin=0, vmax=1, cmap="RdBu", origin="lower")
    ax.contour(image, levels=[0.5], colors="black", linewidths=0.6)
    ax.set_title(f"{stem}  |  {image.shape[1]} × {image.shape[0]} px", fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="cSDF  (1=inside  0=outside)")
    fig.tight_layout()
    out = output_dir / f"{stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("oas_file", type=Path,
                   help="Input layout file (.oas/.oasis for OASIS, .gds/.gds2/.gdsii/.gdx for GDSII)")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("rasterized_out"),
                   help="Output directory (default: rasterized_out/)")
    p.add_argument("--mask-layer",   type=int, default=1,
                   help="GDS layer of mask polygons (default: 1)")
    p.add_argument("--marker-layer", type=int, default=None,
                   help="GDS layer of square marker shapes. "
                        "If omitted, the bounding box of all mask polygons is used "
                        "as the single rasterization region.")
    p.add_argument("--grid-res",   type=float, default=1.0, metavar="NM_PER_PX",
                   help="Grid resolution in nm/px (default: 1.0)")
    p.add_argument("--truncation", type=float, default=5.0, metavar="PX",
                   help="cSDF truncation half-width in pixels (default: 5.0)")
    p.add_argument("--marker-margin", type=float, default=0.0, metavar="NM",
                   help="Expand each marker box by this many nm on every side (default: 0.0)")
    p.add_argument("--cell", type=str, default=None,
                   help="Only process this cell (default: all cells)")
    p.add_argument("--png", action="store_true",
                   help="Also save a PNG visualization")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.oas_file.exists():
        log.error("File not found: %s", args.oas_file)
        sys.exit(1)

    image = rasterize_layout(
        oas_path=args.oas_file,
        mask_layer=args.mask_layer,
        marker_layer=args.marker_layer,
        grid_res_nm_per_px=args.grid_res,
        truncation_px=args.truncation,
        cell_name=args.cell,
        marker_margin_nm=args.marker_margin,
    )

    stem = args.oas_file.stem
    npy_out = save_npy(image, args.output_dir, stem)
    print(f"Saved → {npy_out}  shape={image.shape}  dtype={image.dtype}")

    if args.png:
        png_out = save_png(image, args.output_dir, stem)
        print(f"Saved → {png_out}")


if __name__ == "__main__":
    main()

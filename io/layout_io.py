"""OASIS layout I/O via klayout.db.

Provides helpers for:

* Reading mask polygons from an OASIS file, optionally filtered to a
  spatial region.
* Writing polygon geometry (in nanometres) to a new OASIS file.

Only OASIS is supported — passing a GDSII path raises ``ValueError``.

Coordinate convention
─────────────────────
All public functions accept and return coordinates in **nanometres** (nm).
Conversion to/from KLayout DBU (microns by default, ``dbu=0.001 µm``) is
handled internally.

Usage::

    from io.layout_io import read_polygons_in_region, write_oas

    polys = read_polygons_in_region(
        oas_path=Path("design.oas"),
        mask_layer=1,
        x0_nm=0.0, y0_nm=0.0, x1_nm=5000.0, y1_nm=5000.0,
    )

    write_oas(
        path=Path("reconstructed.oas"),
        hulls_nm=[[(x0, y0), (x1, y0), ...], ...],
        mask_layer=1,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_OASIS_SUFFIXES = frozenset({".oas", ".oasis"})
_GDSII_SUFFIXES = frozenset({".gds", ".gds2", ".gdsii"})


def _check_oasis(path: Path) -> None:
    if path.suffix.lower() in _GDSII_SUFFIXES:
        raise ValueError(
            f"GDSII is not supported — only OASIS (.oas) is accepted: {path}"
        )


def read_polygons_in_region(
    oas_path: Path,
    mask_layer: int,
    x0_nm: float,
    y0_nm: float,
    x1_nm: float,
    y1_nm: float,
) -> list[list[tuple[float, float]]]:
    """Return mask polygon hulls whose centroid falls inside a bounding box.

    Args:
        oas_path: Path to the OASIS layout file.
        mask_layer: GDS layer number of the mask polygons.
        x0_nm: Left edge of the region (nm).
        y0_nm: Bottom edge of the region (nm).
        x1_nm: Right edge of the region (nm).
        y1_nm: Top edge of the region (nm).

    Returns:
        List of hull point lists; each inner list is
        ``[(x0_nm, y0_nm), (x1_nm, y1_nm), ...]`` in nanometres.

    Raises:
        FileNotFoundError: If *oas_path* does not exist.
        ValueError: If *oas_path* is a GDSII file.
    """
    import klayout.db as db

    if not oas_path.exists():
        raise FileNotFoundError(f"OASIS file not found: {oas_path}")
    _check_oasis(oas_path)

    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu
    li = layout.layer(mask_layer, 0)

    # Region box in DBU units
    scale = 1.0 / (dbu_um * 1000.0)  # nm → DBU
    region_box = db.Box(
        round(x0_nm * scale),
        round(y0_nm * scale),
        round(x1_nm * scale),
        round(y1_nm * scale),
    )

    def _dbu_to_nm(v: float) -> float:
        return v * dbu_um * 1000.0

    hulls: list[list[tuple[float, float]]] = []
    for cell in layout.each_cell():
        for shape in cell.shapes(li).each():
            if shape.is_polygon():
                poly = shape.polygon
            elif shape.is_box():
                poly = db.Polygon(shape.box)
            else:
                continue

            hull_pts = list(poly.each_point_hull())
            if not hull_pts:
                continue
            cx = sum(pt.x for pt in hull_pts) // len(hull_pts)
            cy = sum(pt.y for pt in hull_pts) // len(hull_pts)
            if not region_box.contains(db.Point(cx, cy)):
                continue
            hulls.append(
                [(_dbu_to_nm(pt.x), _dbu_to_nm(pt.y)) for pt in hull_pts]
            )

    log.debug(
        "read_polygons_in_region: found %d polygon(s) in (%.1f,%.1f)–(%.1f,%.1f) nm",
        len(hulls), x0_nm, y0_nm, x1_nm, y1_nm,
    )
    return hulls


def write_oas(
    path: Path,
    hulls_nm: list[list[tuple[float, float]]],
    mask_layer: int,
    dbu: float = 0.001,
) -> None:
    """Write polygon hulls to an OASIS file.

    Each polygon is created as a simple closed hull (no holes).  Pass holes
    as separate hull entries if needed.

    Args:
        path: Destination path (created or overwritten).
        hulls_nm: List of hull point lists in nanometres.  Each inner list is
                  ``[(x0, y0), (x1, y1), ...]``.
        mask_layer: Target GDS layer number.
        dbu: Layout DBU in microns (default ``0.001`` µm = 1 nm grid).

    Raises:
        ValueError: If *path* has a GDSII extension.
    """
    import klayout.db as db

    _check_oasis(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    layout = db.Layout()
    layout.dbu = dbu
    cell = layout.create_cell("TOP")
    li = layout.layer(mask_layer, 0)

    scale = 1.0 / (dbu * 1000.0)  # nm → DBU

    for hull in hulls_nm:
        pts = [db.Point(round(x * scale), round(y * scale)) for x, y in hull]
        if len(pts) < 3:
            log.warning("Skipping degenerate polygon with %d point(s)", len(pts))
            continue
        poly = db.Polygon(pts)
        cell.shapes(li).insert(poly)

    opts = db.SaveLayoutOptions()
    opts.format = "OASIS"
    layout.write(str(path), opts)
    log.debug("write_oas: wrote %d polygon(s) to %s", len(hulls_nm), path)

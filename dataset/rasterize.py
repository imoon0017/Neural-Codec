"""Dataset rasterization script.

Iterates over unique layout files listed in ``catalog.csv`` and rasterizes
each one into a single float32 ``[H, W]`` canvas ``.npy`` file, identical to
the ``rasterize_oas.py`` pipeline.  A small YAML sidecar records the canvas
origin so the Dataset can crop the correct per-marker patch at load time.

Output per layout file::

    cache/<split>/<stem>.npy        — float32 [H, W] full canvas
    cache/<split>/<stem>_meta.yaml  — {canvas_x0_nm, canvas_y0_nm}

Also writes ``cache/manifest.yaml`` on completion.

Usage::

    python dataset/rasterize.py \\
        --config  train/config/baseline.yaml \\
        --splits  train validation test \\
        --workers 8

    # Force-rebuild even if files already exist:
    python dataset/rasterize.py --config ... --force
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import klayout.db as db

from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType, rasterize_canvas

log = logging.getLogger(__name__)

_DATASET_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _DATASET_DIR.parent


# ─── Path helpers ─────────────────────────────────────────────────────────────


def npy_path(row: dict[str, Any], cache_dir: Path) -> Path:
    """Full-canvas ``.npy`` path for the layout file that contains *row*."""
    return cache_dir / row["split"] / f"{Path(row['file']).stem}.npy"


def meta_path(row: dict[str, Any], cache_dir: Path) -> Path:
    """Canvas-origin sidecar YAML path for the layout file that contains *row*."""
    return cache_dir / row["split"] / f"{Path(row['file']).stem}_meta.yaml"


# ─── Shared geometry helpers ──────────────────────────────────────────────────


def _dbu_to_nm(val: float, dbu_um: float) -> float:
    return val * dbu_um * 1000.0


def _shape_to_polygon(shape: db.Shape) -> db.Polygon | None:
    if shape.is_polygon():
        return shape.polygon
    if shape.is_box():
        return db.Polygon(shape.box)
    return None


def _poly_to_pwcl(poly: db.Polygon, dbu_um: float) -> list[PwclContour]:
    def pts_nm(it: Any) -> list[tuple[float, float]]:
        return [(_dbu_to_nm(pt.x, dbu_um), _dbu_to_nm(pt.y, dbu_um)) for pt in it]

    def make_contour(pts: list[tuple[float, float]], is_hole: bool) -> PwclContour:
        n = len(pts)
        segs = [PwclSegment(SegmentType.LINE, [pts[i], pts[(i + 1) % n]]) for i in range(n)]
        return PwclContour(segments=segs, is_hole=is_hole)

    out = [make_contour(pts_nm(poly.each_point_hull()), False)]
    for h in range(poly.holes()):
        out.append(make_contour(pts_nm(poly.each_point_hole(h)), True))
    return out


# ─── Per-file rasterization (worker-safe) ────────────────────────────────────


def _rasterize_file(
    oas_abs: str,
    npy_out: str,
    meta_out: str,
    mask_layer: int,
    marker_layer: int,
    grid_res_nm_per_px: float,
    truncation_px: float,
    marker_margin_nm: float,
    force: bool,
) -> bool:
    """Rasterize one layout file → single canvas ``.npy`` + origin sidecar YAML.

    All markers in the layout are rasterized in a single ``rasterize_canvas``
    call and composited onto one float32 ``[H, W]`` array.  Each marker patch
    is expanded by *marker_margin_nm* on every side so that the cSDF is not
    artificially truncated at the marker boundary; the canvas origin is shifted
    outward by the same amount.  The Dataset crops the inner ``[S, S]`` patch
    (the original marker area) at load time using the stored canvas origin.

    Returns ``True`` if the file was written, ``False`` if it already existed
    and *force* is ``False``.
    """
    npy_p = Path(npy_out)
    meta_p = Path(meta_out)
    if npy_p.exists() and meta_p.exists() and not force:
        return False

    layout = db.Layout()
    layout.read(oas_abs)
    dbu_um: float = layout.dbu
    mask_li = layout.layer(mask_layer, 0)
    marker_li = layout.layer(marker_layer, 0)

    # ── Collect all square markers ────────────────────────────────────────────
    markers: list[tuple[db.Box, float, float, float]] = []  # (box, x0_nm, y0_nm, size_nm)
    layout_bbox = db.Box()

    for cell in layout.each_cell():
        for shape in cell.shapes(marker_li).each():
            if not shape.is_box():
                continue
            box = shape.box
            w_nm = _dbu_to_nm(box.width(), dbu_um)
            h_nm = _dbu_to_nm(box.height(), dbu_um)
            if abs(w_nm - h_nm) > 0.5:
                continue
            markers.append((box,
                            _dbu_to_nm(box.left,   dbu_um),
                            _dbu_to_nm(box.bottom, dbu_um),
                            w_nm))
            layout_bbox += box

    if not markers:
        return False

    # ── Group mask polygons by marker ─────────────────────────────────────────
    marker_contours: dict[int, list[PwclContour]] = defaultdict(list)

    for cell in layout.each_cell():
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
                    marker_contours[m_idx].extend(_poly_to_pwcl(poly, dbu_um))
                    break

    # ── Compute canvas dimensions (expanded by marker_margin_nm on every side) ─
    canvas_x0_nm = _dbu_to_nm(layout_bbox.left,   dbu_um) - marker_margin_nm
    canvas_y0_nm = _dbu_to_nm(layout_bbox.bottom, dbu_um) - marker_margin_nm
    canvas_W = math.ceil(
        (_dbu_to_nm(layout_bbox.width(),  dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )
    canvas_H = math.ceil(
        (_dbu_to_nm(layout_bbox.height(), dbu_um) + 2.0 * marker_margin_nm) / grid_res_nm_per_px
    )

    # ── Build batch (each patch expanded by margin) and rasterize ────────────
    batch = [
        (marker_contours[m_idx],
         mx_nm - marker_margin_nm,
         my_nm - marker_margin_nm,
         math.ceil((msize_nm + 2.0 * marker_margin_nm) / grid_res_nm_per_px))
        for m_idx, (_, mx_nm, my_nm, msize_nm) in enumerate(markers)
        if m_idx in marker_contours
    ]

    if not batch:
        return False

    canvas = rasterize_canvas(
        batch,
        canvas_x0_nm=canvas_x0_nm,
        canvas_y0_nm=canvas_y0_nm,
        canvas_H=canvas_H,
        canvas_W=canvas_W,
        grid_res_nm_per_px=grid_res_nm_per_px,
        truncation_px=truncation_px,
    )

    # ── Save canvas + origin sidecar ──────────────────────────────────────────
    npy_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_p, canvas)
    markers_meta = [
        {"x_nm": mx_nm, "y_nm": my_nm, "size_nm": msize_nm}
        for m_idx, (_, mx_nm, my_nm, msize_nm) in enumerate(markers)
        if m_idx in marker_contours
    ]
    with open(meta_p, "w") as f:
        yaml.safe_dump(
            {
                "canvas_x0_nm": canvas_x0_nm,
                "canvas_y0_nm": canvas_y0_nm,
                "markers": markers_meta,
            },
            f, default_flow_style=False,
        )

    return True


# ─── Public API ───────────────────────────────────────────────────────────────


def rasterize_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    workers: int = 1,
    force: bool = False,
) -> list[Path]:
    """Rasterize the layout files referenced by *rows* into canvas ``.npy`` files.

    One ``.npy`` is produced per unique layout file (not per polygon). Rows are
    grouped by ``file`` so each OASIS file is loaded at most once per worker.

    Args:
        rows: Catalog row dicts (as returned by ``load_catalog()``).
        config: Parsed YAML config dict.
        workers: Number of parallel worker processes.
        force: Overwrite existing ``.npy`` / ``_meta.yaml`` files.

    Returns:
        List of ``.npy`` :class:`~pathlib.Path` objects that were written.
    """
    if not rows:
        return []

    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    marker_margin_nm = float(config["csdf"].get("marker_margin_nm", 0.0))
    mask_layer = int(config["csdf"]["mask_layer"])
    marker_layer = int(config["csdf"]["marker_layer"])
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]

    # Deduplicate to one task per unique layout file
    seen: dict[str, tuple[str, str]] = {}  # file_rel → (npy_out, meta_out)
    for row in rows:
        if row["file"] not in seen:
            seen[row["file"]] = (
                str(npy_path(row, cache_dir)),
                str(meta_path(row, cache_dir)),
            )

    generated: list[Path] = []

    if workers <= 1:
        for file_rel, (npy_out, meta_out) in seen.items():
            oas_abs = str(_PROJECT_ROOT / file_rel)
            written = _rasterize_file(
                oas_abs, npy_out, meta_out,
                mask_layer, marker_layer, grid_res, truncation_px, marker_margin_nm, force,
            )
            if written:
                generated.append(Path(npy_out))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for file_rel, (npy_out, meta_out) in seen.items():
                oas_abs = str(_PROJECT_ROOT / file_rel)
                fut = exe.submit(
                    _rasterize_file,
                    oas_abs, npy_out, meta_out,
                    mask_layer, marker_layer, grid_res, truncation_px, marker_margin_nm, force,
                )
                futures[fut] = (file_rel, npy_out)
            for fut in as_completed(futures):
                file_rel, npy_out = futures[fut]
                try:
                    if fut.result():
                        generated.append(Path(npy_out))
                except Exception as exc:
                    log.error("Rasterization failed for %s: %s", file_rel, exc)
                    raise

    return generated


def rasterize_catalog(
    config: dict[str, Any],
    splits: list[str] | None = None,
    workers: int = 1,
    force: bool = False,
    catalog_path: Path | None = None,
) -> list[Path]:
    """Rasterize all (or a subset of) catalog entries.

    Args:
        config: Parsed YAML config dict.
        splits: Split names to process (default: all three splits).
        workers: Parallel worker processes.
        force: Overwrite existing files.
        catalog_path: Override the default ``dataset/catalog.csv`` path.

    Returns:
        List of ``.npy`` files written.
    """
    from dataset.ingest import load_catalog

    if catalog_path is None:
        catalog_path = _DATASET_DIR / "catalog.csv"

    all_rows = load_catalog(catalog_path)
    if splits:
        all_rows = [r for r in all_rows if r["split"] in splits]

    if not all_rows:
        log.info("No catalog rows to rasterize.")
        return []

    unique_files = len({r["file"] for r in all_rows})
    log.info(
        "Rasterizing %d layout file(s) (%d polygon(s)) with %d worker(s) …",
        unique_files, len(all_rows), max(workers, 1),
    )
    generated = rasterize_rows(all_rows, config=config, workers=workers, force=force)

    from dataset.ingest import write_manifest

    manifest_path = write_manifest(config, catalog_path=catalog_path)
    log.info("Manifest → %s", manifest_path.relative_to(_PROJECT_ROOT))

    return generated


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=Path("train/config/baseline.yaml"),
                   help="YAML config file (default: train/config/baseline.yaml)")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                   choices=["train", "validation", "test"],
                   help="Splits to rasterize (default: all)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes (default: 1)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .npy files")
    p.add_argument("--truncation-px", type=float, default=None,
                   help="Override csdf.truncation_px from config")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Override dataset.cache_dir from config")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = _load_config(args.config)

    if args.truncation_px is not None:
        config["csdf"]["truncation_px"] = args.truncation_px
    if args.cache_dir is not None:
        config["dataset"]["cache_dir"] = str(args.cache_dir)

    generated = rasterize_catalog(
        config=config,
        splits=args.splits,
        workers=args.workers,
        force=args.force,
    )
    print(f"Rasterized / updated {len(generated)} layout file(s).")


if __name__ == "__main__":
    main()

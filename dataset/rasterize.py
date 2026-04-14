"""Dataset rasterization script.

Iterates over entries in ``catalog.csv`` and converts each polygon + its
containing marker into a pre-rasterized ``float32 [S, S]`` ``.npy`` patch
saved under ``dataset/cache/<split>/``.

Output filename convention::

    cache/<split>/<oas_stem>_cell_<cell>_layer_<layer>_poly_<idx>.npy

Writes (or refreshes) ``cache/manifest.yaml`` on completion so that
``verify_dataset`` and the Dataset class can confirm cache freshness.

Usage::

    python dataset/rasterize.py \\
        --config  train/config/baseline.yaml \\
        --splits  train validation test \\
        --workers 8 \\
        --device  cpu

    # Force-rebuild even if .npy files already exist:
    python dataset/rasterize.py --config ... --force
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import klayout.db as db

from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType, rasterize_patch

log = logging.getLogger(__name__)

_DATASET_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _DATASET_DIR.parent


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _dbu_to_nm(val: float, dbu_um: float) -> float:
    return val * dbu_um * 1000.0


def _shape_to_polygon(shape: db.Shape) -> db.Polygon | None:
    if shape.is_polygon():
        return shape.polygon
    if shape.is_box():
        return db.Polygon(shape.box)
    return None


def _poly_to_pwcl(poly: db.Polygon, dbu_um: float) -> list[PwclContour]:
    """Convert a KLayout Polygon (with holes) to PWCL LINE contours."""
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


def npy_filename(file_rel: str, cell: str, layer: int | str, poly_idx: int | str) -> str:
    """Return the ``.npy`` filename for one catalog row.

    Sanitises the cell name so the filename is filesystem-safe.
    """
    stem = Path(file_rel).stem
    cell_safe = re.sub(r"[^A-Za-z0-9]", "_", cell)
    return f"{stem}_cell_{cell_safe}_layer_{layer}_poly_{poly_idx}.npy"


def npy_path(row: dict[str, Any], cache_dir: Path) -> Path:
    """Return the full ``.npy`` path for one catalog row."""
    return (
        cache_dir
        / row["split"]
        / npy_filename(row["file"], row["cell"], row["layer"], row["polygon_idx"])
    )


# ─── Per-file rasterization (runs in worker processes) ───────────────────────


def _rasterize_file_rows(
    oas_abs: str,
    rows: list[dict[str, Any]],
    mask_layer: int,
    marker_layer: int,
    grid_res_nm_per_px: float,
    truncation_px: float,
    cache_dir_str: str,
    force: bool,
) -> list[str]:
    """Load one OASIS file and rasterize all catalog rows that belong to it.

    Designed to run inside a worker process — returns a list of generated
    ``.npy`` path strings (only newly written files are included).
    """
    cache_dir = Path(cache_dir_str)
    out_paths: list[str] = []

    layout = db.Layout()
    layout.read(oas_abs)
    dbu_um: float = layout.dbu
    mask_li = layout.layer(mask_layer, 0)
    marker_li = layout.layer(marker_layer, 0)

    for row in rows:
        out = npy_path(row, cache_dir)
        if out.exists() and not force:
            continue

        cell = layout.cell(row["cell"])
        if cell is None:
            log.warning("Cell '%s' not found in %s — skipping poly %s",
                        row["cell"], oas_abs, row["polygon_idx"])
            continue

        # ── Find polygon at poly_idx ─────────────────────────────────────────
        target_idx = int(row["polygon_idx"])
        target_poly: db.Polygon | None = None
        idx = 0
        for shape in cell.shapes(mask_li).each():
            poly = _shape_to_polygon(shape)
            if poly is None:
                continue
            if idx == target_idx:
                target_poly = poly
                break
            idx += 1

        if target_poly is None:
            log.warning("Polygon %d not found in cell '%s' of %s",
                        target_idx, row["cell"], oas_abs)
            continue

        # ── Find containing marker ────────────────────────────────────────────
        hull_pts = list(target_poly.each_point_hull())
        centroid = db.Point(
            sum(pt.x for pt in hull_pts) // len(hull_pts),
            sum(pt.y for pt in hull_pts) // len(hull_pts),
        )
        marker_box: db.Box | None = None
        for shape in cell.shapes(marker_li).each():
            if shape.is_box() and shape.box.contains(centroid):
                marker_box = shape.box
                break

        if marker_box is None:
            log.warning("No marker found for polygon %d in cell '%s' of %s",
                        target_idx, row["cell"], oas_abs)
            continue

        # ── Rasterize ─────────────────────────────────────────────────────────
        contours = _poly_to_pwcl(target_poly, dbu_um)
        patch = rasterize_patch(
            contours=contours,
            origin_x_nm=_dbu_to_nm(marker_box.left, dbu_um),
            origin_y_nm=_dbu_to_nm(marker_box.bottom, dbu_um),
            patch_size_px=int(row["patch_size_px"]),
            grid_res_nm_per_px=grid_res_nm_per_px,
            truncation_px=truncation_px,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, patch)
        out_paths.append(str(out))

    return out_paths


# ─── Public API ───────────────────────────────────────────────────────────────


def rasterize_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    workers: int = 1,
    force: bool = False,
) -> list[Path]:
    """Rasterize a list of catalog rows into ``.npy`` patches.

    Groups rows by source file so each OASIS file is loaded only once per
    worker.  Returns the list of ``.npy`` files that were written (skips
    existing files unless *force* is set).

    Args:
        rows: Catalog row dicts (as returned by ``load_catalog()``).
        config: Parsed YAML config dict.
        workers: Number of parallel worker processes.
        force: Overwrite existing ``.npy`` files.

    Returns:
        List of :class:`~pathlib.Path` objects for every file written.
    """
    if not rows:
        return []

    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    mask_layer = int(config["csdf"]["mask_layer"])
    marker_layer = int(config["csdf"]["marker_layer"])
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]

    # Group rows by source file
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_file.setdefault(row["file"], []).append(row)

    generated: list[Path] = []

    if workers <= 1:
        for file_rel, file_rows in rows_by_file.items():
            oas_abs = str(_PROJECT_ROOT / file_rel)
            paths = _rasterize_file_rows(
                oas_abs, file_rows, mask_layer, marker_layer,
                grid_res, truncation_px, str(cache_dir), force,
            )
            generated.extend(Path(p) for p in paths)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for file_rel, file_rows in rows_by_file.items():
                oas_abs = str(_PROJECT_ROOT / file_rel)
                fut = exe.submit(
                    _rasterize_file_rows,
                    oas_abs, file_rows, mask_layer, marker_layer,
                    grid_res, truncation_px, str(cache_dir), force,
                )
                futures[fut] = file_rel
            for fut in as_completed(futures):
                try:
                    paths = fut.result()
                    generated.extend(Path(p) for p in paths)
                except Exception as exc:
                    log.error("Rasterization failed for %s: %s", futures[fut], exc)
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
        splits: List of split names to process (default: all three splits).
        workers: Parallel worker processes.
        force: Overwrite existing ``.npy`` files.
        catalog_path: Override the default ``dataset/catalog.csv`` path.

    Returns:
        List of ``.npy`` files written.
    """
    from dataset.ingest import load_catalog  # avoid circular at module level

    if catalog_path is None:
        catalog_path = _DATASET_DIR / "catalog.csv"

    all_rows = load_catalog(catalog_path)
    if splits:
        all_rows = [r for r in all_rows if r["split"] in splits]

    if not all_rows:
        log.info("No catalog rows to rasterize.")
        return []

    log.info("Rasterizing %d polygon(s) with %d worker(s) …", len(all_rows), max(workers, 1))
    generated = rasterize_rows(all_rows, config=config, workers=workers, force=force)

    # Refresh manifest
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
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = _load_config(args.config)

    generated = rasterize_catalog(
        config=config,
        splits=args.splits,
        workers=args.workers,
        force=args.force,
    )
    print(f"Rasterized / updated {len(generated)} .npy file(s).")


if __name__ == "__main__":
    main()

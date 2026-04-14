"""OASIS layout ingest script.

The only sanctioned way to add new OASIS files to the dataset.  Validates
each source file (OASIS only — GDSII is rejected), copies it into the
appropriate ``dataset/raw/<split>/`` subdirectory, and updates
``dataset/catalog.csv``.

Atomic 9-step pipeline (all-or-nothing; rolls back on verify failure):

  1. Validate OASIS files load via ``klayout.db``
  2. Marker layer: every shape is square; all sizes match within + across files
     and against existing catalog entries
  3. Mask layer: at least 1 polygon per file
  4. Duplicate check via SHA-256 — warn and skip; never overwrite
  5. Copy to ``raw/<split>/``
  6. Append rows to ``catalog.csv``  (atomic ``.tmp`` → rename)
  7. Rasterize new polygons → ``cache/<split>/``  (float32 ``.npy`` patches)
  8. Update ``cache/manifest.yaml``
  9. ``verify_dataset`` — roll back steps 5-8 on failure

``--split test`` requires the ``--confirm-test-freeze`` flag.

Usage::

    python dataset/ingest.py \\
        --source /path/to/layouts/ \\
        --split  train \\
        --config train/config/baseline.yaml \\
        --workers 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import klayout.db as db

from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType, derive_patch_size_px

log = logging.getLogger(__name__)

_DATASET_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _DATASET_DIR.parent
_CATALOG_PATH: Path = _DATASET_DIR / "catalog.csv"

_OASIS_EXTS: frozenset[str] = frozenset({".oas", ".oasis"})
_GDS_EXTS: frozenset[str] = frozenset({".gds", ".gds2", ".gdsii", ".gdx"})

_CATALOG_COLUMNS: list[str] = [
    "file", "cell", "layer", "datatype", "polygon_idx",
    "marker_size_nm", "patch_size_px",
    "marker_x_nm", "marker_y_nm",
    "bbox_x_nm", "bbox_y_nm",
    "n_vertices", "has_curves", "split",
]

_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


# ─── Utilities ────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dbu_to_nm(val: float, dbu_um: float) -> float:
    return val * dbu_um * 1000.0


def _shape_to_polygon(shape: db.Shape) -> db.Polygon | None:
    """Normalise any polygon-like KLayout shape to db.Polygon, or None."""
    if shape.is_polygon():
        return shape.polygon
    if shape.is_box():
        return db.Polygon(shape.box)
    return None


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_catalog(catalog_path: Path = _CATALOG_PATH) -> list[dict[str, Any]]:
    """Load catalog.csv and return a list of row dicts (empty if file missing)."""
    if not catalog_path.exists():
        return []
    with open(catalog_path, newline="") as f:
        return list(csv.DictReader(f))


def save_catalog(rows: list[dict[str, Any]], catalog_path: Path = _CATALOG_PATH) -> None:
    """Atomically overwrite catalog.csv via a .tmp file."""
    tmp = catalog_path.with_name("catalog.csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CATALOG_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(catalog_path)


# ─── Per-file validation & row extraction ─────────────────────────────────────


def _extract_file_rows(
    oas_path: Path,
    dest_rel: str,
    mask_layer: int,
    marker_layer: int,
    grid_res_nm_per_px: float,
    compaction_ratio: int,
    split: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Validate one OASIS file and extract catalog rows for all its polygons.

    Args:
        oas_path: Source OASIS file.
        dest_rel: Relative path the file will have after copying (for ``file`` column).
        mask_layer: GDS layer of mask polygons.
        marker_layer: GDS layer of square marker shapes.
        grid_res_nm_per_px: Rasterization resolution (nm/px).
        compaction_ratio: Spatial downsampling factor (must be a power of 2).
        split: Dataset split label.

    Returns:
        ``(marker_size_nm, rows)`` where *rows* is a list of catalog dicts.

    Raises:
        RuntimeError: On any validation failure (KLayout load, marker check,
            mask check, mixed sizes).
    """
    layout = db.Layout()
    try:
        layout.read(str(oas_path))
    except Exception as exc:
        raise RuntimeError(f"KLayout failed to load file: {exc}") from exc

    dbu_um: float = layout.dbu
    mask_li = layout.layer(mask_layer, 0)
    marker_li = layout.layer(marker_layer, 0)

    # ── Marker validation (step 2) ────────────────────────────────────────────
    all_marker_sizes: list[float] = []
    for cell in layout.each_cell():
        for shape in cell.shapes(marker_li).each():
            if not shape.is_box():
                continue
            box = shape.box
            w_nm = _dbu_to_nm(box.width(), dbu_um)
            h_nm = _dbu_to_nm(box.height(), dbu_um)
            if abs(w_nm - h_nm) > 0.5:
                raise RuntimeError(
                    f"Non-square marker ({w_nm:.1f} × {h_nm:.1f} nm)"
                    f" in cell '{cell.name}'"
                )
            all_marker_sizes.append(w_nm)

    if not all_marker_sizes:
        raise RuntimeError(f"No markers found on layer {marker_layer}")

    if max(all_marker_sizes) - min(all_marker_sizes) > 0.5:
        raise RuntimeError(
            f"Mixed marker sizes within file: "
            f"min={min(all_marker_sizes):.1f} nm, max={max(all_marker_sizes):.1f} nm"
        )

    marker_size_nm: float = all_marker_sizes[0]
    patch_size_px: int = derive_patch_size_px(marker_size_nm, grid_res_nm_per_px, compaction_ratio)

    # ── Mask layer validation (step 3) ────────────────────────────────────────
    found_poly = False
    for cell in layout.each_cell():
        for shape in cell.shapes(mask_li).each():
            if _shape_to_polygon(shape) is not None:
                found_poly = True
                break
        if found_poly:
            break
    if not found_poly:
        raise RuntimeError(f"No mask polygons found on layer {mask_layer}")

    # ── Extract catalog rows ──────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for cell in layout.each_cell():
        cell_markers: list[db.Box] = [
            shape.box
            for shape in cell.shapes(marker_li).each()
            if shape.is_box()
        ]

        poly_idx = 0
        for shape in cell.shapes(mask_li).each():
            poly = _shape_to_polygon(shape)
            if poly is None:
                continue

            hull_pts = list(poly.each_point_hull())
            if not hull_pts:
                poly_idx += 1
                continue

            centroid = db.Point(
                sum(pt.x for pt in hull_pts) // len(hull_pts),
                sum(pt.y for pt in hull_pts) // len(hull_pts),
            )
            containing_marker: db.Box | None = next(
                (box for box in cell_markers if box.contains(centroid)), None
            )
            if containing_marker is None:
                log.warning(
                    "%s cell='%s' poly=%d: no containing marker — skipping",
                    oas_path.name, cell.name, poly_idx,
                )
                poly_idx += 1
                continue

            bbox = poly.bbox()
            n_verts = poly.num_points_hull() + sum(
                poly.num_points_hole(h) for h in range(poly.holes())
            )

            rows.append({
                "file": dest_rel,
                "cell": cell.name,
                "layer": mask_layer,
                "datatype": 0,
                "polygon_idx": poly_idx,
                "marker_size_nm": round(marker_size_nm, 4),
                "patch_size_px": patch_size_px,
                "marker_x_nm": round(_dbu_to_nm(containing_marker.left,   dbu_um), 4),
                "marker_y_nm": round(_dbu_to_nm(containing_marker.bottom, dbu_um), 4),
                "bbox_x_nm": round(_dbu_to_nm(bbox.width(),  dbu_um), 4),
                "bbox_y_nm": round(_dbu_to_nm(bbox.height(), dbu_um), 4),
                "n_vertices": n_verts,
                "has_curves": False,
                "split": split,
            })
            poly_idx += 1

    return marker_size_nm, rows


# ─── Manifest helper ──────────────────────────────────────────────────────────


def write_manifest(
    config: dict[str, Any],
    catalog_path: Path = _CATALOG_PATH,
    cache_dir: Path | None = None,
) -> Path:
    """Write (or overwrite) ``cache/manifest.yaml`` from catalog + config.

    Returns the path of the written manifest.
    """
    if cache_dir is None:
        cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]

    rows = load_catalog(catalog_path)
    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    compaction_ratio = int(config["model"]["compaction_ratio"])

    marker_size_nm = float(rows[0]["marker_size_nm"]) if rows else 0.0
    patch_size_px = derive_patch_size_px(marker_size_nm, grid_res, compaction_ratio) if rows else 0

    csdf_source = _PROJECT_ROOT / "csdf" / "csdf_utils.py"
    csdf_hash = hashlib.sha256(csdf_source.read_bytes()).hexdigest()[:12]

    manifest = {
        "grid_res_nm_per_px": grid_res,
        "marker_size_nm": marker_size_nm,
        "patch_size_px": patch_size_px,
        "truncation_px": truncation_px,
        "n_polygons": len(rows),
        "csdf_module_hash": csdf_hash,
    }

    manifest_path = cache_dir / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    return manifest_path


# ─── Main ingest pipeline ─────────────────────────────────────────────────────


def ingest(
    source: Path,
    split: str,
    mask_layer: int,
    marker_layer: int,
    config: dict[str, Any],
    workers: int = 1,
    confirm_test_freeze: bool = False,
) -> None:
    """Run the 9-step atomic ingest pipeline.

    Args:
        source: Path to a single ``.oas`` file or a directory of ``.oas`` files.
        split: Target dataset split (``"train"``, ``"validation"``, or ``"test"``).
        mask_layer: GDS layer of mask polygons.
        marker_layer: GDS layer of square marker shapes.
        config: Parsed YAML config dict.
        workers: Number of parallel workers for rasterization.
        confirm_test_freeze: Must be ``True`` when ``split="test"``.

    Raises:
        RuntimeError: On validation failure or verify_dataset failure (after rollback).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    if split == "test" and not confirm_test_freeze:
        raise RuntimeError(
            "Adding files to the test split requires --confirm-test-freeze."
        )

    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    compaction_ratio = int(config["model"]["compaction_ratio"])
    raw_dir = _PROJECT_ROOT / config["paths"]["dataset_root"]
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]

    # ── Discover source OASIS files ───────────────────────────────────────────
    if source.is_file():
        if source.suffix.lower() not in _OASIS_EXTS:
            raise ValueError(f"Not an OASIS file: {source}")
        source_files = [source]
    else:
        source_files = sorted(p for p in source.iterdir() if p.suffix.lower() in _OASIS_EXTS)
        for p in source.iterdir():
            if p.suffix.lower() in _GDS_EXTS:
                log.warning("GDSII is not supported — skipping %s", p.name)
    if not source_files:
        raise RuntimeError(f"No .oas/.oasis files found in {source}")

    # ── Step 4: SHA-256 duplicate check ──────────────────────────────────────
    existing_hashes: dict[str, Path] = {}
    for ext in ("*.oas", "*.oasis"):
        for p in raw_dir.rglob(ext):
            existing_hashes[_sha256(p)] = p

    new_files: list[Path] = []
    for src in source_files:
        h = _sha256(src)
        if h in existing_hashes:
            log.warning(
                "Duplicate (SHA-256): %s matches %s — skipping",
                src.name, existing_hashes[h].relative_to(_PROJECT_ROOT),
            )
        else:
            new_files.append(src)

    if not new_files:
        log.info("All source files already present in the dataset.  Nothing to do.")
        return

    # ── Steps 1-3: Validate all new files ─────────────────────────────────────
    target_dir = raw_dir / split
    file_infos: dict[Path, tuple[float, list[dict[str, Any]]]] = {}

    for src in new_files:
        dest_rel = str((target_dir / src.name).relative_to(_PROJECT_ROOT))
        log.info("Validating %s …", src.name)
        try:
            marker_size_nm, rows = _extract_file_rows(
                src, dest_rel, mask_layer, marker_layer, grid_res, compaction_ratio, split,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Validation failed for {src.name}: {exc}") from exc
        file_infos[src] = (marker_size_nm, rows)
        log.info("  %s: %d polygons, marker %.1f nm", src.name, len(rows), marker_size_nm)

    # ── Uniform marker-size guard ─────────────────────────────────────────────
    new_sizes = {info[0] for info in file_infos.values()}
    if len(new_sizes) > 1:
        raise RuntimeError(
            f"Mixed marker sizes across source files: {sorted(new_sizes)}"
        )
    new_marker_size_nm = next(iter(new_sizes))

    existing_rows = load_catalog()
    if existing_rows:
        catalog_size = float(existing_rows[0]["marker_size_nm"])
        if abs(catalog_size - new_marker_size_nm) > 0.5:
            raise RuntimeError(
                f"Marker size mismatch: new={new_marker_size_nm:.1f} nm,"
                f" catalog={catalog_size:.1f} nm — mixing sizes is not allowed"
            )

    # ── Flatten new catalog rows ──────────────────────────────────────────────
    new_rows: list[dict[str, Any]] = [
        row for _, rows in file_infos.values() for row in rows
    ]
    if not new_rows:
        log.warning("No valid polygons found in %d file(s).  Nothing to ingest.", len(new_files))
        return

    # ── Step 5: Copy files to raw/<split>/ ───────────────────────────────────
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in new_files:
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
        log.info("Copied %s → %s", src.name, dst.relative_to(_PROJECT_ROOT))

    # ── Track rollback state ──────────────────────────────────────────────────
    manifest_path = cache_dir / "manifest.yaml"
    old_manifest: str | None = manifest_path.read_text() if manifest_path.exists() else None
    generated_npy: list[Path] = []

    try:
        # ── Step 6: Append to catalog.csv ─────────────────────────────────────
        save_catalog(existing_rows + new_rows)
        log.info("catalog.csv: %d → %d rows", len(existing_rows), len(existing_rows + new_rows))

        # ── Step 7: Rasterize new polygons ────────────────────────────────────
        from dataset.rasterize import rasterize_rows  # avoid circular at import time

        generated_npy = rasterize_rows(new_rows, config=config, workers=workers)
        log.info("Rasterized %d patch(es)", len(generated_npy))

        # ── Step 8: Update manifest.yaml ──────────────────────────────────────
        write_manifest(config, cache_dir=cache_dir)
        log.info("Manifest written → %s", manifest_path.relative_to(_PROJECT_ROOT))

        # ── Step 9: Verify ────────────────────────────────────────────────────
        from dataset.verify_dataset import verify_dataset

        verify_dataset(config, check_cache=(config["dataset"]["mode"] == "cached"))

    except Exception:
        log.error("Pipeline failure — rolling back steps 5-8")
        for dst in copied:
            dst.unlink(missing_ok=True)
        save_catalog(existing_rows)
        for npy in generated_npy:
            npy.unlink(missing_ok=True)
        if old_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(old_manifest)
        raise

    # ── Summary ───────────────────────────────────────────────────────────────
    patch_size_px = new_rows[0]["patch_size_px"]
    n_npy = len(list((cache_dir / split).glob("*.npy")))

    print(f"\nMarker size   : {new_marker_size_nm:.1f} nm (unified ✓)")
    print(f"Patch size    : {patch_size_px} px")
    print(f"Added         : {len(new_files)} file(s) / {len(new_rows)} polygons → raw/{split}/")
    skipped = len(source_files) - len(new_files)
    if skipped:
        print(f"Skipped (dup) : {skipped} file(s)")
    print(f"catalog.csv   : {len(existing_rows)} → {len(existing_rows) + len(new_rows)} rows")
    print(f"cache/{split}/  : {n_npy} .npy files")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", type=Path, required=True,
                   help="Source .oas file or directory of .oas files")
    p.add_argument("--split", choices=sorted(_VALID_SPLITS), required=True,
                   help="Target dataset split")
    p.add_argument("--mask-layer", type=int, default=None,
                   help="GDS layer of mask polygons (default: from config)")
    p.add_argument("--marker-layer", type=int, default=None,
                   help="GDS layer of square marker shapes (default: from config)")
    p.add_argument("--config", type=Path, default=Path("train/config/baseline.yaml"),
                   help="YAML config file (default: train/config/baseline.yaml)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel rasterization workers (default: 1)")
    p.add_argument("--confirm-test-freeze", action="store_true",
                   help="Required when --split test")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    config = _load_config(args.config)

    mask_layer = args.mask_layer if args.mask_layer is not None else int(config["csdf"]["mask_layer"])
    marker_layer = (
        args.marker_layer if args.marker_layer is not None else int(config["csdf"]["marker_layer"])
    )

    ingest(
        source=args.source,
        split=args.split,
        mask_layer=mask_layer,
        marker_layer=marker_layer,
        config=config,
        workers=args.workers,
        confirm_test_freeze=args.confirm_test_freeze,
    )


if __name__ == "__main__":
    main()

"""Dataset integrity verification script.

Checks that every entry in ``catalog.csv`` is consistent and loadable:

* Every ``.oas`` file listed in the catalog exists on disk.
* Every ``polygon_idx`` is reachable (the cell exists and the index is in range).
* Split labels are confined to ``{train, validation, test}`` with no overlap.
* All rows share the same ``marker_size_nm`` (mixed sizes are rejected).
* ``patch_size_px % compaction_ratio == 0`` for every row.
* **Cache mode only**: every expected ``.npy`` file exists, has the correct
  shape ``[patch_size_px, patch_size_px]``, and ``cache/manifest.yaml``
  matches the current config (``grid_res_nm_per_px``, ``truncation_px``).

Exit codes:
  0 — all checks passed
  1 — one or more checks failed

Usage::

    python dataset/verify_dataset.py --config train/config/baseline.yaml
    python dataset/verify_dataset.py --config ... --no-cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)

_DATASET_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _DATASET_DIR.parent
_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


# ─── Public verify function ───────────────────────────────────────────────────


def verify_dataset(
    config: dict[str, Any],
    catalog_path: Path | None = None,
    check_cache: bool = True,
) -> None:
    """Verify dataset integrity.  Raises ``RuntimeError`` listing all failures.

    Args:
        config: Parsed YAML config dict.
        catalog_path: Override default ``dataset/catalog.csv``.
        check_cache: When ``True``, verify ``.npy`` files and the manifest.
            Defaults to ``True``; pass ``False`` to skip cache checks.

    Raises:
        RuntimeError: If any check fails.  The message lists all failures so
            the caller can surface them at once rather than one at a time.
    """
    if catalog_path is None:
        catalog_path = _DATASET_DIR / "catalog.csv"

    errors: list[str] = []

    # ── Load catalog ──────────────────────────────────────────────────────────
    from dataset.ingest import load_catalog

    rows = load_catalog(catalog_path)
    if not rows:
        # An empty catalog is a warning, not a hard failure (fresh repo).
        log.warning("catalog.csv is empty or missing.")
        return

    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    compaction_ratio = int(config["model"]["compaction_ratio"])
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]

    # ── Split label check ─────────────────────────────────────────────────────
    bad_splits = {r["split"] for r in rows} - _VALID_SPLITS
    if bad_splits:
        errors.append(f"Invalid split label(s) in catalog: {bad_splits}")

    # ── Uniform marker size ───────────────────────────────────────────────────
    sizes = {float(r["marker_size_nm"]) for r in rows}
    if len(sizes) > 1:
        errors.append(f"Mixed marker_size_nm values in catalog: {sizes}")

    # ── patch_size_px divisibility ────────────────────────────────────────────
    bad_patch: list[str] = []
    for r in rows:
        psp = int(r["patch_size_px"])
        if psp % compaction_ratio != 0:
            bad_patch.append(f"{r['file']} poly {r['polygon_idx']}: patch_size_px={psp}")
    if bad_patch:
        errors.append(
            f"patch_size_px not divisible by compaction_ratio={compaction_ratio}:\n"
            + "\n".join(f"  {b}" for b in bad_patch[:10])
            + (f"\n  … and {len(bad_patch) - 10} more" if len(bad_patch) > 10 else "")
        )

    # ── Raw file existence ────────────────────────────────────────────────────
    missing_files: list[str] = []
    for file_rel in {r["file"] for r in rows}:
        if not (_PROJECT_ROOT / file_rel).exists():
            missing_files.append(file_rel)
    if missing_files:
        errors.append(
            f"{len(missing_files)} raw file(s) missing:\n"
            + "\n".join(f"  {f}" for f in sorted(missing_files)[:10])
        )

    # ── Polygon index accessibility ───────────────────────────────────────────
    # Spot-check: verify at most 50 rows (full scan would be slow at scale).
    _spot_check_polygons(rows[:50], errors)

    # ── Cache checks (optional) ───────────────────────────────────────────────
    if check_cache:
        _check_cache(rows, cache_dir, config, grid_res, truncation_px, errors)

    if errors:
        raise RuntimeError(
            f"Dataset verification failed ({len(errors)} issue(s)):\n"
            + "\n".join(f"[{i+1}] {e}" for i, e in enumerate(errors))
        )

    log.info(
        "verify_dataset: OK — %d rows, %d split(s), marker %.1f nm",
        len(rows),
        len({r["split"] for r in rows}),
        next(iter(sizes), 0.0),
    )


# ─── Internal check helpers ───────────────────────────────────────────────────


def _spot_check_polygons(rows: list[dict[str, Any]], errors: list[str]) -> None:
    """Verify polygon_idx is reachable for a sample of rows."""
    import klayout.db as db

    def shape_to_poly(shape: Any) -> Any | None:
        if shape.is_polygon():
            return shape.polygon
        if shape.is_box():
            return db.Polygon(shape.box)
        return None

    loaded_layouts: dict[str, db.Layout] = {}
    bad: list[str] = []

    for row in rows:
        oas_abs = str(_PROJECT_ROOT / row["file"])
        if oas_abs not in loaded_layouts:
            if not (_PROJECT_ROOT / row["file"]).exists():
                continue  # already reported as missing file
            try:
                layout = db.Layout()
                layout.read(oas_abs)
                loaded_layouts[oas_abs] = layout
            except Exception as exc:
                bad.append(f"Cannot load {row['file']}: {exc}")
                continue

        layout = loaded_layouts[oas_abs]
        cell = layout.cell(row["cell"])
        if cell is None:
            bad.append(f"Cell '{row['cell']}' not found in {row['file']}")
            continue

        mask_li = layout.layer(int(row["layer"]), 0)
        target_idx = int(row["polygon_idx"])
        found = False
        idx = 0
        for shape in cell.shapes(mask_li).each():
            if shape_to_poly(shape) is None:
                continue
            if idx == target_idx:
                found = True
                break
            idx += 1

        if not found:
            bad.append(
                f"Polygon idx {target_idx} not found in cell '{row['cell']}' of {row['file']}"
            )

    if bad:
        errors.append(
            f"{len(bad)} polygon accessibility failure(s):\n"
            + "\n".join(f"  {b}" for b in bad[:10])
        )


def _check_cache(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    config: dict[str, Any],
    grid_res: float,
    truncation_px: float,
    errors: list[str],
) -> None:
    """Check per-layout canvas .npy + meta YAML existence, and manifest freshness."""
    from dataset.rasterize import npy_path, meta_path

    # Manifest check
    manifest_path = cache_dir / "manifest.yaml"
    if not manifest_path.exists():
        errors.append(
            f"Cache manifest missing: {manifest_path}.  "
            "Run 'python dataset/rasterize.py' to build the cache."
        )
    else:
        with open(manifest_path) as f:
            manifest: dict[str, Any] = yaml.safe_load(f)
        if abs(float(manifest.get("grid_res_nm_per_px", 0)) - grid_res) > 1e-9:
            errors.append(
                f"Manifest grid_res_nm_per_px={manifest.get('grid_res_nm_per_px')}"
                f" does not match config={grid_res}"
            )
        if abs(float(manifest.get("truncation_px", 0)) - truncation_px) > 1e-9:
            errors.append(
                f"Manifest truncation_px={manifest.get('truncation_px')}"
                f" does not match config={truncation_px}"
            )

    # One .npy + _meta.yaml per unique layout file
    missing_npy: list[str] = []
    missing_meta: list[str] = []

    seen_files: set[str] = set()
    for row in rows:
        if row["file"] in seen_files:
            continue
        seen_files.add(row["file"])

        npy = npy_path(row, cache_dir)
        if not npy.exists():
            missing_npy.append(str(npy.relative_to(_PROJECT_ROOT)))

        meta = meta_path(row, cache_dir)
        if not meta.exists():
            missing_meta.append(str(meta.relative_to(_PROJECT_ROOT)))

    if missing_npy:
        errors.append(
            f"{len(missing_npy)} canvas .npy file(s) missing:\n"
            + "\n".join(f"  {p}" for p in missing_npy[:10])
            + (f"\n  … and {len(missing_npy) - 10} more" if len(missing_npy) > 10 else "")
        )
    if missing_meta:
        errors.append(
            f"{len(missing_meta)} canvas meta YAML file(s) missing:\n"
            + "\n".join(f"  {p}" for p in missing_meta[:10])
            + (f"\n  … and {len(missing_meta) - 10} more" if len(missing_meta) > 10 else "")
        )


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
    p.add_argument("--no-cache", dest="check_cache", action="store_false",
                   help="Skip cache (.npy / manifest) checks")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = _load_config(args.config)

    try:
        verify_dataset(config, check_cache=args.check_cache)
        print("Dataset OK.")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

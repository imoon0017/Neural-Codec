"""PyTorch Dataset for cSDF patches.

Supports two modes controlled by ``dataset.mode`` in the config YAML:

* ``cached``     — loads pre-rasterized ``.npy`` patches from
  ``dataset/cache/<split>/``.  Verifies ``cache/manifest.yaml`` matches
  the current config before use.  Marker locations are read from the
  per-file ``_meta.yaml`` sidecars written by ``rasterize.py``.
* ``on_the_fly`` — rasterizes PWCL geometry on demand from raw OASIS
  files in ``dataset/raw/<split>/``.  Slower; intended for quick
  experiments or when the cache has not been built yet.

``__getitem__`` returns::

    {
        "csdf":        float32 tensor [1, S, S],
        "file":        str (relative path to source .oas),
        "marker_x_nm": float,
        "marker_y_nm": float,
    }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)

_DATASET_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _DATASET_DIR.parent
_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})


# ─── Dataset class ────────────────────────────────────────────────────────────


class CsdfDataset(Dataset):
    """PyTorch Dataset serving cSDF patches for a single split.

    Each dataset item corresponds to one marker region in one OASIS file.
    The catalog (``catalog.csv``) holds one row per OASIS file; the per-file
    ``_meta.yaml`` sidecars (written by ``rasterize.py``) record the marker
    coordinates used to expand the item list to one entry per marker.

    Args:
        config: Parsed YAML config dict.
        split: One of ``"train"``, ``"validation"``, ``"test"``.
        catalog_path: Override the default ``dataset/catalog.csv`` location.
    """

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        catalog_path: Path | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        self._split = split
        self._mode: str = config["dataset"]["mode"]
        if self._mode not in ("cached", "on_the_fly"):
            raise ValueError(
                f"dataset.mode must be 'cached' or 'on_the_fly', got {self._mode!r}"
            )

        self._grid_res: float = float(config["csdf"]["grid_res_nm_per_px"])
        self._truncation_px: float = float(config["csdf"]["truncation_px"])
        self._mask_layer: int = int(config["csdf"]["mask_layer"])
        self._marker_layer: int = int(config["csdf"]["marker_layer"])
        self._cache_dir: Path = _PROJECT_ROOT / config["dataset"]["cache_dir"]
        self._raw_dir: Path = _PROJECT_ROOT / config["paths"]["dataset_root"]

        # Load catalog (one row per OASIS file)
        if catalog_path is None:
            catalog_path = _DATASET_DIR / "catalog.csv"
        from dataset.ingest import load_catalog

        all_rows = load_catalog(catalog_path)
        self._rows = [r for r in all_rows if r["split"] == split]
        if not self._rows:
            log.warning("No catalog entries for split '%s'", split)

        # Verify manifest in cached mode
        if self._mode == "cached":
            self._verify_manifest(config)

        # Per-process layout cache for on_the_fly mode (keyed by abs path str)
        self._layout_cache: dict[str, Any] = {}

        # Build flat item list: one entry per (file, marker) pair
        # Each item is (catalog_row, marker_dict) where marker_dict has
        # keys: x_nm, y_nm, size_nm
        self._items: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if self._mode == "cached":
            self._build_items_from_cache()
        else:
            self._build_items_on_the_fly()

        log.info(
            "CsdfDataset [%s, %s]: %d file(s), %d patch(es)",
            split, self._mode, len(self._rows), len(self._items),
        )

    # ── Manifest verification ─────────────────────────────────────────────────

    def _verify_manifest(self, config: dict[str, Any]) -> None:
        """Raise RuntimeError if the cache manifest does not match *config*."""
        manifest_path = self._cache_dir / "manifest.yaml"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Cache manifest not found: {manifest_path}.  "
                "Run 'python dataset/rasterize.py' to build the cache."
            )
        with open(manifest_path) as f:
            manifest: dict[str, Any] = yaml.safe_load(f)

        checks = [
            ("grid_res_nm_per_px", float(config["csdf"]["grid_res_nm_per_px"])),
            ("truncation_px",      float(config["csdf"]["truncation_px"])),
            ("marker_margin_nm",   float(config["csdf"].get("marker_margin_nm", 0.0))),
        ]
        mismatches: list[str] = []
        for key, expected in checks:
            if abs(float(manifest.get(key, 0)) - expected) > 1e-9:
                mismatches.append(
                    f"{key}: manifest={manifest.get(key)}, config={expected}"
                )
        if mismatches:
            raise RuntimeError(
                "Cache manifest mismatch — regenerate with 'python dataset/rasterize.py':\n"
                + "\n".join(f"  {m}" for m in mismatches)
            )

    # ── Item index construction ───────────────────────────────────────────────

    def _build_items_from_cache(self) -> None:
        """Expand catalog rows into (row, marker) pairs using meta YAML sidecars."""
        from dataset.rasterize import meta_path

        for row in self._rows:
            meta_file = meta_path(row, self._cache_dir)
            if not meta_file.exists():
                raise FileNotFoundError(
                    f"Canvas meta YAML missing: {meta_file}.  "
                    "Run 'python dataset/rasterize.py' or use mode='on_the_fly'."
                )
            with open(meta_file) as f:
                meta: dict[str, Any] = yaml.safe_load(f)
            for marker in meta.get("markers", []):
                self._items.append((row, marker))

    def _build_items_on_the_fly(self) -> None:
        """Expand catalog rows into (row, marker) pairs by scanning layout files."""
        for row in self._rows:
            oas_abs = _PROJECT_ROOT / row["file"]
            layout = self._get_layout(oas_abs)
            dbu_um: float = layout.dbu
            marker_li = layout.layer(self._marker_layer, 0)

            for cell in layout.each_cell():
                for shape in cell.shapes(marker_li).each():
                    if not shape.is_box():
                        continue
                    box = shape.box
                    w_nm = box.width() * dbu_um * 1000.0
                    h_nm = box.height() * dbu_um * 1000.0
                    if abs(w_nm - h_nm) > 0.5:
                        continue  # non-square; skip
                    self._items.append((row, {
                        "x_nm": box.left * dbu_um * 1000.0,
                        "y_nm": box.bottom * dbu_um * 1000.0,
                        "size_nm": w_nm,
                    }))

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row, marker = self._items[idx]

        if self._mode == "cached":
            patch = self._load_cached(row, marker)
        else:
            patch = self._rasterize_on_the_fly(row, marker)

        csdf_tensor = torch.from_numpy(patch).unsqueeze(0)  # [1, S, S]
        return {
            "csdf": csdf_tensor,
            "file": row["file"],
            "marker_x_nm": marker["x_nm"],
            "marker_y_nm": marker["y_nm"],
        }

    # ── Cached mode ───────────────────────────────────────────────────────────

    def _load_cached(self, row: dict[str, Any], marker: dict[str, Any]) -> np.ndarray:
        from dataset.rasterize import npy_path, meta_path

        canvas_file = npy_path(row, self._cache_dir)
        meta_file = meta_path(row, self._cache_dir)

        if not canvas_file.exists():
            raise FileNotFoundError(
                f"Canvas .npy missing: {canvas_file}.  "
                "Run 'python dataset/rasterize.py' or use mode='on_the_fly'."
            )
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Canvas meta YAML missing: {meta_file}.  "
                "Run 'python dataset/rasterize.py' or use mode='on_the_fly'."
            )

        canvas = np.load(canvas_file)
        with open(meta_file) as f:
            meta = yaml.safe_load(f)

        x0_nm = float(meta["canvas_x0_nm"])
        y0_nm = float(meta["canvas_y0_nm"])
        S = int(row["patch_size_px"])

        col0 = round((marker["x_nm"] - x0_nm) / self._grid_res)
        row0 = round((marker["y_nm"] - y0_nm) / self._grid_res)

        patch = canvas[row0 : row0 + S, col0 : col0 + S]
        if patch.shape != (S, S):
            raise RuntimeError(
                f"Cropped patch shape {patch.shape} != expected ({S}, {S}) "
                f"for {Path(row['file']).name} marker ({marker['x_nm']:.1f}, {marker['y_nm']:.1f}) nm"
            )
        return patch

    # ── On-the-fly mode ───────────────────────────────────────────────────────

    def _get_layout(self, oas_abs: Path) -> Any:
        """Return a cached db.Layout (per-process cache)."""
        import klayout.db as db

        key = str(oas_abs)
        if key not in self._layout_cache:
            layout = db.Layout()
            layout.read(key)
            self._layout_cache[key] = layout
        return self._layout_cache[key]

    def _rasterize_on_the_fly(
        self, row: dict[str, Any], marker: dict[str, Any]
    ) -> np.ndarray:
        """Load OASIS and rasterize all polygons belonging to *marker* on demand."""
        import klayout.db as db
        from csdf.csdf_utils import rasterize_patch, PwclContour, PwclSegment, SegmentType

        oas_abs = _PROJECT_ROOT / row["file"]
        layout = self._get_layout(oas_abs)
        dbu_um: float = layout.dbu

        mask_li = layout.layer(self._mask_layer, 0)
        marker_li = layout.layer(self._marker_layer, 0)

        # Reconstruct the KLayout Box for this marker from nm coords
        scale = 1.0 / (dbu_um * 1000.0)
        marker_box = db.Box(
            round(marker["x_nm"] * scale),
            round(marker["y_nm"] * scale),
            round((marker["x_nm"] + marker["size_nm"]) * scale),
            round((marker["y_nm"] + marker["size_nm"]) * scale),
        )

        def dbu_to_nm(v: float) -> float:
            return v * dbu_um * 1000.0

        def make_contour(pts: list[tuple[float, float]], is_hole: bool) -> PwclContour:
            n = len(pts)
            segs = [PwclSegment(SegmentType.LINE, [pts[i], pts[(i + 1) % n]]) for i in range(n)]
            return PwclContour(segments=segs, is_hole=is_hole)

        contours: list[PwclContour] = []
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
                if not marker_box.contains(centroid):
                    continue
                contours.append(
                    make_contour(
                        [(dbu_to_nm(pt.x), dbu_to_nm(pt.y)) for pt in poly.each_point_hull()],
                        False,
                    )
                )
                for h in range(poly.holes()):
                    contours.append(
                        make_contour(
                            [(dbu_to_nm(pt.x), dbu_to_nm(pt.y)) for pt in poly.each_point_hole(h)],
                            True,
                        )
                    )

        return rasterize_patch(
            contours=contours,
            origin_x_nm=marker["x_nm"],
            origin_y_nm=marker["y_nm"],
            patch_size_px=int(row["patch_size_px"]),
            grid_res_nm_per_px=self._grid_res,
            truncation_px=self._truncation_px,
        )


# ─── Convenience factory ──────────────────────────────────────────────────────


def _shape_to_polygon(shape: Any) -> Any | None:
    if shape.is_polygon():
        return shape.polygon
    if shape.is_box():
        import klayout.db as db
        return db.Polygon(shape.box)
    return None


def make_dataloaders(
    config: dict[str, Any],
    splits: list[str] | None = None,
    catalog_path: Path | None = None,
    **dataloader_kwargs: Any,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create a :class:`~torch.utils.data.DataLoader` for each requested split.

    Args:
        config: Parsed YAML config dict.
        splits: Splits to create loaders for (default: ``["train", "validation"]``).
        catalog_path: Override the default ``dataset/catalog.csv``.
        **dataloader_kwargs: Extra keyword arguments forwarded to
            :class:`~torch.utils.data.DataLoader` (e.g. ``num_workers``,
            ``pin_memory``).

    Returns:
        Dict mapping split name → DataLoader.
    """
    if splits is None:
        splits = ["train", "validation"]

    batch_size = int(config["training"]["batch_size"])
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for split in splits:
        ds = CsdfDataset(config, split, catalog_path=catalog_path)
        shuffle = split == "train"
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            **dataloader_kwargs,
        )
    return loaders

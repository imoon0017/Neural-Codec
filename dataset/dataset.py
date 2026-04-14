"""PyTorch Dataset for cSDF patches.

Supports two modes controlled by ``dataset.mode`` in the config YAML:

* ``cached``     — loads pre-rasterized ``.npy`` patches from
  ``dataset/cache/<split>/``.  Verifies ``cache/manifest.yaml`` matches
  the current config before use.
* ``on_the_fly`` — rasterizes PWCL geometry on demand from raw OASIS
  files in ``dataset/raw/<split>/``.  Slower; intended for quick
  experiments or when the cache has not been built yet.

``__getitem__`` returns::

    {
        "csdf":        float32 tensor [1, S, S],
        "file":        str (relative path to source .oas),
        "polygon_idx": int,
    }

A random-translation augmentation (shift up to ±8 px, zero-padded) is
applied **only** to the ``"train"`` split.
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
_AUGMENT_MAX_SHIFT_PX: int = 8


# ─── Translation augmentation ─────────────────────────────────────────────────


def _random_translate(patch: np.ndarray, max_shift: int) -> np.ndarray:
    """Randomly shift a 2-D cSDF patch and zero-pad the vacated border.

    Args:
        patch: float32 array of shape ``[H, W]``.
        max_shift: Maximum translation magnitude in pixels (each axis, each sign).

    Returns:
        Shifted array of the same shape; exterior pixels are filled with 0.0.
    """
    H, W = patch.shape
    dy = int(np.random.randint(-max_shift, max_shift + 1))
    dx = int(np.random.randint(-max_shift, max_shift + 1))

    result = np.zeros_like(patch)
    # Source window
    sy0 = max(0, -dy);  sy1 = min(H, H - dy)
    sx0 = max(0, -dx);  sx1 = min(W, W - dx)
    # Destination window
    dy0 = max(0, dy);   dy1 = min(H, H + dy)
    dx0 = max(0, dx);   dx1 = min(W, W + dx)
    result[dy0:dy1, dx0:dx1] = patch[sy0:sy1, sx0:sx1]
    return result


# ─── Dataset class ────────────────────────────────────────────────────────────


class CsdfDataset(Dataset):
    """PyTorch Dataset serving cSDF patches for a single split.

    Args:
        config: Parsed YAML config dict.
        split: One of ``"train"``, ``"validation"``, ``"test"``.
        catalog_path: Override the default ``dataset/catalog.csv`` location.
        augment: Apply random-translation augmentation.  Defaults to
            ``True`` when ``split="train"``, ``False`` otherwise.
    """

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        catalog_path: Path | None = None,
        augment: bool | None = None,
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
        self._augment: bool = (split == "train") if augment is None else augment

        # Load catalog
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

        log.info(
            "CsdfDataset [%s, %s]: %d samples, augment=%s",
            split, self._mode, len(self._rows), self._augment,
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

        grid_res = float(config["csdf"]["grid_res_nm_per_px"])
        truncation_px = float(config["csdf"]["truncation_px"])

        mismatches: list[str] = []
        if abs(float(manifest.get("grid_res_nm_per_px", 0)) - grid_res) > 1e-9:
            mismatches.append(
                f"grid_res_nm_per_px: manifest={manifest.get('grid_res_nm_per_px')},"
                f" config={grid_res}"
            )
        if abs(float(manifest.get("truncation_px", 0)) - truncation_px) > 1e-9:
            mismatches.append(
                f"truncation_px: manifest={manifest.get('truncation_px')},"
                f" config={truncation_px}"
            )
        if mismatches:
            raise RuntimeError(
                "Cache manifest mismatch — regenerate with 'python dataset/rasterize.py':\n"
                + "\n".join(f"  {m}" for m in mismatches)
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]

        if self._mode == "cached":
            patch = self._load_cached(row)
        else:
            patch = self._rasterize_on_the_fly(row)

        if self._augment:
            patch = _random_translate(patch, _AUGMENT_MAX_SHIFT_PX)

        csdf_tensor = torch.from_numpy(patch).unsqueeze(0)  # [1, S, S]
        return {
            "csdf": csdf_tensor,
            "file": row["file"],
            "polygon_idx": int(row["polygon_idx"]),
        }

    # ── Cached mode ───────────────────────────────────────────────────────────

    def _load_cached(self, row: dict[str, Any]) -> np.ndarray:
        from dataset.rasterize import npy_path

        path = npy_path(row, self._cache_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Cached patch missing: {path}.  "
                "Run 'python dataset/rasterize.py' or use mode='on_the_fly'."
            )
        return np.load(path)

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

    def _rasterize_on_the_fly(self, row: dict[str, Any]) -> np.ndarray:
        """Load OASIS and rasterize the polygon described by *row* on demand."""
        import klayout.db as db
        from csdf.csdf_utils import rasterize_patch, PwclContour, PwclSegment, SegmentType

        oas_abs = _PROJECT_ROOT / row["file"]
        layout = self._get_layout(oas_abs)
        dbu_um: float = layout.dbu

        mask_li = layout.layer(self._mask_layer, 0)
        marker_li = layout.layer(self._marker_layer, 0)

        cell = layout.cell(row["cell"])
        if cell is None:
            raise RuntimeError(f"Cell '{row['cell']}' not found in {row['file']}")

        # Find polygon at polygon_idx
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
            raise RuntimeError(
                f"Polygon {target_idx} not found in cell '{row['cell']}' of {row['file']}"
            )

        # Find containing marker
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
            raise RuntimeError(
                f"No marker for polygon {target_idx} in cell '{row['cell']}' of {row['file']}"
            )

        # Build PWCL contours
        def dbu_to_nm(v: float) -> float:
            return v * dbu_um * 1000.0

        def make_contour(pts: list[tuple[float, float]], is_hole: bool) -> PwclContour:
            n = len(pts)
            segs = [PwclSegment(SegmentType.LINE, [pts[i], pts[(i + 1) % n]]) for i in range(n)]
            return PwclContour(segments=segs, is_hole=is_hole)

        contours: list[PwclContour] = [
            make_contour(
                [(dbu_to_nm(pt.x), dbu_to_nm(pt.y)) for pt in target_poly.each_point_hull()],
                False,
            )
        ]
        for h in range(target_poly.holes()):
            contours.append(
                make_contour(
                    [(dbu_to_nm(pt.x), dbu_to_nm(pt.y)) for pt in target_poly.each_point_hole(h)],
                    True,
                )
            )

        return rasterize_patch(
            contours=contours,
            origin_x_nm=dbu_to_nm(marker_box.left),
            origin_y_nm=dbu_to_nm(marker_box.bottom),
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

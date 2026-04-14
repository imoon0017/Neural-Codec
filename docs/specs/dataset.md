# dataset/ — Layout Repository & Data Pipeline

## Overview

Single source of truth for all training, validation, and test layout data.

```
dataset/
├── raw/
│   ├── train/        # ≥ 80% — used in training loop
│   ├── validation/   # ≈ 10% — monitored during training (early stopping, hyperparameter tuning)
│   └── test/         # ≈ 10% — held out; never seen during training
├── cache/
│   ├── train/        # pre-rasterized .npy patches, float32 [S,S]
│   ├── validation/
│   └── test/
├── catalog.csv
├── ingest.py         # ONLY way to add OASIS files
├── rasterize.py      # PWCL → .npy cache (called by ingest.py)
├── dataset.py        # PyTorch Dataset
└── verify_dataset.py
```

Raw OASIS files and cache `.npy` files are both tracked via **DVC** — never commit binaries to git.

---

## catalog.csv Schema

| Column | Type | Description |
|---|---|---|
| `file` | str | Relative path to `.oas` file |
| `cell` | str | KLayout cell name |
| `layer` | int | GDS layer number |
| `datatype` | int | GDS datatype |
| `polygon_idx` | int | Index within cell |
| `marker_size_nm` | float | Side length of square marker shape (nm) |
| `patch_size_px` | int | Derived: `ceil(marker_size_nm / grid_res_nm_per_px)` rounded to next multiple of `c` |
| `bbox_x_nm` | float | Mask polygon bounding box width (nm) |
| `bbox_y_nm` | float | Mask polygon bounding box height (nm) |
| `n_vertices` | int | Total PWCL vertex count (all segments, including curve control points) |
| `has_curves` | bool | True if any segment is non-linear |
| `split` | str | `train` / `validation` / `test` |

---

## ingest.py

**The only sanctioned way to add OASIS files.** Never copy manually into `raw/`.

```bash
python dataset/ingest.py \
    --source        /path/to/layouts/ \
    --split         train \
    --mask-layer    1 \
    --marker-layer  10 \
    --config        train/config/baseline.yaml \
    --workers       16
```

Atomic 9-step pipeline (all-or-nothing, rolls back on failure):

1. Validate OASIS files load via `klayout.db`
2. **Marker layer checks**: exists, every shape is square, all marker sizes match across source files and existing `raw/` — reject mixed sizes
3. Mask layer check: at least one polygon per file
4. Duplicate check via SHA-256 — warn and skip, never overwrite
5. Copy to `raw/<split>/`
6. Append rows to `catalog.csv` (atomic write via `.tmp` → rename)
7. Call `rasterize.py` on new files only
8. Update `cache/manifest.yaml`
9. Run `verify_dataset.py` — roll back steps 5–8 on failure

`--split test` requires `--confirm-test-freeze` flag.

Summary printed on success:
```
Marker size   : 500.0 nm (unified ✓)
Patch size    : 512 px
Added         : 3 files / 1,204 polygons → raw/train/
Skipped (dup) : 1 file
catalog.csv   : 8,512 → 9,716 rows
cache/train/  : 8,102 → 9,306 .npy files
```

---

## rasterize.py

Rasterizes all polygons in `catalog.csv` → `cache/` `.npy` files. Called by `ingest.py`;
run standalone to rebuild the full cache.

```bash
python dataset/rasterize.py \
    --config  train/config/baseline.yaml \
    --splits  train validation test \
    --workers 16 \
    --device  cpu
```

- Output: `cache/<split>/<stem>_cell_<cell>_layer_<layer>_poly_<idx>.npy`, `float32 [S,S]`
- Skips existing files unless `--force` passed (idempotent)
- Writes `cache/manifest.yaml` on completion:

```yaml
grid_res_nm_per_px: 1.0
marker_size_nm: 500.0
patch_size_px: 512
truncation_px: 3.0
n_polygons: 12345
csdf_module_hash: "abc123..."
```

---

## dataset.py

PyTorch `Dataset`. Two modes:

| Mode | Config | Behaviour |
|---|---|---|
| `cached` | `dataset.mode: cached` | Loads `.npy` from `cache/` — default |
| `on_the_fly` | `dataset.mode: on_the_fly` | Rasterizes at load time — for quick experiments |

- `split` must be `"train"`, `"validation"`, or `"test"` — raises `ValueError` otherwise
- `__getitem__` returns `{"csdf": float32 [1,S,S], "file": str, "polygon_idx": int}`
- Random translation augmentation applied in both modes for `train` split only
- In cached mode, reads `manifest.yaml` on init — raises error if config mismatch

---

## verify_dataset.py

Run before every training or eval job:
- Every OASIS file in `catalog.csv` loads cleanly
- Every polygon index is accessible
- Split counts are consistent
- In cached mode: `cache/manifest.yaml` matches current config

---

## Gotchas

- **Mixed marker sizes** — rejected by `ingest.py`; all polygons must share one `marker_size_nm`
- **Non-square marker** — rejected explicitly; never use width or average silently
- **`patch_size_px` must not be set in config** — derived from `marker_size_nm` and `grid_res_nm_per_px`
- **Stale cache** — if `grid_res_nm_per_px` or `truncation_px` changes, regenerate with `rasterize.py`; `verify_dataset.py` catches mismatches via `manifest.yaml`

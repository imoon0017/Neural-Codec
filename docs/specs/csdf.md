# csdf/ — Custom Signed Distance Function Rasterizer

## Overview

Converts a PWCL mask polygon + marker layer shape (both from OASIS) into a square `float32`
cSDF grid. Implemented in C++/CUDA with Python bindings via pybind11.

- CPU path: OpenMP-parallelised C++ (`rasterizer.cpp`) — production target, not a fallback
- GPU path: CUDA kernel (`rasterizer.cu`)
- Python dispatch: `csdf_utils.py` selects path from `device` argument or auto-detects CUDA

## Three-Step Rasterization Process

Must be implemented in this order:

1. **SDF computation** — Euclidean signed distance from each pixel center to the nearest PWCL
   edge. Convention: **negative inside**, positive outside.
2. **Truncation** — clamp to band `[-t, +t]` where `t = csdf.truncation_px`. Values outside
   the band are set to the limit.
3. **Normalization** — `csdf = (clamp(sdf, -t, +t) + t) / (2 * t)` → range `[0, 1]`

## cSDF Value Convention (hard rule — never deviate)

| Value | Meaning |
|---|---|
| `1.0` | Pixel fully inside (SDF ≤ −t) |
| `0.0` | Pixel fully outside (SDF ≥ +t) |
| `(0.0, 1.0)` | Within truncation band — normalized signed distance, **not** coverage fraction |
| `0.5` | Pixel center exactly on polygon boundary (SDF = 0) |

Iso-level **0.5** is the correct marching squares threshold — corresponds to SDF = 0 after
normalization. Using any other iso-level is a silent geometry error.

## Patch Size Determination (Marker Layer)

The rasterization region is defined by a **square marker shape** on `csdf.marker_layer` in
the OASIS file, co-located with the mask polygon.

```
S = ceil(marker_size_nm / grid_res_nm_per_px)
S = S + (c - S % c) % c          # round up to next multiple of compaction ratio c
```

- Marker size does **not** need to be an integer multiple of `grid_res_nm_per_px`
- `S` is one fixed value across the entire dataset — `ingest.py` enforces this
- `patch_size_px` in config is **derived**, never set manually

## Lossless Round-Trip Guarantee

At sufficient grid resolution, the pipeline `PWCL → cSDF → marching squares (0.5) → PWCL`
recovers the original geometry within sub-pixel tolerance. The rasterizer and `contouring/`
module must be co-designed to preserve this.

Resolution requirement: no polygon feature (edge, corner, curve) should span fewer than ~2
pixels within the truncation band. Validate `grid_res_nm_per_px` against dataset minimum
feature size before training.

## Gotchas

- **Iso-level must be 0.5** — any other value is a silent geometry error
- **Truncation band `t` must match encode and decode time** — store in `.cdna` `meta.json`
- **KLayout DBU is in microns** — multiply `layout.dbu × 1000` to get nm/DBU
- **OASIS only** — raise `ValueError` on GDSII input; never silently fall through
- **Winding order** — CCW for outer polygons, CW for holes; check after every contouring step

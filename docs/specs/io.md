# io/ — File I/O and .cdna Archive

## Overview

Handles serialization of integer curve DNA to the `.cdna` archive format and OASIS layout
read/write via `klayout.db`. OASIS is the **only** supported layout format — raise `ValueError`
on any GDSII input.

## `.cdna` Archive Format

A zip container with two entries:

### `dna.bin`

Raw integer array, C-contiguous:
- dtype: integer, little-endian — `int8` if `quantizer_bits=8`, `int16` if `quantizer_bits=16`
- shape: `[N_polygons, D, S/c, S/c]`

### `meta.json`

```json
{
  "latent_dim": 32,
  "compaction_ratio": 8,
  "patch_size_px": 512,
  "quantizer_bits": 16,
  "grid_res_nm_per_px": 1.0,
  "truncation_px": 3.0,
  "scale_factors": [0.12, 0.08, "..."],
  "model_version": "v1.0.0",
  "checkpoint_hash": "abc123..."
}
```

- `S/c` is not stored — always recompute as `patch_size_px // compaction_ratio`
- `scale_factors` has `D` entries (one per latent channel) — required for dequantization
- `checkpoint_hash`: SHA-256 of the model checkpoint for reproducible decoding

## pack.py / unpack.py

**`pack.py`**: called via `model.export_dna(x)` — takes integer tensor (dtype from `quantizer_bits`) + config, writes `.cdna`

**`unpack.py`**: reads `quantizer_bits` from `meta.json` to determine dtype, loads `dna.bin` as the correct integer type, reads `scale_factors`, dequantizes to `float32` via `z_hat = q * s_d`, passes to decoder.

## layout_io.py

- Read: `klayout.db.Layout` + `klayout.db.RecursiveShapeIterator`
- Convert DBU → nm: `shape.dbu_to_micron() * 1000` (KLayout DBU is in microns)
- Write reconstructed OASIS: create new `klayout.db.Layout`, insert polygons, call `layout.write("output.oas")`

## Gotchas

- **Scale factors must come from `.cdna`** — a mismatch produces silently wrong geometry
- **`S/c` is derived** — never store or read it from the archive
- **Bump archive version** in `meta.json` if the serialization format changes

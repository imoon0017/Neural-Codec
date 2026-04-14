# eval/ — Evaluation Module

## Overview

Official benchmark runner. Enforces a strict sequential execution order — never skip steps
or compute metrics from in-memory shortcuts.

## Execution Order (mandatory)

```
Step 1: Encode   — OASIS → .cdna (save to disk)
Step 2: Decode   — .cdna → reconstructed OASIS (save to disk)
Step 3: Metrics  — compare original vs reconstructed, report
```

Metrics must be computed from files saved to disk, not from in-memory tensors.

## evaluate.py

```bash
python eval/evaluate.py \
    --checkpoint checkpoints/<run_id>/best.pt \
    --config     train/config/baseline.yaml \
    --data-dir   dataset/raw/test/ \
    --output-dir eval/results/<run_id>/ \
    --device     cpu    # or "cuda"
```

1. **Encode pass**: load PWCL → rasterize → encode → quantize → save `.cdna` to `--output-dir/dna/`
2. **Decode pass**: load `.cdna` → dequantize → decode → contour → save reconstructed `.oas` to `--output-dir/reconstructed/`
3. **Metrics pass**: call `metrics.compression_ratio()` and `metrics.area_difference_ratio()` per polygon
4. **Report**: write `--output-dir/results.csv` and `--output-dir/summary.json`

## metrics.py

```python
def compression_ratio(original_oas_path: Path, cdna_path: Path) -> float:
    """Returns filesize(original) / filesize(.cdna). Both must exist on disk."""

def area_difference_ratio(original_oas_path: Path, reconstructed_oas_path: Path) -> float:
    """Returns Area(original XOR reconstructed) / Area(original), range [0, 1]."""
```

- XOR via `klayout.db.EdgeProcessor.boolean(ModeXor)`
- Areas in nm² after DBU conversion — never in pixel units
- If XOR area > original area: clamp to 1.0 and log a warning — do not discard the sample

## Acceptance Targets

| Metric | Target |
|---|---|
| Compression ratio | ≥ 10× |
| Area difference ratio | ≤ 0.02 (≤ 2%) |
| Encoder inference, GPU | ≤ 50 ms |
| Encoder inference, CPU | ≤ 500 ms |
| Decoder + contouring, GPU | ≤ 100 ms |
| Decoder + contouring, CPU | ≤ 1000 ms |
| Rasterizer throughput, GPU/CPU | TBD after profiling on target process node |

Hausdorff distance is tracked during training as a proxy but is **not** an official metric.

## report.py

Produces `summary.json`:
```json
{
  "n_polygons": 1234,
  "compression_ratio_mean": 12.4,
  "compression_ratio_min": 8.1,
  "area_difference_ratio_mean": 0.011,
  "area_difference_ratio_p95": 0.031,
  "checkpoint": "checkpoints/run_01/best.pt",
  "model_version": "v1.0.0"
}
```

## results.csv Schema

`file, polygon_idx, compression_ratio, area_difference_ratio`

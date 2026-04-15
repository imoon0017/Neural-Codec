"""Write per-sample results and aggregate summary for a CurveCodec evaluation run.

Two output artefacts are produced:

* ``results.csv`` — one row per (file, marker) with per-sample metrics.
* ``summary.json`` — aggregate statistics matching the acceptance-target
  table in ``docs/specs/eval.md``.

Usage::

    from eval.report import write_results_csv, write_summary_json

    write_results_csv(results, output_dir / "results.csv")
    write_summary_json(results, output_dir / "summary.json",
                       checkpoint_path=ckpt_path, n_files=len(test_files))
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Column order for results.csv
_CSV_COLUMNS = [
    "file",
    "marker_idx",
    "marker_x_nm",
    "marker_y_nm",
    "compression_ratio",
    "area_difference_ratio",
    "encode_ms",
    "decode_ms",
]


def write_results_csv(
    results: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write per-sample evaluation results to a CSV file.

    Args:
        results: List of result dicts.  Each dict must contain the keys
                 defined in ``_CSV_COLUMNS`` (extra keys are ignored).
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    log.info("Results CSV → %s  (%d row(s))", path, len(results))


def write_summary_json(
    results: list[dict[str, Any]],
    path: Path,
    checkpoint_path: Path | None = None,
    n_files: int = 0,
) -> None:
    """Write aggregate evaluation summary to a JSON file.

    Computes mean, min, p95 statistics over all samples.  Latency stats are
    included only when ``encode_ms`` / ``decode_ms`` are present in the
    results.

    Args:
        results: List of result dicts (same as :func:`write_results_csv`).
        path: Destination JSON path.
        checkpoint_path: Path to the evaluated checkpoint (stored as a
                         string for reference).
        n_files: Number of OASIS test files processed.
    """
    if not results:
        log.warning("No results to summarise — writing empty summary.json")
        summary: dict[str, Any] = {"n_samples": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return

    crs = [r["compression_ratio"] for r in results if r.get("compression_ratio") is not None]
    adrs = [r["area_difference_ratio"] for r in results if r.get("area_difference_ratio") is not None]
    enc_ms = [r["encode_ms"] for r in results if r.get("encode_ms") is not None]
    dec_ms = [r["decode_ms"] for r in results if r.get("decode_ms") is not None]

    def _stats(values: list[float], label: str) -> dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=np.float64)
        return {
            f"{label}_mean": float(arr.mean()),
            f"{label}_min":  float(arr.min()),
            f"{label}_max":  float(arr.max()),
            f"{label}_p95":  float(np.percentile(arr, 95)),
        }

    summary = {
        "n_samples": len(results),
        "n_files": n_files,
        **_stats(crs,    "compression_ratio"),
        **_stats(adrs,   "area_difference_ratio"),
        **_stats(enc_ms, "encode_ms"),
        **_stats(dec_ms, "decode_ms"),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "model_version": "v1.0.0",
    }

    # Acceptance targets (informational)
    targets: dict[str, Any] = {
        "compression_ratio_target": ">=10",
        "area_difference_ratio_target": "<=0.02",
        "encode_ms_gpu_target": "<=50",
        "decode_ms_gpu_target": "<=100",
    }
    passed: dict[str, bool] = {}
    if crs:
        passed["compression_ratio_ge10"] = float(np.mean(crs)) >= 10.0
    if adrs:
        passed["area_difference_ratio_le002"] = float(np.mean(adrs)) <= 0.02

    summary["targets"] = targets
    summary["targets_passed"] = passed

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Summary JSON → %s  (n=%d, CR_mean=%.2f, ADR_mean=%.4f)",
        path,
        len(results),
        float(np.mean(crs)) if crs else float("nan"),
        float(np.mean(adrs)) if adrs else float("nan"),
    )

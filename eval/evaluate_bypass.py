"""Bypass round-trip evaluation: rasterize → marching squares, no codec.

Measures the theoretical minimum ADR introduced by the discretisation step
alone (rasterize to cSDF → marching squares at iso=0.5).  No checkpoint or
model is required.  The raw cSDF is passed directly to the contouring
pipeline; the resulting ADR is the reference floor that any trained codec
can only approach, never beat.

Pipeline per marker
-------------------
1. Read raw OASIS; locate square markers.
2. Expand each marker by ``marker_margin_nm`` (from manifest) on all sides.
3. Rasterize to cSDF  ``[N, S_exp, S_exp]``.
4. Crop back to the original marker footprint  (erode by margin pixels).
5. Marching squares → contours.
6. ADR vs. original polygons.

No ``.cdna`` files are written; ``compression_ratio`` is reported as N/A.
``decode_ms`` in the output CSV records the marching-squares wall-clock time.

Usage::

    python eval/evaluate_bypass.py \\
        --config train/config/baseline.yaml \\
        --output-dir eval/results/bypass_reference/ \\
        --split test
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "io"))

# Reuse shared helpers from evaluate.py — avoids duplicating 200+ lines of
# OASIS I/O and canvas-cropping logic.
from eval.evaluate import (  # noqa: E402
    _INSP_MARKER_LAYER,
    _INSP_ORIG_LAYER,
    _INSP_RECON_LAYER,
    _INSP_XOR_LAYER,
    _load_manifest,
    _load_split_rows,
    _postprocess_per_marker,
    _rasterize_file_expanded,
    _write_inspection_oas,
)
from eval.metrics import area_difference_ratio  # noqa: E402
from eval.report import write_results_csv  # noqa: E402

log = logging.getLogger(__name__)

_DATASET_DIR = _PROJECT_ROOT / "dataset"


# ─── Summary ─────────────────────────────────────────────────────────────────


def _write_bypass_summary(
    results: list[dict[str, Any]],
    path: Path,
    n_files: int,
) -> None:
    """Write aggregate summary JSON for the bypass reference run.

    ``decode_ms`` fields represent the marching-squares wall-clock time;
    there is no encode step and no compression ratio.
    """
    if not results:
        log.warning("No results to summarise — writing empty summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"n_samples": 0, "run_type": "bypass_reference"}, f, indent=2)
        return

    adrs = [r["area_difference_ratio"] for r in results if r.get("area_difference_ratio") is not None]
    contour_ms_vals = [r["decode_ms"] for r in results if r.get("decode_ms") is not None]

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

    summary: dict[str, Any] = {
        "run_type": "bypass_reference",
        "description": (
            "Rasterize→marching-squares only; no codec. "
            "ADR here is the theoretical minimum floor for any trained model."
        ),
        "n_samples": len(results),
        "n_files": n_files,
        **_stats(adrs, "area_difference_ratio"),
        **_stats(contour_ms_vals, "contour_ms"),
        "notes": {
            "compression_ratio": "N/A (no .cdna produced)",
            "encode_ms": "N/A (no encoder)",
            "decode_ms_column": "marching-squares wall-clock time per marker",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Bypass summary → %s  (n=%d  ADR_mean=%.4f  contour_ms_mean=%.2f)",
        path,
        len(results),
        float(np.mean(adrs)) if adrs else float("nan"),
        float(np.mean(contour_ms_vals)) if contour_ms_vals else float("nan"),
    )


# ─── Per-file bypass pass ─────────────────────────────────────────────────────


def _bypass_file(
    row: dict[str, Any],
    orig_markers: list[dict[str, Any]],
    expanded_markers: list[dict[str, Any]],
    patches_np: np.ndarray,
    oas_path: Path,
    config: dict[str, Any],
    recon_dir: Path,
    orig_dir: Path,
    rf_erosion_bypass: int,
    dbu_um: float,
    inspect_dir: Path | None,
) -> tuple[list[Path], list[Path], float]:
    """Run marching squares on the raw cSDF patches (no encode/decode).

    The rasterized cSDF is used directly as the "reconstructed" field.
    Cropping uses ``rf_erosion_bypass`` (= margin in pixels) rather than the
    model receptive-field erosion, because there is no encoder boundary
    contamination in bypass mode.

    Returns:
        ``(recon_paths, orig_paths, contour_ms)``
    """
    stem = Path(row["file"]).stem

    t0 = time.perf_counter()
    recon_paths, orig_paths = _postprocess_per_marker(
        x_hat_np=patches_np,          # bypass: cSDF IS the reconstruction
        markers=expanded_markers,
        oas_path=oas_path,
        config=config,
        recon_dir=recon_dir,
        orig_dir=orig_dir,
        rf_erosion=rf_erosion_bypass,
        dbu_um=dbu_um,
        stem=stem,
        inspect_dir=inspect_dir,
    )
    contour_ms = (time.perf_counter() - t0) * 1000.0

    log.debug("Bypass: %s  N=%d  contour_ms=%.1f", stem, len(expanded_markers), contour_ms)
    return recon_paths, orig_paths, contour_ms


# ─── Metrics ─────────────────────────────────────────────────────────────────


def _compute_bypass_metrics(
    row: dict[str, Any],
    orig_markers: list[dict[str, Any]],
    recon_paths: list[Path],
    orig_paths: list[Path],
    contour_ms: float,
    mask_layer: int,
) -> list[dict[str, Any]]:
    """Compute per-marker ADR for the bypass run.

    ``encode_ms`` is reported as 0.0 (no encoder); ``decode_ms`` holds the
    marching-squares wall-clock time split equally across markers.
    ``compression_ratio`` is ``None`` (no ``.cdna``).
    """
    n = len(orig_markers)
    contour_per = contour_ms / max(n, 1)

    results: list[dict[str, Any]] = []
    for i, (marker, recon_path, orig_path) in enumerate(
        zip(orig_markers, recon_paths, orig_paths)
    ):
        try:
            adr = area_difference_ratio(orig_path, recon_path, mask_layer=mask_layer)
        except Exception as exc:
            log.warning("area_difference_ratio failed for %s m%d: %s", row["file"], i, exc)
            adr = float("nan")

        results.append({
            "file":                  row["file"],
            "marker_idx":            i,
            "marker_x_nm":           float(marker["x_nm"]),
            "marker_y_nm":           float(marker["y_nm"]),
            "compression_ratio":     None,
            "area_difference_ratio": round(adr, 6) if not np.isnan(adr) else None,
            "encode_ms":             0.0,
            "decode_ms":             round(contour_per, 2),
        })
    return results


# ─── Main evaluation loop ────────────────────────────────────────────────────


def evaluate_bypass(
    config: dict[str, Any],
    output_dir: Path,
    split: str = "test",
    inspection: bool = False,
) -> None:
    """Run the bypass reference evaluation on a dataset split.

    Args:
        config: Parsed YAML config dict.
        output_dir: Root output directory; ``reconstructed/`` and
                    ``original/`` subdirs are created.
        split: Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
        inspection: Write per-marker 4-layer inspection ``.oas`` files to
                    ``<output_dir>/inspection/``.
    """
    recon_dir = output_dir / "reconstructed"
    orig_dir  = output_dir / "original"
    inspect_dir: Path | None = None
    if inspection:
        inspect_dir = output_dir / "inspection"
        inspect_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Inspection mode on — writing 4-layer .oas files to %s/  "
            "(layers orig=%d recon=%d xor=%d marker=%d)",
            inspect_dir.name,
            _INSP_ORIG_LAYER, _INSP_RECON_LAYER, _INSP_XOR_LAYER, _INSP_MARKER_LAYER,
        )
    for d in (recon_dir, orig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Manifest ──────────────────────────────────────────────────────────────
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]
    manifest  = _load_manifest(cache_dir)
    margin_nm = float(manifest.get("marker_margin_nm", config["csdf"].get("marker_margin_nm", 0.0)))
    log.info("marker_margin_nm = %.1f nm (from manifest)", margin_nm)

    # ── Derived parameters ────────────────────────────────────────────────────
    grid_res      = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    mask_layer    = int(config["csdf"]["mask_layer"])
    marker_layer  = int(config["csdf"]["marker_layer"])

    # In bypass mode: no model RF contamination — erode only by the margin so
    # the comparison region is the original (unexpanded) marker footprint.
    rf_erosion_bypass = round(margin_nm / grid_res)
    log.info(
        "bypass rf_erosion = %d px  (margin_nm=%.1f / grid_res=%.3f)",
        rf_erosion_bypass, margin_nm, grid_res,
    )

    # c=1 means patch sizes are not rounded up to model multiples.
    # The margin erosion always recovers the original marker footprint.
    c_align = 1

    # ── Catalog rows ──────────────────────────────────────────────────────────
    split_rows = _load_split_rows(config, split)
    if not split_rows:
        log.error(
            "No rows found for split '%s' in catalog.csv.  "
            "Run dataset/ingest.py with --split %s first.",
            split, split,
        )
        return

    log.info(
        "Bypass reference eval — split=%s  files=%d  rf_erosion=%d px",
        split, len(split_rows), rf_erosion_bypass,
    )

    all_results: list[dict[str, Any]] = []

    for row_idx, row in enumerate(split_rows):
        stem = Path(row["file"]).stem
        oas_path = _PROJECT_ROOT / row["file"]
        if not oas_path.exists():
            log.warning("OASIS not found, skipping: %s", oas_path)
            continue

        log.info("[%d/%d] %s", row_idx + 1, len(split_rows), row["file"])

        # Step 1: rasterize
        try:
            orig_markers, expanded_markers, patches_np, dbu_um = _rasterize_file_expanded(
                oas_path=oas_path,
                mask_layer=mask_layer,
                marker_layer=marker_layer,
                grid_res=grid_res,
                truncation_px=truncation_px,
                margin_nm=margin_nm,
                c=c_align,
            )
        except Exception as exc:
            log.error("Rasterization failed for %s: %s", row["file"], exc)
            continue

        if not orig_markers:
            log.warning("No markers found for %s — skipping", row["file"])
            continue

        log.debug("  markers=%d  S_exp=%d", len(orig_markers), patches_np.shape[1])

        # Step 2: bypass — feed raw cSDF directly to marching squares
        try:
            recon_paths, orig_paths, contour_ms = _bypass_file(
                row=row,
                orig_markers=orig_markers,
                expanded_markers=expanded_markers,
                patches_np=patches_np,
                oas_path=oas_path,
                config=config,
                recon_dir=recon_dir,
                orig_dir=orig_dir,
                rf_erosion_bypass=rf_erosion_bypass,
                dbu_um=dbu_um,
                inspect_dir=inspect_dir,
            )
        except Exception as exc:
            log.error("Bypass pass failed for %s: %s", row["file"], exc)
            continue

        if not recon_paths:
            log.warning("No valid contours for %s — skipping metrics", row["file"])
            continue

        # Step 3: metrics
        file_results = _compute_bypass_metrics(
            row=row,
            orig_markers=orig_markers[: len(recon_paths)],
            recon_paths=recon_paths,
            orig_paths=orig_paths,
            contour_ms=contour_ms,
            mask_layer=mask_layer,
        )
        all_results.extend(file_results)

        adrs = [r["area_difference_ratio"] for r in file_results if r.get("area_difference_ratio") is not None]
        log.info(
            "  markers=%d  ADR_mean=%.4f  contour_ms=%.1f",
            len(orig_markers),
            float(np.nanmean(adrs)) if adrs else float("nan"),
            contour_ms,
        )

    # ── Write reports ─────────────────────────────────────────────────────────
    write_results_csv(all_results, output_dir / "results.csv")
    _write_bypass_summary(all_results, output_dir / "summary.json", n_files=len(split_rows))
    log.info("Bypass evaluation complete.  Output: %s", output_dir)



"""Segment-center to target-boundary distance evaluation.

For every segment in the reconstructed PWCL layer, computes the Euclidean
distance (nm) from the segment's center point to the nearest edge on the
original (target) polygon boundary.  Statistics are reported to stdout and
written to a JSON summary file.

The script expects an ``evaluate.py`` output directory (which contains
``original/`` and ``reconstructed/`` per-marker ``.oas`` subdirectories) OR
explicitly specified ``--orig-dir`` / ``--recon-dir`` paths.  Pairs are
matched by filename stem.

Center-point definition per segment type
─────────────────────────────────────────
* LINE    — midpoint of the two endpoints
* ARC     — ``pts[1]``, the on-arc midpoint (third control point)
* BEZIER2 — ``B(0.5) = 0.25·p0 + 0.5·ctrl + 0.25·p2``
* BEZIER3 — ``B(0.5) = 0.125·p0 + 0.375·c1 + 0.375·c2 + 0.125·p3``

Usage::

    # Operate on an existing evaluate.py output directory
    python eval/evaluate_segment_distance.py \\
        --results-dir eval/results/baseline_v1/ \\
        --mask-layer 1

    # Operate on explicitly named directories
    python eval/evaluate_segment_distance.py \\
        --orig-dir  eval/results/baseline_v1/original/ \\
        --recon-dir eval/results/baseline_v1/reconstructed/ \\
        --mask-layer 1 \\
        --output    eval/results/baseline_v1/segment_distance_summary.json

    # Operate on an evaluate_correction.py output directory (baseline vs corrected)
    python eval/evaluate_segment_distance.py \\
        --correction-results-dir eval/results/correction_eval/ \\
        --mask-layer 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "io"))

from csdf.csdf_utils import PwclContour, PwclSegment, SegmentType

log = logging.getLogger(__name__)

# Sampling pitch used when building a cKDTree from original polygon edges.
# At 0.5 nm the approximation error is ≤ 0.25 nm (well below the 1 nm/px
# grid resolution), so the sampled-tree distance is effectively exact.
_SAMPLE_PITCH_NM: float = 0.5


# ─── Geometry helpers ─────────────────────────────────────────────────────────


def _segment_center(seg: PwclSegment) -> tuple[float, float]:
    """Return the center point of a PWCL segment in nm.

    Args:
        seg: A PwclSegment with control points in nm.

    Returns:
        (x_nm, y_nm) of the segment's parametric midpoint.
    """
    pts = seg.pts
    if seg.type == SegmentType.LINE:
        return ((pts[0][0] + pts[1][0]) * 0.5, (pts[0][1] + pts[1][1]) * 0.5)

    if seg.type == SegmentType.ARC:
        # pts = [start, on_arc_mid, end]; pts[1] is at t≈0.5
        return (float(pts[1][0]), float(pts[1][1]))

    if seg.type == SegmentType.BEZIER2:
        p0, c1, p2 = pts[0], pts[1], pts[2]
        return (0.25 * p0[0] + 0.5 * c1[0] + 0.25 * p2[0],
                0.25 * p0[1] + 0.5 * c1[1] + 0.25 * p2[1])

    if seg.type == SegmentType.BEZIER3:
        p0, c1, c2, p3 = pts[0], pts[1], pts[2], pts[3]
        return (0.125 * p0[0] + 0.375 * c1[0] + 0.375 * c2[0] + 0.125 * p3[0],
                0.125 * p0[1] + 0.375 * c1[1] + 0.375 * c2[1] + 0.125 * p3[1])

    # Fallback for unknown types: midpoint of first and last control points.
    return ((pts[0][0] + pts[-1][0]) * 0.5, (pts[0][1] + pts[-1][1]) * 0.5)


def _load_edges_from_oas(
    oas_path: Path,
    mask_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all polygon boundary edges from an OASIS file.

    Each polygon hull is converted to a sequence of directed line segments
    (closed, so the last vertex connects back to the first).

    Args:
        oas_path: Path to the OASIS layout file.
        mask_layer: GDS layer number of the mask polygons.

    Returns:
        ``(seg_starts, seg_ends)`` — two ``float64 [S, 2]`` arrays of segment
        start and end coordinates in nm.  Returns empty ``[0, 2]`` arrays if
        the layer contains no polygons.
    """
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu
    dbu_nm = dbu_um * 1000.0
    li = layout.layer(mask_layer, 0)

    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []

    for cell in layout.each_cell():
        for shape in cell.shapes(li).each():
            if shape.is_polygon():
                poly = shape.polygon
            elif shape.is_box():
                poly = db.Polygon(shape.box)
            else:
                continue
            pts = [(pt.x * dbu_nm, pt.y * dbu_nm) for pt in poly.each_point_hull()]
            n = len(pts)
            if n < 2:
                continue
            for i in range(n):
                starts.append(pts[i])
                ends.append(pts[(i + 1) % n])

    if not starts:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    return np.array(starts, dtype=np.float64), np.array(ends, dtype=np.float64)


def _load_segment_centers_from_oas(
    oas_path: Path,
    mask_layer: int,
) -> np.ndarray:
    """Load polygon hull segments from an OASIS file and return their centers.

    Polygons are read as LINE-segment PwclContours (the only segment type
    produced by the marching-squares pipeline).

    Args:
        oas_path: Path to the reconstructed OASIS file.
        mask_layer: GDS layer number of the mask polygons.

    Returns:
        ``float64 [Q, 2]`` array of segment center coordinates (nm).
        Empty ``[0, 2]`` array if no segments are found.
    """
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu
    dbu_nm = dbu_um * 1000.0
    li = layout.layer(mask_layer, 0)

    centers: list[tuple[float, float]] = []

    for cell in layout.each_cell():
        for shape in cell.shapes(li).each():
            if shape.is_polygon():
                poly = shape.polygon
            elif shape.is_box():
                poly = db.Polygon(shape.box)
            else:
                continue
            pts = [(pt.x * dbu_nm, pt.y * dbu_nm) for pt in poly.each_point_hull()]
            n = len(pts)
            if n < 2:
                continue
            for i in range(n):
                seg = PwclSegment(
                    type=SegmentType.LINE,
                    pts=[pts[i], pts[(i + 1) % n]],
                )
                centers.append(_segment_center(seg))

    if not centers:
        return np.zeros((0, 2), dtype=np.float64)

    return np.array(centers, dtype=np.float64)


# ─── KDTree-based distance computation ───────────────────────────────────────


def _build_boundary_tree(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> cKDTree:
    """Dense-sample polygon boundary edges and build a cKDTree.

    Each edge is sampled at ``_SAMPLE_PITCH_NM`` nm intervals (minimum 2
    points per edge: the two endpoints).  This gives ≤ ``_SAMPLE_PITCH_NM / 2``
    nm approximation error in the subsequent nearest-neighbour queries.

    Args:
        seg_starts: ``float64 [S, 2]`` — edge start coordinates (nm).
        seg_ends:   ``float64 [S, 2]`` — edge end coordinates (nm).

    Returns:
        cKDTree built from all sampled boundary points.
    """
    lengths = np.hypot(
        seg_ends[:, 0] - seg_starts[:, 0],
        seg_ends[:, 1] - seg_starts[:, 1],
    )

    chunks: list[np.ndarray] = []
    for a, b, L in zip(seg_starts, seg_ends, lengths):
        if L < 1e-10:
            chunks.append(a[None, :])
            continue
        n_pts = max(2, int(np.ceil(L / _SAMPLE_PITCH_NM)) + 1)
        t = np.linspace(0.0, 1.0, n_pts)
        chunks.append(a + t[:, None] * (b - a))

    all_pts = np.vstack(chunks)
    log.debug("Boundary tree: %d sample points from %d edges", len(all_pts), len(seg_starts))
    return cKDTree(all_pts)


def _min_dist_points_to_boundary(
    query_pts: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray:
    """Minimum distance from each query point to the original polygon boundary.

    Builds a cKDTree from densely sampled boundary points, then performs a
    batched nearest-neighbour query.  Memory is O(N_samples + Q) regardless
    of the number of edges.

    Args:
        query_pts:  ``float64 [Q, 2]`` — segment center coordinates (nm).
        seg_starts: ``float64 [S, 2]`` — original edge start coordinates.
        seg_ends:   ``float64 [S, 2]`` — original edge end coordinates.

    Returns:
        ``float64 [Q]`` array of minimum distances (nm).
    """
    tree = _build_boundary_tree(seg_starts, seg_ends)
    dists, _ = tree.query(query_pts, k=1, workers=-1)
    return np.asarray(dists, dtype=np.float64)


# ─── Per-pair processing ──────────────────────────────────────────────────────


def _process_pair(
    orig_path: Path,
    recon_path: Path,
    mask_layer: int,
) -> dict[str, Any] | None:
    """Compute distance statistics for one (original, reconstructed) pair.

    Args:
        orig_path: Path to the original ``.oas`` file.
        recon_path: Path to the reconstructed ``.oas`` file.
        mask_layer: GDS mask layer number.

    Returns:
        Dict with ``n_segments``, ``mean_nm``, ``std_nm``, ``min_nm``,
        ``max_nm``, ``median_nm``, ``p95_nm``, ``p99_nm`` for this pair.
        Returns ``None`` if either file has no usable geometry.
    """
    seg_starts, seg_ends = _load_edges_from_oas(orig_path, mask_layer)
    if len(seg_starts) == 0:
        log.debug("Original has no edges, skipping: %s", orig_path.name)
        return None

    centers = _load_segment_centers_from_oas(recon_path, mask_layer)
    if len(centers) == 0:
        log.debug("Reconstructed has no segments, skipping: %s", recon_path.name)
        return None

    distances = _min_dist_points_to_boundary(centers, seg_starts, seg_ends)

    return {
        "pair":        recon_path.stem,
        "n_segments":  int(len(distances)),
        "mean_nm":     float(distances.mean()),
        "std_nm":      float(distances.std()),
        "min_nm":      float(distances.min()),
        "max_nm":      float(distances.max()),
        "median_nm":   float(np.median(distances)),
        "p95_nm":      float(np.percentile(distances, 95)),
        "p99_nm":      float(np.percentile(distances, 99)),
    }


# ─── Aggregate statistics ─────────────────────────────────────────────────────


def _aggregate(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics over all per-pair distance arrays.

    Uses a numerically stable incremental mean/variance (Welford's online
    algorithm) to avoid materialising all distances at once.

    Args:
        per_pair: List of dicts returned by ``_process_pair``.

    Returns:
        Summary dict with global ``n_pairs``, ``n_segments_total``, and
        per-statistic aggregate columns (mean, std, min, max, p95, p99).
    """
    if not per_pair:
        return {"n_pairs": 0, "n_segments_total": 0}

    # Aggregate the per-pair summary statistics rather than re-materialising
    # all distances.  Mean and std are weighted by n_segments.
    n_total = sum(r["n_segments"] for r in per_pair)
    weighted_mean = sum(r["mean_nm"] * r["n_segments"] for r in per_pair) / n_total

    # Variance: E[X²] - E[X]²  via per-pair (var + mean²)
    weighted_var = (
        sum((r["std_nm"] ** 2 + r["mean_nm"] ** 2) * r["n_segments"] for r in per_pair)
        / n_total
        - weighted_mean ** 2
    )
    weighted_std = math.sqrt(max(weighted_var, 0.0))

    global_min = min(r["min_nm"] for r in per_pair)
    global_max = max(r["max_nm"] for r in per_pair)

    # p95/p99: report mean of per-pair percentiles (approximate but useful)
    p95_mean = float(np.mean([r["p95_nm"] for r in per_pair]))
    p99_mean = float(np.mean([r["p99_nm"] for r in per_pair]))
    median_mean = float(np.mean([r["median_nm"] for r in per_pair]))

    return {
        "n_pairs":          len(per_pair),
        "n_segments_total": int(n_total),
        "mean_nm":          round(weighted_mean, 4),
        "std_nm":           round(weighted_std, 4),
        "min_nm":           round(global_min, 4),
        "max_nm":           round(global_max, 4),
        "median_nm":        round(median_mean, 4),
        "p95_nm":           round(p95_mean, 4),
        "p99_nm":           round(p99_mean, 4),
    }


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def _write_per_pair_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-pair distance statistics to a CSV file."""
    if not rows:
        return
    cols = ["pair", "n_segments", "mean_nm", "std_nm", "min_nm", "max_nm",
            "median_nm", "p95_nm", "p99_nm"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Per-pair CSV → %s  (%d row(s))", path, len(rows))


# ─── Main ─────────────────────────────────────────────────────────────────────


def evaluate_segment_distance(
    orig_dir: Path,
    recon_dir: Path,
    mask_layer: int,
    output_json: Path,
) -> dict[str, Any]:
    """Compute segment-center distances for all matched pairs in two directories.

    Files in ``recon_dir`` are matched to files in ``orig_dir`` by stem
    (e.g., ``design_m0000.oas`` in both directories).

    Args:
        orig_dir:    Directory containing original per-marker ``.oas`` files.
        recon_dir:   Directory containing reconstructed per-marker ``.oas`` files.
        mask_layer:  GDS layer number of mask polygons.
        output_json: Destination path for the aggregate JSON summary.

    Returns:
        Aggregate summary dict (also written to ``output_json``).
    """
    recon_files = sorted(recon_dir.glob("*.oas"))
    if not recon_files:
        log.warning("No .oas files found in recon_dir: %s", recon_dir)
        return {}

    total = len(recon_files)
    per_pair: list[dict[str, Any]] = []
    skipped = 0

    log.info("Processing %d file pair(s) from %s", total, recon_dir.name)

    for idx, recon_path in enumerate(recon_files, 1):
        orig_path = orig_dir / recon_path.name
        if not orig_path.exists():
            log.debug("No matching original for %s — skipping", recon_path.name)
            skipped += 1
            continue

        try:
            result = _process_pair(orig_path, recon_path, mask_layer)
        except Exception as exc:
            log.warning("Error processing %s: %s", recon_path.name, exc)
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        per_pair.append(result)
        log.info(
            "[%d/%d] %s  n=%d  mean=%.3f nm  p95=%.3f nm",
            idx, total, result["pair"],
            result["n_segments"], result["mean_nm"], result["p95_nm"],
        )

    log.info(
        "Pairs processed: %d  skipped: %d",
        len(per_pair), skipped,
    )

    summary = _aggregate(per_pair)

    # Write per-pair CSV alongside the summary JSON
    csv_path = output_json.with_suffix(".csv")
    _write_per_pair_csv(per_pair, csv_path)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Summary JSON → %s", output_json)
    return summary


# ─── Printing helpers ────────────────────────────────────────────────────────


def _print_summary(label: str, summary: dict[str, Any], output_json: Path) -> None:
    """Print a human-readable distance summary table to stdout."""
    print(f"\n── {label} ──")
    print(f"  Pairs evaluated  : {summary['n_pairs']}")
    print(f"  Total segments   : {summary['n_segments_total']}")
    print(f"  Mean   (nm)      : {summary['mean_nm']:.4f}")
    print(f"  Std    (nm)      : {summary['std_nm']:.4f}")
    print(f"  Min    (nm)      : {summary['min_nm']:.4f}")
    print(f"  Max    (nm)      : {summary['max_nm']:.4f}")
    print(f"  Median (nm)      : {summary['median_nm']:.4f}")
    print(f"  P95    (nm)      : {summary['p95_nm']:.4f}")
    print(f"  P99    (nm)      : {summary['p99_nm']:.4f}")
    print(f"  Output           : {output_json}")


def _print_comparison(base: dict[str, Any], corr: dict[str, Any]) -> None:
    """Print a side-by-side baseline vs. corrected comparison table."""
    metrics = ["mean_nm", "std_nm", "median_nm", "p95_nm", "p99_nm", "max_nm"]
    labels  = ["Mean  ", "Std   ", "Median", "P95   ", "P99   ", "Max   "]

    print("\n── Segment-distance comparison: baseline vs. corrected ───────────")
    print(f"  {'Metric':<10}  {'Baseline':>10}  {'Corrected':>10}  {'Δ (corr−base)':>14}")
    print("  " + "─" * 52)
    for label, key in zip(labels, metrics):
        b, c = base.get(key, float("nan")), corr.get(key, float("nan"))
        delta = c - b
        sign  = "+" if delta > 0 else ""
        print(f"  {label}  {b:>10.4f}  {c:>10.4f}  {sign}{delta:>13.4f}")
    print("  " + "─" * 52)
    print("  (negative Δ = corrected reconstruction is closer to target)")
    print("──────────────────────────────────────────────────────────────────")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    loc = p.add_mutually_exclusive_group()
    loc.add_argument(
        "--results-dir", type=Path,
        metavar="DIR",
        help=(
            "Root output directory from evaluate.py.  "
            "``<DIR>/original/`` and ``<DIR>/reconstructed/`` are used automatically."
        ),
    )
    loc.add_argument(
        "--correction-results-dir", type=Path,
        metavar="DIR",
        help=(
            "Root output directory from evaluate_correction.py.  "
            "Runs the distance metric on both ``<DIR>/baseline/reconstructed/`` "
            "and ``<DIR>/corrected/reconstructed/`` against ``<DIR>/original/``, "
            "then prints a side-by-side comparison."
        ),
    )

    p.add_argument(
        "--orig-dir", type=Path, metavar="DIR",
        help="Directory of original per-marker .oas files.",
    )
    p.add_argument(
        "--recon-dir", type=Path, metavar="DIR",
        help="Directory of reconstructed per-marker .oas files.",
    )
    p.add_argument(
        "--mask-layer", type=int, default=1,
        help="GDS layer number of mask polygons (default: 1).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output JSON summary path.  "
            "Defaults to ``<results-dir>/segment_distance_summary.json`` "
            "or ``<recon-dir>/../segment_distance_summary.json``.  "
            "Ignored when ``--correction-results-dir`` is used "
            "(outputs are written automatically alongside the correction results)."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    mask_layer = args.mask_layer

    # ── Correction-eval mode: baseline vs. corrected side by side ────────────
    if args.correction_results_dir is not None:
        root     = args.correction_results_dir
        orig_dir = root / "original"
        base_recon_dir = root / "baseline"  / "reconstructed"
        corr_recon_dir = root / "corrected" / "reconstructed"

        for d, label in (
            (orig_dir,       "original"),
            (base_recon_dir, "baseline/reconstructed"),
            (corr_recon_dir, "corrected/reconstructed"),
        ):
            if not d.is_dir():
                log.error(
                    "%s does not exist or is not a directory: %s", label, d
                )
                sys.exit(1)

        log.info("Correction mode: %s", root)

        base_json = root / "segment_distance_baseline.json"
        corr_json = root / "segment_distance_corrected.json"

        log.info("── Baseline pass ──")
        base_summary = evaluate_segment_distance(
            orig_dir=orig_dir,
            recon_dir=base_recon_dir,
            mask_layer=mask_layer,
            output_json=base_json,
        )

        log.info("── Corrected pass ──")
        corr_summary = evaluate_segment_distance(
            orig_dir=orig_dir,
            recon_dir=corr_recon_dir,
            mask_layer=mask_layer,
            output_json=corr_json,
        )

        if not base_summary or not corr_summary:
            log.warning("No results produced.")
            sys.exit(0)

        _print_summary("Baseline reconstruction", base_summary, base_json)
        _print_summary("Corrected reconstruction", corr_summary, corr_json)
        _print_comparison(base_summary, corr_summary)
        return

    # ── Standard mode ────────────────────────────────────────────────────────
    if args.results_dir is not None:
        orig_dir  = args.results_dir / "original"
        recon_dir = args.results_dir / "reconstructed"
        default_output = args.results_dir / "segment_distance_summary.json"
    elif args.orig_dir is not None and args.recon_dir is not None:
        orig_dir  = args.orig_dir
        recon_dir = args.recon_dir
        default_output = recon_dir.parent / "segment_distance_summary.json"
    else:
        log.error(
            "Provide --results-dir, --correction-results-dir, "
            "or both --orig-dir and --recon-dir."
        )
        sys.exit(1)

    for d, label in ((orig_dir, "orig-dir"), (recon_dir, "recon-dir")):
        if not d.is_dir():
            log.error("%s does not exist or is not a directory: %s", label, d)
            sys.exit(1)

    output_json = args.output if args.output is not None else default_output

    summary = evaluate_segment_distance(
        orig_dir=orig_dir,
        recon_dir=recon_dir,
        mask_layer=mask_layer,
        output_json=output_json,
    )

    if not summary:
        log.warning("No results produced.")
        sys.exit(0)

    _print_summary("Segment-center distance to original boundary", summary, output_json)


if __name__ == "__main__":
    main()

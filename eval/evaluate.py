"""Evaluation script for a trained CurveCodec checkpoint.

Runs the full encode → decode pipeline on the **test** split and computes
official metrics.  Execution order is strictly enforced:

1. **Encode** — rasterize test patches → export integer DNA → save ``.cdna``
   to ``<output-dir>/dna/``
2. **Decode** — load ``.cdna`` from disk → dequantize → decode → contour →
   save per-marker reconstructed OASIS to ``<output-dir>/reconstructed/``
3. **Metrics** — compare original vs reconstructed OASIS files; write
   ``<output-dir>/results.csv`` and ``<output-dir>/summary.json``

Metrics are always computed from files on disk — never from in-memory
tensors.

Usage::

    python eval/evaluate.py \\
        --checkpoint checkpoints/baseline_v1/best.pt \\
        --config     train/config/baseline.yaml \\
        --data-dir   dataset/raw/test/ \\
        --output-dir eval/results/baseline_v1/ \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
# io/ package name shadows Python's builtin io — import submodules directly
# by adding the io/ directory itself to sys.path.
sys.path.insert(0, str(_PROJECT_ROOT / "io"))

from codec.model import CurveCodec
from contouring.contour import csdf_to_contours
from eval.metrics import area_difference_ratio, compression_ratio
from eval.report import write_results_csv, write_summary_json
from layout_io import read_polygons_in_region, write_oas
from pack import save_cdna
from unpack import dequantize, load_cdna

log = logging.getLogger(__name__)

_DATASET_DIR = _PROJECT_ROOT / "dataset"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def _load_test_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return catalog rows for the test split."""
    catalog_path = _DATASET_DIR / "catalog.csv"
    from dataset.ingest import load_catalog

    all_rows = load_catalog(catalog_path)
    rows = [r for r in all_rows if r["split"] == "test"]
    log.info("Test split: %d file(s) in catalog", len(rows))
    return rows


def _load_markers(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load per-marker metadata from the cache sidecar YAML.

    Falls back to scanning the raw OASIS file if the sidecar is absent.
    """
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]
    from dataset.rasterize import meta_path

    meta_file = meta_path(row, cache_dir)
    if meta_file.exists():
        with open(meta_file) as f:
            meta = yaml.safe_load(f)
        return meta.get("markers", [])

    # Fallback: scan the raw OASIS file for square markers
    log.warning(
        "Cache sidecar not found for %s — scanning raw OASIS file", row["file"]
    )
    import klayout.db as db

    oas_abs = _PROJECT_ROOT / row["file"]
    if not oas_abs.exists():
        log.error("Raw OASIS not found: %s", oas_abs)
        return []

    layout = db.Layout()
    layout.read(str(oas_abs))
    dbu_um: float = layout.dbu
    marker_li = layout.layer(int(config["csdf"]["marker_layer"]), 0)
    markers = []
    for cell in layout.each_cell():
        for shape in cell.shapes(marker_li).each():
            if not shape.is_box():
                continue
            box = shape.box
            w_nm = box.width() * dbu_um * 1000.0
            h_nm = box.height() * dbu_um * 1000.0
            if abs(w_nm - h_nm) > 0.5:
                continue
            markers.append(
                {
                    "x_nm": box.left * dbu_um * 1000.0,
                    "y_nm": box.bottom * dbu_um * 1000.0,
                    "size_nm": w_nm,
                }
            )
    return markers


def _load_patches(
    row: dict[str, Any],
    markers: list[dict[str, Any]],
    config: dict[str, Any],
) -> np.ndarray:
    """Load cSDF patches for all markers of one OASIS file.

    Returns:
        float32 numpy array ``[N, S, S]``.
    """
    from dataset.dataset import CsdfDataset

    # Re-use dataset logic: build a temporary single-file dataset
    mode = config["dataset"]["mode"]
    if mode == "cached":
        cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]
        from dataset.rasterize import npy_path, meta_path

        canvas_file = npy_path(row, cache_dir)
        meta_file = meta_path(row, cache_dir)
        if not canvas_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"Cache missing for {row['file']} — run dataset/rasterize.py first."
            )
        canvas = np.load(canvas_file)
        with open(meta_file) as f:
            meta = yaml.safe_load(f)

        x0_nm = float(meta["canvas_x0_nm"])
        y0_nm = float(meta["canvas_y0_nm"])
        grid_res = float(config["csdf"]["grid_res_nm_per_px"])
        S = int(row["patch_size_px"])

        patches = []
        for m in markers:
            col0 = round((m["x_nm"] - x0_nm) / grid_res)
            row0 = round((m["y_nm"] - y0_nm) / grid_res)
            patch = canvas[row0 : row0 + S, col0 : col0 + S]
            if patch.shape != (S, S):
                raise RuntimeError(
                    f"Patch shape {patch.shape} != ({S}, {S}) for "
                    f"{Path(row['file']).name} marker ({m['x_nm']:.1f}, {m['y_nm']:.1f}) nm"
                )
            patches.append(patch)
        return np.stack(patches, axis=0)  # [N, S, S]

    # on_the_fly mode: use the dataset _rasterize_on_the_fly logic
    # Build a minimal temporary dataset just for this file
    ds = CsdfDataset.__new__(CsdfDataset)
    ds._split = row["split"]
    ds._mode = "on_the_fly"
    ds._grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    ds._truncation_px = float(config["csdf"]["truncation_px"])
    ds._mask_layer = int(config["csdf"]["mask_layer"])
    ds._marker_layer = int(config["csdf"]["marker_layer"])
    ds._cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]
    ds._raw_dir = _PROJECT_ROOT / config["paths"]["dataset_root"]
    ds._layout_cache: dict[str, Any] = {}
    ds._rows = [row]
    ds._items = [(row, m) for m in markers]

    patches = []
    for m in markers:
        patch = ds._rasterize_on_the_fly(row, m)
        patches.append(patch)
    return np.stack(patches, axis=0)  # [N, S, S]


# ─── Encode pass ─────────────────────────────────────────────────────────────


def encode_file(
    row: dict[str, Any],
    markers: list[dict[str, Any]],
    patches_np: np.ndarray,
    model: CurveCodec,
    device: torch.device,
    config: dict[str, Any],
    dna_dir: Path,
    checkpoint_path: Path,
) -> tuple[Path, float]:
    """Encode all patches of one OASIS file → ``.cdna``.

    Args:
        row: Catalog row dict.
        markers: Per-marker metadata dicts.
        patches_np: float32 array ``[N, S, S]``.
        model: Loaded CurveCodec in eval mode.
        device: Target device.
        config: YAML config dict.
        dna_dir: Output directory for ``.cdna`` files.
        checkpoint_path: Used for checkpoint_hash in meta.json.

    Returns:
        ``(cdna_path, encode_ms)`` — path to the saved archive and wall-clock
        encode time in milliseconds.
    """
    stem = Path(row["file"]).stem
    cdna_path = dna_dir / f"{stem}.cdna"

    x = torch.from_numpy(patches_np[:, None, :, :]).to(device)  # [N, 1, S, S]

    t0 = time.perf_counter()
    with torch.no_grad():
        dna_tensor = model.export_dna(x)  # [N, D, Sc, Sc]  int8/int16
    encode_ms = (time.perf_counter() - t0) * 1000.0

    save_cdna(
        path=cdna_path,
        dna=dna_tensor,
        config=config,
        scale_factors=model.quantizer.scale_factors,
        markers=markers,
        checkpoint_path=checkpoint_path,
    )
    log.debug("Encode: %s  N=%d  %.1f ms → %s", stem, len(markers), encode_ms, cdna_path)
    return cdna_path, encode_ms


# ─── Decode pass ─────────────────────────────────────────────────────────────


def decode_file(
    cdna_path: Path,
    model: CurveCodec,
    device: torch.device,
    recon_dir: Path,
    orig_dir: Path,
    oas_path: Path,
    config: dict[str, Any],
) -> tuple[list[Path], list[Path], float]:
    """Decode ``.cdna`` → reconstructed per-marker OASIS files.

    Loads the archive from disk (no in-memory shortcuts), dequantizes,
    decodes each patch, extracts contours, and writes one ``.oas`` per
    marker.  Also extracts the corresponding original polygons and writes
    them alongside for metric comparison.

    Args:
        cdna_path: Path to the ``.cdna`` archive.
        model: CurveCodec in eval mode.
        device: Target device.
        recon_dir: Output directory for reconstructed ``.oas`` files.
        orig_dir: Output directory for extracted-original ``.oas`` files.
        oas_path: Original OASIS file (for polygon extraction).
        config: YAML config dict.

    Returns:
        ``(recon_paths, orig_paths, decode_ms)`` — per-marker reconstructed
        and extracted-original paths, and total decode wall-clock time in ms.
    """
    dna, meta = load_cdna(cdna_path)  # load from disk

    z = dequantize(dna, meta["scale_factors"]).to(device)  # [N, D, Sc, Sc]

    t0 = time.perf_counter()
    with torch.no_grad():
        x_hat = model.decode(z)  # [N, 1, S, S]
    decode_ms = (time.perf_counter() - t0) * 1000.0

    x_hat_np = x_hat.cpu().numpy()[:, 0, :, :]  # [N, S, S]

    grid_res = float(config["csdf"]["grid_res_nm_per_px"])
    mask_layer = int(config["csdf"]["mask_layer"])
    stem = cdna_path.stem

    recon_paths: list[Path] = []
    orig_paths: list[Path] = []

    for i, marker in enumerate(meta["markers"]):
        csdf_patch = x_hat_np[i]  # [S, S]

        # Contour in physical nm
        contours = csdf_to_contours(
            csdf=csdf_patch.astype(np.float32),
            origin_x_nm=float(marker["x_nm"]),
            origin_y_nm=float(marker["y_nm"]),
            grid_res_nm_per_px=grid_res,
        )

        # Reconstruct hull point lists from PwclContour objects
        recon_hulls: list[list[tuple[float, float]]] = []
        for contour in contours:
            pts: list[tuple[float, float]] = []
            for seg in contour.segments:
                pts.append(seg.pts[0])
            if pts:
                recon_hulls.append(pts)

        # Write reconstructed .oas
        recon_path = recon_dir / f"{stem}_m{i:04d}.oas"
        write_oas(recon_path, recon_hulls, mask_layer=mask_layer)
        recon_paths.append(recon_path)

        # Extract original polygons within this marker region and write .oas
        x0 = float(marker["x_nm"])
        y0 = float(marker["y_nm"])
        size = float(marker["size_nm"])
        orig_hulls = read_polygons_in_region(
            oas_path=oas_path,
            mask_layer=mask_layer,
            x0_nm=x0, y0_nm=y0,
            x1_nm=x0 + size, y1_nm=y0 + size,
        )
        orig_path = orig_dir / f"{stem}_m{i:04d}.oas"
        write_oas(orig_path, orig_hulls, mask_layer=mask_layer)
        orig_paths.append(orig_path)

    log.debug(
        "Decode: %s  N=%d  %.1f ms",
        cdna_path.name, len(meta["markers"]), decode_ms,
    )
    return recon_paths, orig_paths, decode_ms


# ─── Metrics pass ────────────────────────────────────────────────────────────


def compute_metrics(
    row: dict[str, Any],
    markers: list[dict[str, Any]],
    oas_path: Path,
    cdna_path: Path,
    recon_paths: list[Path],
    orig_paths: list[Path],
    encode_ms: float,
    decode_ms: float,
    mask_layer: int,
) -> list[dict[str, Any]]:
    """Compute per-marker metrics from on-disk files.

    Args:
        row: Catalog row dict.
        markers: Per-marker metadata dicts.
        oas_path: Original OASIS file (for compression_ratio).
        cdna_path: Encoded ``.cdna`` file.
        recon_paths: Per-marker reconstructed ``.oas`` paths.
        orig_paths: Per-marker extracted-original ``.oas`` paths.
        encode_ms: Encode wall-clock time for this file (ms).
        decode_ms: Decode wall-clock time for this file (ms).
        mask_layer: GDS layer of mask polygons.

    Returns:
        List of result dicts, one per marker.
    """
    cr = compression_ratio(oas_path, cdna_path)
    n = len(markers)
    # Distribute encode/decode time equally across markers
    enc_per = encode_ms / max(n, 1)
    dec_per = decode_ms / max(n, 1)

    results: list[dict[str, Any]] = []
    for i, (marker, recon_path, orig_path) in enumerate(
        zip(markers, recon_paths, orig_paths)
    ):
        try:
            adr = area_difference_ratio(orig_path, recon_path, mask_layer=mask_layer)
        except Exception as exc:
            log.warning("area_difference_ratio failed for %s m%d: %s", row["file"], i, exc)
            adr = float("nan")

        results.append(
            {
                "file": row["file"],
                "marker_idx": i,
                "marker_x_nm": float(marker["x_nm"]),
                "marker_y_nm": float(marker["y_nm"]),
                "compression_ratio": round(cr, 4),
                "area_difference_ratio": round(adr, 6) if not np.isnan(adr) else None,
                "encode_ms": round(enc_per, 2),
                "decode_ms": round(dec_per, 2),
            }
        )
    return results


# ─── Main evaluation loop ────────────────────────────────────────────────────


def evaluate(
    checkpoint_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run the full encode → decode → metrics pipeline on the test split.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint.
        config: Parsed YAML config dict.
        output_dir: Root output directory; subdirs ``dna/``,
                    ``reconstructed/``, and ``original/`` are created.
        device: PyTorch device.
    """
    dna_dir   = output_dir / "dna"
    recon_dir = output_dir / "reconstructed"
    orig_dir  = output_dir / "original"
    for d in (dna_dir, recon_dir, orig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Use config from checkpoint so model architecture is always consistent
    ckpt_cfg: dict[str, Any] = ckpt.get("config", config)
    model = CurveCodec(ckpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(
        "Loaded checkpoint: epoch=%d  best_val=%.6f",
        ckpt.get("epoch", -1),
        ckpt.get("best_val_loss", float("nan")),
    )

    mask_layer = int(config["csdf"]["mask_layer"])

    # ── Test catalog rows ─────────────────────────────────────────────────────
    test_rows = _load_test_rows(config)
    if not test_rows:
        log.error(
            "No test rows found in catalog.csv.  "
            "Run dataset/ingest.py with --split test first."
        )
        return

    all_results: list[dict[str, Any]] = []

    for row_idx, row in enumerate(test_rows):
        stem = Path(row["file"]).stem
        oas_path = _PROJECT_ROOT / row["file"]
        if not oas_path.exists():
            log.warning("OASIS not found, skipping: %s", oas_path)
            continue

        log.info(
            "[%d/%d] %s",
            row_idx + 1, len(test_rows), row["file"],
        )

        # ── Load markers ──────────────────────────────────────────────────────
        markers = _load_markers(row, config)
        if not markers:
            log.warning("No markers found for %s — skipping", row["file"])
            continue

        # ── Load patches ──────────────────────────────────────────────────────
        try:
            patches_np = _load_patches(row, markers, config)
        except Exception as exc:
            log.error("Failed to load patches for %s: %s", row["file"], exc)
            continue

        # ── Step 1: Encode → .cdna ────────────────────────────────────────────
        try:
            cdna_path, encode_ms = encode_file(
                row=row,
                markers=markers,
                patches_np=patches_np,
                model=model,
                device=device,
                config=config,
                dna_dir=dna_dir,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            log.error("Encode failed for %s: %s", row["file"], exc)
            continue

        # ── Step 2: Decode .cdna → reconstructed .oas ─────────────────────────
        try:
            recon_paths, orig_paths, decode_ms = decode_file(
                cdna_path=cdna_path,
                model=model,
                device=device,
                recon_dir=recon_dir,
                orig_dir=orig_dir,
                oas_path=oas_path,
                config=config,
            )
        except Exception as exc:
            log.error("Decode failed for %s: %s", row["file"], exc)
            continue

        # ── Step 3: Metrics ───────────────────────────────────────────────────
        file_results = compute_metrics(
            row=row,
            markers=markers,
            oas_path=oas_path,
            cdna_path=cdna_path,
            recon_paths=recon_paths,
            orig_paths=orig_paths,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
            mask_layer=mask_layer,
        )
        all_results.extend(file_results)

        # Per-file progress log
        crs  = [r["compression_ratio"] for r in file_results if r.get("compression_ratio")]
        adrs = [r["area_difference_ratio"] for r in file_results
                if r.get("area_difference_ratio") is not None]
        log.info(
            "  markers=%d  CR=%.2f  ADR_mean=%.4f  enc=%.1fms  dec=%.1fms",
            len(markers),
            crs[0] if crs else float("nan"),
            float(np.nanmean(adrs)) if adrs else float("nan"),
            encode_ms,
            decode_ms,
        )

    # ── Write report ──────────────────────────────────────────────────────────
    write_results_csv(all_results, output_dir / "results.csv")
    write_summary_json(
        all_results,
        output_dir / "summary.json",
        checkpoint_path=checkpoint_path,
        n_files=len(test_rows),
    )

    log.info("Evaluation complete.  Output: %s", output_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=_PROJECT_ROOT / "checkpoints" / "baseline_v1" / "best.pt",
        help="Path to .pt checkpoint (default: checkpoints/baseline_v1/best.pt)",
    )
    p.add_argument(
        "--config", type=Path,
        default=_PROJECT_ROOT / "train" / "config" / "baseline.yaml",
        help="YAML config file (default: train/config/baseline.yaml)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "eval" / "results" / "baseline_v1",
        help="Root directory for outputs (default: eval/results/baseline_v1/)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device (default: cuda if available, else cpu)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    config = _load_config(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

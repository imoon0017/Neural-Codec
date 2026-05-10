"""Neural codec evaluation suite — unified entry point.

Mode is selected by optional flags; the default path is the full codec eval.

Flags
-----
(none / --checkpoint only)
    Full encode → quantize → decode → ADR pipeline.
--bypass
    Rasterize → marching-squares only; no codec, no checkpoint required.
    Measures the theoretical ADR floor introduced by discretisation alone.
--correction-factor FLOAT
    Baseline + iterative field-correction comparison (ADR + field MSE).
    Requires --checkpoint.
--segment-distance
    Segment-centre EPE to nearest original boundary edge (mean, std, p95, …).
    Runs after the main eval pass, or standalone with --results-dir /
    --correction-results-dir / --orig-dir + --recon-dir.

Usage::

    python eval/evaluate.py \\
        --checkpoint checkpoints/baseline_v1/best.pt \\
        --config     train/config/baseline.yaml \\
        --output-dir eval/results/baseline_v1/ \\
        --device cuda --split test [--inspection]

    python eval/evaluate.py --bypass \\
        --config     train/config/baseline.yaml \\
        --output-dir eval/results/bypass_reference/

    python eval/evaluate.py \\
        --checkpoint checkpoints/baseline_v1/best.pt \\
        --config     train/config/baseline.yaml \\
        --output-dir eval/results/correction_baseline_v1/ \\
        --correction-factor 0.5 --n-iterations 5

    python eval/evaluate.py --segment-distance \\
        --results-dir eval/results/baseline_v1/ \\
        --mask-layer 1 [--spec 2.0]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import defaultdict
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
from codec.model_attn import CurveCodecAttn
from codec.model_multiattn import CurveCodecMultiAttn
from codec.model_v2 import CurveCodecV2
from contouring.contour import csdf_to_contours
from csdf.csdf_utils import rasterize_canvas
from dataset.rasterize import _dbu_to_nm, _poly_to_pwcl, _shape_to_polygon
from eval.metrics import area_difference_ratio, compression_ratio
from eval.report import write_results_csv, write_summary_json
from layout_io import read_polygons_in_region, write_oas
from pack import save_cdna
from unpack import dequantize, load_cdna

log = logging.getLogger(__name__)

_DATASET_DIR = _PROJECT_ROOT / "dataset"

# ── Inspection layer numbers (written into per-marker *_inspect.oas files) ────
_INSP_ORIG_LAYER   = 1   # original mask polygons  (cropped to valid area)
_INSP_RECON_LAYER  = 2   # reconstructed polygons  (cropped to valid area)
_INSP_XOR_LAYER    = 3   # XOR symmetric difference (cropped to valid area)
_INSP_MARKER_LAYER = 10  # valid-marker boundary box


# ─── Architecture helpers ─────────────────────────────────────────────────────


def _patch_size_px(size_nm: float, grid_res: float, c: int) -> int:
    """Pixel side-length for a region of *size_nm*, rounded up to a multiple of *c*."""
    s = math.ceil(size_nm / grid_res)
    rem = s % c
    return s + (c - rem) % c


def _rf_erosion_px(config: dict[str, Any]) -> int:
    """Encoder RF erosion radius in image-space pixels.

    Encoder architecture: stem Conv(3×3, pad=1) + N×DownBlock Conv(3×3,
    stride=2, pad=1).  The total receptive field is RF = 2c + 1, so the
    erosion radius (pixels contaminated by zero-padding at each edge) is c.
    """
    return int(config["model"]["compaction_ratio"])


# ─── Config / manifest ────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    """Load ``cache/manifest.yaml`` written by dataset/rasterize.py."""
    manifest_path = cache_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.yaml not found at {manifest_path}. "
            "Run dataset/rasterize.py first to generate it."
        )
    with open(manifest_path) as f:
        return yaml.safe_load(f)


# ─── Catalog helpers ─────────────────────────────────────────────────────────


def _load_split_rows(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return catalog rows for the given split."""
    catalog_path = _DATASET_DIR / "catalog.csv"
    from dataset.ingest import load_catalog

    all_rows = load_catalog(catalog_path)
    rows = [r for r in all_rows if r["split"] == split]
    log.info("%s split: %d file(s) in catalog", split, len(rows))
    return rows


# ─── Rasterization (fresh from raw OASIS) ────────────────────────────────────


def _rasterize_file_expanded(
    oas_path: Path,
    mask_layer: int,
    marker_layer: int,
    grid_res: float,
    truncation_px: float,
    margin_nm: float,
    c: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, float]:
    """Load a raw OASIS file and rasterize expanded-marker cSDF patches.

    For every square marker on *marker_layer*, the bounding box is enlarged by
    *margin_nm* on all four sides.  The expanded patch size is rounded up to a
    multiple of *c* so the encoder can process it without remainder.

    A single ``rasterize_canvas`` call handles all markers in the file; the
    per-marker patches are then cropped from the resulting canvas.

    Args:
        oas_path: Path to the raw OASIS file.
        mask_layer: GDS layer of mask polygons.
        marker_layer: GDS layer of square marker shapes.
        grid_res: nm per pixel.
        truncation_px: cSDF truncation half-width in pixels.
        margin_nm: Expansion distance on each side (nm).
        c: Compaction ratio (must be a power of 2).

    Returns:
        orig_markers: Original marker dicts (``x_nm``, ``y_nm``, ``size_nm``).
        expanded_markers: Expanded marker dicts (``x_nm``, ``y_nm``, ``size_nm``
            in nm, where ``size_nm = S_exp * grid_res``).
        patches: ``float32 [N, S_exp, S_exp]`` cSDF patches.
        dbu_um: Layout DBU in µm (e.g. ``0.001`` for 1 nm grid).  Must be
            forwarded to ``write_oas`` so output coordinates match the source.
    """
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(oas_path))
    dbu_um: float = layout.dbu
    mask_li = layout.layer(mask_layer, 0)
    marker_li = layout.layer(marker_layer, 0)

    # ── Collect square markers ────────────────────────────────────────────────
    raw_markers: list[tuple[db.Box, float, float, float]] = []  # box, x_nm, y_nm, size_nm
    layout_bbox = db.Box()

    for cell in layout.each_cell():
        for shape in cell.shapes(marker_li).each():
            if not shape.is_box():
                continue
            box = shape.box
            w_nm = _dbu_to_nm(box.width(), dbu_um)
            h_nm = _dbu_to_nm(box.height(), dbu_um)
            if abs(w_nm - h_nm) > 0.5:
                continue
            raw_markers.append((
                box,
                _dbu_to_nm(box.left,   dbu_um),
                _dbu_to_nm(box.bottom, dbu_um),
                w_nm,
            ))
            layout_bbox += box

    if not raw_markers:
        return [], [], np.zeros((0, 0, 0), dtype=np.float32), dbu_um

    # ── Group mask polygons by marker (centroid-in-box rule) ──────────────────
    marker_contours: dict[int, list] = defaultdict(list)

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
            for m_idx, (box, *_) in enumerate(raw_markers):
                if box.contains(centroid):
                    marker_contours[m_idx].extend(_poly_to_pwcl(poly, dbu_um))
                    break

    valid_indices = [i for i in range(len(raw_markers)) if i in marker_contours]
    if not valid_indices:
        return [], [], np.zeros((0, 0, 0), dtype=np.float32), dbu_um

    # ── Canvas dimensions sized for the natural (un-rounded) expanded patches ───
    # s_nat = ceil((size + 2*margin) / grid_res) — no c-rounding, matches rasterize.py.
    # s_exp = s_nat rounded up to multiple of c — required by the encoder.
    # The canvas must cover s_nat; s_exp - s_nat <= c-1 < c = rf_erosion, so the
    # zero-padding added below is always removed by the rf-erosion crop.
    canvas_x0_nm = _dbu_to_nm(layout_bbox.left,   dbu_um) - margin_nm
    canvas_y0_nm = _dbu_to_nm(layout_bbox.bottom, dbu_um) - margin_nm
    canvas_W = math.ceil(
        (_dbu_to_nm(layout_bbox.width(),  dbu_um) + 2.0 * margin_nm) / grid_res
    )
    canvas_H = math.ceil(
        (_dbu_to_nm(layout_bbox.height(), dbu_um) + 2.0 * margin_nm) / grid_res
    )

    # ── Batch: use s_nat so patch size matches canvas dimensions ─────────────
    # Passing s_exp here would make rasterize_canvas try to write pixels beyond
    # the canvas right/top boundary, silently clipping the top and right edges.
    batch = []
    s_nat_per_marker: list[int] = []
    s_exp_per_marker: list[int] = []
    for m_idx in valid_indices:
        _, mx_nm, my_nm, msize_nm = raw_markers[m_idx]
        s_nat = math.ceil((msize_nm + 2.0 * margin_nm) / grid_res)
        s_exp = _patch_size_px(msize_nm + 2.0 * margin_nm, grid_res, c)
        batch.append((marker_contours[m_idx], mx_nm - margin_nm, my_nm - margin_nm, s_nat))
        s_nat_per_marker.append(s_nat)
        s_exp_per_marker.append(s_exp)

    # Verify canvas covers every natural expanded patch (catches sizing bugs).
    canvas_right_nm = canvas_x0_nm + canvas_W * grid_res
    canvas_top_nm   = canvas_y0_nm + canvas_H * grid_res
    for m_idx, s_nat in zip(valid_indices, s_nat_per_marker):
        _, mx_nm, my_nm, msize_nm = raw_markers[m_idx]
        exp_right_nm = mx_nm - margin_nm + s_nat * grid_res
        exp_top_nm   = my_nm - margin_nm + s_nat * grid_res
        if canvas_right_nm < exp_right_nm - 0.5 or canvas_top_nm < exp_top_nm - 0.5:
            raise RuntimeError(
                f"Canvas [{canvas_x0_nm:.1f},{canvas_y0_nm:.1f} + "
                f"{canvas_W}×{canvas_H} px] does not cover expanded marker "
                f"right={exp_right_nm:.1f} nm / top={exp_top_nm:.1f} nm for "
                f"{oas_path.name} marker idx {m_idx}."
            )

    canvas = rasterize_canvas(
        batch,
        canvas_x0_nm=canvas_x0_nm,
        canvas_y0_nm=canvas_y0_nm,
        canvas_H=canvas_H,
        canvas_W=canvas_W,
        grid_res_nm_per_px=grid_res,
        truncation_px=truncation_px,
    )

    # Pad zeros around the canvas so that polygons reaching the canvas edge
    # see explicit exterior (0) rather than an abrupt data boundary.
    # canvas_x0_nm / canvas_y0_nm are updated so per-marker col0/row0
    # indices remain correct after the origin shift.
    _CANVAS_PAD = max(1, round(truncation_px))
    canvas = np.pad(canvas, pad_width=_CANVAS_PAD, mode="constant", constant_values=0.0)
    canvas_x0_nm -= _CANVAS_PAD * grid_res
    canvas_y0_nm -= _CANVAS_PAD * grid_res

    # ── Crop per-marker patches from canvas, then zero-pad to s_exp ──────────
    # Zero-padding is applied at the right and top (high indices).  Because
    # s_exp - s_nat <= c-1 < c = rf_erosion, the padded pixels are always
    # inside the rf-erosion border and are removed before the ADR crop.
    orig_markers_out:     list[dict[str, Any]] = []
    expanded_markers_out: list[dict[str, Any]] = []
    patches:              list[np.ndarray]     = []

    for m_idx, s_nat, s_exp in zip(valid_indices, s_nat_per_marker, s_exp_per_marker):
        _, mx_nm, my_nm, msize_nm = raw_markers[m_idx]
        exp_x_nm = mx_nm - margin_nm
        exp_y_nm = my_nm - margin_nm

        col0 = round((exp_x_nm - canvas_x0_nm) / grid_res)
        row0 = round((exp_y_nm - canvas_y0_nm) / grid_res)
        patch_nat = canvas[row0: row0 + s_nat, col0: col0 + s_nat]

        if s_exp > s_nat:
            patch = np.zeros((s_exp, s_exp), dtype=np.float32)
            patch[: patch_nat.shape[0], : patch_nat.shape[1]] = patch_nat
        else:
            patch = patch_nat

        orig_markers_out.append({"x_nm": mx_nm,    "y_nm": my_nm,    "size_nm": msize_nm})
        expanded_markers_out.append({"x_nm": exp_x_nm, "y_nm": exp_y_nm, "size_nm": s_exp * grid_res})
        patches.append(patch)

    patches_arr = np.stack(patches, axis=0)  # [N, S_exp, S_exp]
    return orig_markers_out, expanded_markers_out, patches_arr, dbu_um


# ─── Encode pass ─────────────────────────────────────────────────────────────


def encode_file(
    row: dict[str, Any],
    expanded_markers: list[dict[str, Any]],
    patches_np: np.ndarray,
    model: CurveCodec,
    device: torch.device,
    config: dict[str, Any],
    dna_dir: Path,
    checkpoint_path: Path,
) -> tuple[Path, float]:
    """Encode expanded-marker patches → ``.cdna``.

    Args:
        row: Catalog row dict.
        expanded_markers: Expanded marker dicts (``x_nm``, ``y_nm``,
            ``size_nm``), one per patch.
        patches_np: ``float32 [N, S_exp, S_exp]`` cSDF patches.
        model: CurveCodec in eval mode.
        device: Target device.
        config: YAML config dict.
        dna_dir: Output directory for ``.cdna`` files.
        checkpoint_path: Used to embed a fingerprint in ``meta.json``.

    Returns:
        ``(cdna_path, encode_ms)``
    """
    stem = Path(row["file"]).stem
    cdna_path = dna_dir / f"{stem}.cdna"

    x = torch.from_numpy(patches_np[:, None, :, :]).to(device)  # [N, 1, S_exp, S_exp]

    t0 = time.perf_counter()
    with torch.no_grad():
        dna_tensor = model.export_dna(x)  # [N, D, S_exp/c, S_exp/c]
    encode_ms = (time.perf_counter() - t0) * 1000.0

    save_cdna(
        path=cdna_path,
        dna=dna_tensor,
        config=config,
        scale_factors=model.quantizer.scale_factors,
        markers=expanded_markers,
        checkpoint_path=checkpoint_path,
    )
    log.debug("Encode: %s  N=%d  %.1f ms → %s", stem, len(expanded_markers), encode_ms, cdna_path)
    return cdna_path, encode_ms


# ─── Inspection layer writer ─────────────────────────────────────────────────


def _write_inspection_oas(
    path: Path,
    orig_hulls: list[list[tuple[float, float]]],
    recon_hulls: list[list[tuple[float, float]]],
    crop_x0_nm: float,
    crop_y0_nm: float,
    crop_size_nm: float,
    dbu_um: float,
) -> None:
    """Write a 4-layer inspection OASIS file for one marker.

    All geometry is already cropped to the valid area (the cropping marker).
    The XOR region is computed here via ``klayout.db.Region`` boolean XOR.

    Layers written:

    * ``_INSP_ORIG_LAYER``   (1)  — original mask polygons
    * ``_INSP_RECON_LAYER``  (2)  — reconstructed mask polygons
    * ``_INSP_XOR_LAYER``    (3)  — XOR symmetric difference
    * ``_INSP_MARKER_LAYER`` (10) — valid-marker boundary box

    Args:
        path: Destination ``.oas`` path (created or overwritten).
        orig_hulls: Original polygon hulls in nm, already cropped.
        recon_hulls: Reconstructed polygon hulls in nm, already cropped.
        crop_x0_nm: Left/bottom origin of the valid (cropping) marker in nm.
        crop_y0_nm: Left/bottom origin of the valid (cropping) marker in nm.
        crop_size_nm: Side length of the valid marker in nm.
        dbu_um: Layout DBU in µm — must match the source file.
    """
    import klayout.db as db

    scale = 1.0 / (dbu_um * 1000.0)  # nm → DBU

    def _hulls_to_region(hulls: list[list[tuple[float, float]]]) -> "db.Region":
        region = db.Region()
        for hull in hulls:
            pts = [db.Point(round(x * scale), round(y * scale)) for x, y in hull]
            if len(pts) >= 3:
                region.insert(db.Polygon(pts))
        return region

    orig_region  = _hulls_to_region(orig_hulls)
    recon_region = _hulls_to_region(recon_hulls)
    xor_region   = orig_region ^ recon_region

    layout = db.Layout()
    layout.dbu = dbu_um
    cell = layout.create_cell("TOP")

    def _insert_region(layer_num: int, region: "db.Region") -> None:
        li = layout.layer(layer_num, 0)
        for poly in region.each():
            cell.shapes(li).insert(poly)

    _insert_region(_INSP_ORIG_LAYER,  orig_region)
    _insert_region(_INSP_RECON_LAYER, recon_region)
    _insert_region(_INSP_XOR_LAYER,   xor_region)

    li_marker = layout.layer(_INSP_MARKER_LAYER, 0)
    cell.shapes(li_marker).insert(db.Box(
        round(crop_x0_nm * scale),
        round(crop_y0_nm * scale),
        round((crop_x0_nm + crop_size_nm) * scale),
        round((crop_y0_nm + crop_size_nm) * scale),
    ))

    path.parent.mkdir(parents=True, exist_ok=True)
    opts = db.SaveLayoutOptions()
    opts.format = "OASIS"
    layout.write(str(path), opts)
    log.debug("Inspection: wrote %s", path.name)


# ─── Decode + crop pass ──────────────────────────────────────────────────────


def _clip_hulls_to_crop(
    hulls: list[list[tuple[float, float]]],
    crop_x0_nm: float,
    crop_y0_nm: float,
    crop_size_nm: float,
    dbu_um: float,
) -> list[list[tuple[float, float]]]:
    """Geometrically clip polygon hulls to the crop bounding box.

    Uses ``klayout.db.Region`` boolean AND so the result is exact regardless
    of whether a polygon's centroid lies inside or outside the box.

    Args:
        hulls: Polygon hulls in nm.
        crop_x0_nm: Left/bottom origin of the crop box (nm).
        crop_y0_nm: Left/bottom origin of the crop box (nm).
        crop_size_nm: Side length of the (square) crop box (nm).
        dbu_um: Layout DBU in µm (nm → DBU conversion factor).

    Returns:
        Clipped hull list; polygons fully outside the box are dropped,
        polygons partially outside are trimmed to the box boundary.
    """
    import klayout.db as db

    if not hulls:
        return hulls

    scale  = 1.0 / (dbu_um * 1000.0)   # nm → DBU
    dbu_nm = dbu_um * 1000.0            # DBU → nm

    crop_region = db.Region(db.Box(
        round(crop_x0_nm * scale),
        round(crop_y0_nm * scale),
        round((crop_x0_nm + crop_size_nm) * scale),
        round((crop_y0_nm + crop_size_nm) * scale),
    ))

    hull_region = db.Region()
    for hull in hulls:
        pts = [db.Point(round(x * scale), round(y * scale)) for x, y in hull]
        if len(pts) >= 3:
            hull_region.insert(db.Polygon(pts))

    clipped = hull_region & crop_region

    result: list[list[tuple[float, float]]] = []
    for poly in clipped.each():
        pts_nm = [(pt.x * dbu_nm, pt.y * dbu_nm) for pt in poly.each_point_hull()]
        if pts_nm:
            result.append(pts_nm)
    return result


def _postprocess_per_marker(
    x_hat_np: np.ndarray,
    markers: list[dict[str, Any]],
    oas_path: Path,
    config: dict[str, Any],
    recon_dir: Path,
    orig_dir: Path,
    rf_erosion: int,
    dbu_um: float,
    stem: str,
    inspect_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Crop reconstructed cSDF and source polygons per marker; write ``.oas`` files.

    Shared post-processing used by both the ``.cdna`` (quantized) and
    in-memory (no-quant) eval paths.

    Args:
        x_hat_np: Decoder output ``[N, S_exp, S_exp]`` in ``float32``.
        markers: Expanded-marker dicts (``x_nm``, ``y_nm``, ``size_nm``),
            length ``N``.
        oas_path: Original OASIS file (used to re-extract polygons).
        config: YAML config dict (reads ``csdf.grid_res_nm_per_px`` and
            ``csdf.mask_layer``).
        recon_dir: Output dir for per-marker reconstructed ``.oas``.
        orig_dir: Output dir for per-marker extracted-original ``.oas``.
        rf_erosion: Receptive-field erosion in pixels (= compaction_ratio).
        dbu_um: Source layout DBU in µm.
        stem: Filename stem (without ``.oas``) used for per-marker names.
        inspect_dir: Optional 4-layer inspection ``.oas`` output dir.

    Returns:
        ``(recon_paths, orig_paths)``.
    """
    grid_res   = float(config["csdf"]["grid_res_nm_per_px"])
    mask_layer = int(config["csdf"]["mask_layer"])
    rf_nm      = rf_erosion * grid_res

    recon_paths: list[Path] = []
    orig_paths:  list[Path] = []

    for i, marker in enumerate(markers):
        # Expanded marker origin and size (stored in .cdna meta.json)
        exp_x_nm   = float(marker["x_nm"])
        exp_y_nm   = float(marker["y_nm"])
        exp_size_nm = float(marker["size_nm"])

        # Cropping marker: shrink expanded marker by rf_erosion on every side
        crop_x0_nm = exp_x_nm   + rf_nm
        crop_y0_nm = exp_y_nm   + rf_nm
        crop_size_nm = exp_size_nm - 2.0 * rf_nm

        if crop_size_nm <= 0:
            log.warning(
                "Cropping marker has non-positive size (%.1f nm) for %s m%d — skipping",
                crop_size_nm, stem, i,
            )
            continue

        # Run marching squares on the FULL expanded patch (not the rf-eroded
        # crop).  The margin pixels provide accurate SDF context for polygons
        # near the crop boundary, so the traced contour follows the real
        # polygon edge.  _clip_hulls_to_crop below then cuts the contour to
        # the crop box, producing the same straight edge as read_polygons_in_
        # region — eliminating the slit XOR that appeared when marching squares
        # ran on the cropped cSDF and closed the contour slightly inside the
        # crop boundary.
        csdf_full = x_hat_np[i]  # [S_exp, S_exp]

        contours = csdf_to_contours(
            csdf=csdf_full.astype(np.float32),
            origin_x_nm=exp_x_nm,
            origin_y_nm=exp_y_nm,
            grid_res_nm_per_px=grid_res,
        )

        recon_hulls: list[list[tuple[float, float]]] = []
        for contour in contours:
            pts: list[tuple[float, float]] = [seg.pts[0] for seg in contour.segments]
            if pts:
                recon_hulls.append(pts)

        # Clip both reconstructed and original polygons to the crop box so the
        # ADR comparison is bounded by the same region on both sides.
        # read_polygons_in_region uses centroid-in-box (no geometric clip), so
        # polygons whose bodies extend beyond the crop boundary are returned in
        # full; without an explicit clip they would inflate the XOR area.
        recon_hulls = _clip_hulls_to_crop(
            recon_hulls, crop_x0_nm, crop_y0_nm, crop_size_nm, dbu_um
        )

        recon_path = recon_dir / f"{stem}_m{i:04d}.oas"
        write_oas(recon_path, recon_hulls, mask_layer=mask_layer, dbu=dbu_um)
        recon_paths.append(recon_path)

        # Extract original polygons within the cropping marker region
        orig_hulls = read_polygons_in_region(
            oas_path=oas_path,
            mask_layer=mask_layer,
            x0_nm=crop_x0_nm,
            y0_nm=crop_y0_nm,
            x1_nm=crop_x0_nm + crop_size_nm,
            y1_nm=crop_y0_nm + crop_size_nm,
        )
        orig_hulls = _clip_hulls_to_crop(
            orig_hulls, crop_x0_nm, crop_y0_nm, crop_size_nm, dbu_um
        )
        orig_path = orig_dir / f"{stem}_m{i:04d}.oas"
        write_oas(orig_path, orig_hulls, mask_layer=mask_layer, dbu=dbu_um)
        orig_paths.append(orig_path)

        if inspect_dir is not None:
            try:
                _write_inspection_oas(
                    path=inspect_dir / f"{stem}_m{i:04d}_inspect.oas",
                    orig_hulls=orig_hulls,
                    recon_hulls=recon_hulls,
                    crop_x0_nm=crop_x0_nm,
                    crop_y0_nm=crop_y0_nm,
                    crop_size_nm=crop_size_nm,
                    dbu_um=dbu_um,
                )
            except Exception as exc:
                log.warning("Inspection write failed for %s m%d: %s", stem, i, exc)

    return recon_paths, orig_paths


def decode_file(
    cdna_path: Path,
    oas_path: Path,
    model: CurveCodec,
    device: torch.device,
    config: dict[str, Any],
    recon_dir: Path,
    orig_dir: Path,
    rf_erosion: int,
    dbu_um: float,
    inspect_dir: Path | None = None,
) -> tuple[list[Path], list[Path], float]:
    """Quantized eval path: ``.cdna`` → dequantize → decode → per-marker ``.oas``."""
    dna, meta = load_cdna(cdna_path)
    z = dequantize(dna, meta["scale_factors"]).to(device)  # [N, D, Sc, Sc]

    t0 = time.perf_counter()
    with torch.no_grad():
        x_hat = model.decode(z)  # [N, 1, S_exp, S_exp]
    decode_ms = (time.perf_counter() - t0) * 1000.0

    x_hat_np = x_hat.cpu().numpy()[:, 0, :, :]
    recon_paths, orig_paths = _postprocess_per_marker(
        x_hat_np=x_hat_np,
        markers=meta["markers"],
        oas_path=oas_path,
        config=config,
        recon_dir=recon_dir,
        orig_dir=orig_dir,
        rf_erosion=rf_erosion,
        dbu_um=dbu_um,
        stem=cdna_path.stem,
        inspect_dir=inspect_dir,
    )
    log.debug("Decode: %s  N=%d  %.1f ms", cdna_path.name, len(meta["markers"]), decode_ms)
    return recon_paths, orig_paths, decode_ms


def encode_decode_inmem(
    row: dict[str, Any],
    expanded_markers: list[dict[str, Any]],
    patches_np: np.ndarray,
    oas_path: Path,
    model: CurveCodec,
    device: torch.device,
    config: dict[str, Any],
    recon_dir: Path,
    orig_dir: Path,
    rf_erosion: int,
    dbu_um: float,
    inspect_dir: Path | None = None,
) -> tuple[list[Path], list[Path], float, float]:
    """No-quant eval path: encoder → decoder in memory, no ``.cdna`` roundtrip.

    Used when ``model.quantize=False`` to measure pure AE reconstruction
    quality without the destructive quantize-at-scale-1 step.  Encode and
    decode are run in a single ``model(x)`` forward pass; wall-clock is
    split 50/50 between the reported encode/decode times for parity with
    the quantized path.

    Returns:
        ``(recon_paths, orig_paths, encode_ms, decode_ms)``.
    """
    stem = Path(row["file"]).stem
    x = torch.from_numpy(patches_np[:, None, :, :]).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        x_hat = model(x)  # encoder → (bypassed quantizer) → decoder
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    x_hat_np = x_hat.cpu().numpy()[:, 0, :, :]
    recon_paths, orig_paths = _postprocess_per_marker(
        x_hat_np=x_hat_np,
        markers=expanded_markers,
        oas_path=oas_path,
        config=config,
        recon_dir=recon_dir,
        orig_dir=orig_dir,
        rf_erosion=rf_erosion,
        dbu_um=dbu_um,
        stem=stem,
        inspect_dir=inspect_dir,
    )
    half = elapsed_ms / 2.0
    log.debug("In-mem codec: %s  N=%d  %.1f ms", stem, len(expanded_markers), elapsed_ms)
    return recon_paths, orig_paths, half, half


# ─── Metrics pass ────────────────────────────────────────────────────────────


def compute_metrics(
    row: dict[str, Any],
    orig_markers: list[dict[str, Any]],
    oas_path: Path,
    cdna_path: Path | None,
    recon_paths: list[Path],
    orig_paths: list[Path],
    encode_ms: float,
    decode_ms: float,
    mask_layer: int,
) -> list[dict[str, Any]]:
    """Compute per-marker metrics from on-disk files.

    Args:
        row: Catalog row dict.
        orig_markers: Original (unexpanded) marker dicts.
        oas_path: Original OASIS file (for ``compression_ratio``).
        cdna_path: Encoded ``.cdna`` file.
        recon_paths: Per-marker reconstructed ``.oas`` paths (cropped).
        orig_paths: Per-marker extracted-original ``.oas`` paths (cropped).
        encode_ms: Encode wall-clock time for this file (ms).
        decode_ms: Decode wall-clock time for this file (ms).
        mask_layer: GDS layer of mask polygons.

    Returns:
        List of result dicts, one per marker.
    """
    cr = compression_ratio(oas_path, cdna_path) if cdna_path is not None else float("nan")
    n = len(orig_markers)
    enc_per = encode_ms / max(n, 1)
    dec_per = decode_ms / max(n, 1)

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
            "compression_ratio":     round(cr, 4) if not np.isnan(cr) else None,
            "area_difference_ratio": round(adr, 6) if not np.isnan(adr) else None,
            "encode_ms":             round(enc_per, 2),
            "decode_ms":             round(dec_per, 2),
        })
    return results


# ─── Main evaluation loop ────────────────────────────────────────────────────


def evaluate(
    checkpoint_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    split: str = "test",
    inspection: bool = False,
) -> None:
    """Run the full encode → decode → metrics pipeline on a dataset split.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint.
        config: Parsed YAML config dict.
        output_dir: Root output directory; subdirs ``dna/``,
                    ``reconstructed/``, and ``original/`` are created.
        device: PyTorch device.
        split: Dataset split — ``"train"``, ``"validation"``, or ``"test"``.
        inspection: If ``True``, write per-marker 4-layer inspection ``.oas``
            files to ``<output_dir>/inspection/``.
    """
    dna_dir   = output_dir / "dna"
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
    for d in (dna_dir, recon_dir, orig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg: dict[str, Any] = ckpt.get("config", config)
    arch = ckpt_cfg.get("model", {}).get("arch", "curve_codec_attn")
    _ARCH_MAP = {"curve_codec": CurveCodec, "curve_codec_v2": CurveCodecV2, "curve_codec_attn": CurveCodecAttn, "curve_codec_multiattn": CurveCodecMultiAttn}
    if arch not in _ARCH_MAP:
        raise ValueError(f"model.arch must be one of {list(_ARCH_MAP)}, got '{arch}'")
    model = _ARCH_MAP[arch](ckpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(
        "Loaded checkpoint: epoch=%d  best_val=%.6f",
        ckpt.get("epoch", -1),
        ckpt.get("best_val_loss", float("nan")),
    )
    if not getattr(model, "quantize", True):
        log.info(
            "model.quantize=False — bypassing .cdna roundtrip; "
            "running encoder→decoder in memory.  CR is reported as N/A."
        )

    # ── Read manifest for marker_margin_nm ───────────────────────────────────
    cache_dir = _PROJECT_ROOT / config["dataset"]["cache_dir"]
    manifest  = _load_manifest(cache_dir)
    margin_nm = float(manifest.get("marker_margin_nm", config["csdf"].get("marker_margin_nm", 0.0)))
    log.info("marker_margin_nm = %.1f nm (from manifest)", margin_nm)

    # ── Derived parameters ────────────────────────────────────────────────────
    grid_res     = float(config["csdf"]["grid_res_nm_per_px"])
    truncation_px = float(config["csdf"]["truncation_px"])
    mask_layer   = int(config["csdf"]["mask_layer"])
    marker_layer = int(config["csdf"]["marker_layer"])
    c            = int(ckpt_cfg["model"]["compaction_ratio"])
    rf_erosion   = _rf_erosion_px(ckpt_cfg)
    log.info("compaction_ratio=%d  rf_erosion=%d px  (%.1f nm)", c, rf_erosion, rf_erosion * grid_res)

    # ── Catalog rows ──────────────────────────────────────────────────────────
    split_rows = _load_split_rows(config, split)
    if not split_rows:
        log.error(
            "No rows found for split '%s' in catalog.csv.  "
            "Run dataset/ingest.py with --split %s first.",
            split, split,
        )
        return

    all_results: list[dict[str, Any]] = []

    for row_idx, row in enumerate(split_rows):
        stem = Path(row["file"]).stem
        oas_path = _PROJECT_ROOT / row["file"]
        if not oas_path.exists():
            log.warning("OASIS not found, skipping: %s", oas_path)
            continue

        log.info("[%d/%d] %s", row_idx + 1, len(split_rows), row["file"])

        # ── Step 1: Read + expand + rasterize ────────────────────────────────
        try:
            orig_markers, expanded_markers, patches_np, dbu_um = _rasterize_file_expanded(
                oas_path=oas_path,
                mask_layer=mask_layer,
                marker_layer=marker_layer,
                grid_res=grid_res,
                truncation_px=truncation_px,
                margin_nm=margin_nm,
                c=c,
            )
        except Exception as exc:
            log.error("Rasterization failed for %s: %s", row["file"], exc)
            continue

        if not orig_markers:
            log.warning("No markers found for %s — skipping", row["file"])
            continue

        log.debug("  markers=%d  S_exp=%d", len(orig_markers), patches_np.shape[1])

        cdna_path: Path | None = None
        if getattr(model, "quantize", True):
            # ── Quantized path: encode → .cdna → load → dequantize → decode ──
            try:
                cdna_path, encode_ms = encode_file(
                    row=row,
                    expanded_markers=expanded_markers,
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

            try:
                recon_paths, orig_paths, decode_ms = decode_file(
                    cdna_path=cdna_path,
                    oas_path=oas_path,
                    model=model,
                    device=device,
                    config=config,
                    recon_dir=recon_dir,
                    orig_dir=orig_dir,
                    rf_erosion=rf_erosion,
                    dbu_um=dbu_um,
                    inspect_dir=inspect_dir,
                )
            except Exception as exc:
                log.error("Decode failed for %s: %s", row["file"], exc)
                continue
        else:
            # ── No-quant path: encoder → decoder in memory (no .cdna roundtrip)
            try:
                recon_paths, orig_paths, encode_ms, decode_ms = encode_decode_inmem(
                    row=row,
                    expanded_markers=expanded_markers,
                    patches_np=patches_np,
                    oas_path=oas_path,
                    model=model,
                    device=device,
                    config=config,
                    recon_dir=recon_dir,
                    orig_dir=orig_dir,
                    rf_erosion=rf_erosion,
                    dbu_um=dbu_um,
                    inspect_dir=inspect_dir,
                )
            except Exception as exc:
                log.error("In-memory codec failed for %s: %s", row["file"], exc)
                continue

        if not recon_paths:
            log.warning("No valid decoded markers for %s — skipping metrics", row["file"])
            continue

        # ── Step 4: Metrics ───────────────────────────────────────────────────
        file_results = compute_metrics(
            row=row,
            orig_markers=orig_markers[: len(recon_paths)],
            oas_path=oas_path,
            cdna_path=cdna_path,
            recon_paths=recon_paths,
            orig_paths=orig_paths,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
            mask_layer=mask_layer,
        )
        all_results.extend(file_results)

        crs  = [r["compression_ratio"] for r in file_results if r.get("compression_ratio")]
        adrs = [r["area_difference_ratio"] for r in file_results
                if r.get("area_difference_ratio") is not None]
        log.info(
            "  markers=%d  CR=%.2f  ADR_mean=%.4f  enc=%.1f ms  dec=%.1f ms",
            len(orig_markers),
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
        n_files=len(split_rows),
    )
    log.info("Evaluation complete.  Output: %s", output_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

_DEFAULT_CFG = _PROJECT_ROOT / "train" / "config" / "baseline.yaml"


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core arguments ─────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to .pt checkpoint (required for codec/correction eval).",
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CFG,
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
    p.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    p.add_argument(
        "--inspection", action="store_true",
        help=(
            f"Write per-marker 4-layer inspection .oas files to <output-dir>/inspection/  "
            f"(orig=layer {_INSP_ORIG_LAYER}, recon={_INSP_RECON_LAYER}, "
            f"xor={_INSP_XOR_LAYER}, valid-marker={_INSP_MARKER_LAYER})"
        ),
    )

    # ── Mode flags ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--bypass", action="store_true",
        help="Rasterize→marching-squares only; no codec, no checkpoint required.",
    )
    p.add_argument(
        "--correction-factor", type=float, default=None,
        help=(
            "Enable iterative field-correction with step size α ∈ [0, 1].  "
            "Runs baseline + corrected passes side-by-side.  Requires --checkpoint."
        ),
    )
    p.add_argument(
        "--n-iterations", type=int, default=5,
        help="Number of correction iterations (default: 5; only used with --correction-factor).",
    )

    # ── Segment-distance arguments ─────────────────────────────────────────────
    p.add_argument(
        "--segment-distance", action="store_true",
        help=(
            "Run segment-centre EPE evaluation after the main eval pass, "
            "or standalone when combined with --results-dir / --correction-results-dir."
        ),
    )
    p.add_argument(
        "--results-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Root output dir from a prior codec/bypass run.  "
            "<DIR>/original/ and <DIR>/reconstructed/ are used automatically.  "
            "Implies --segment-distance."
        ),
    )
    p.add_argument(
        "--correction-results-dir", type=Path, default=None, metavar="DIR",
        help=(
            "Root output dir from a prior correction run.  "
            "Runs baseline and corrected passes side-by-side.  "
            "Implies --segment-distance."
        ),
    )
    p.add_argument(
        "--orig-dir", type=Path, default=None, metavar="DIR",
        help="Explicit original .oas directory (use together with --recon-dir).",
    )
    p.add_argument(
        "--recon-dir", type=Path, default=None, metavar="DIR",
        help="Explicit reconstructed .oas directory (use together with --orig-dir).",
    )
    p.add_argument(
        "--mask-layer", type=int, default=1,
        help="GDS layer number of mask polygons for segment-distance (default: 1).",
    )
    p.add_argument(
        "--spec", type=float, default=None, metavar="NM",
        help=(
            "EPE spec in nm.  Segments exceeding this distance are written as square "
            "markers (layer 100) to <output>_violations/ alongside the summary JSON."
        ),
    )
    p.add_argument(
        "--violation-marker-size-nm", type=float, default=2.0, metavar="NM",
        help="Side length of violation marker squares in nm (default: 2.0).",
    )
    p.add_argument(
        "--segment-distance-output", type=Path, default=None,
        help=(
            "Output JSON path for segment-distance summary.  "
            "Defaults to <output-dir>/segment_distance_summary.json."
        ),
    )

    return p


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_segdist_correction(
    root: Path,
    mask_layer: int,
    spec_nm: float | None,
    violation_marker_size: float,
    evaluate_segment_distance: Any,
    _print_summary: Any,
    _print_comparison: Any,
) -> None:
    orig_dir       = root / "original"
    base_recon_dir = root / "baseline"  / "reconstructed"
    corr_recon_dir = root / "corrected" / "reconstructed"
    for d, label in (
        (orig_dir,       "original"),
        (base_recon_dir, "baseline/reconstructed"),
        (corr_recon_dir, "corrected/reconstructed"),
    ):
        if not d.is_dir():
            log.error("%s does not exist or is not a directory: %s", label, d)
            sys.exit(1)
    base_json = root / "segment_distance_baseline.json"
    corr_json = root / "segment_distance_corrected.json"
    log.info("── Baseline segment-distance pass ──")
    base_summary = evaluate_segment_distance(
        orig_dir=orig_dir, recon_dir=base_recon_dir,
        mask_layer=mask_layer, output_json=base_json,
        spec_nm=spec_nm, violation_marker_size_nm=violation_marker_size,
    )
    log.info("── Corrected segment-distance pass ──")
    corr_summary = evaluate_segment_distance(
        orig_dir=orig_dir, recon_dir=corr_recon_dir,
        mask_layer=mask_layer, output_json=corr_json,
        spec_nm=spec_nm, violation_marker_size_nm=violation_marker_size,
    )
    if not base_summary or not corr_summary:
        log.warning("No segment-distance results produced.")
        return
    _print_summary("Baseline reconstruction",  base_summary, base_json)
    _print_summary("Corrected reconstruction", corr_summary, corr_json)
    _print_comparison(base_summary, corr_summary)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _make_parser().parse_args()

    run_segdist = (
        args.segment_distance
        or args.results_dir is not None
        or args.correction_results_dir is not None
        or (args.orig_dir is not None and args.recon_dir is not None)
    )
    run_eval = args.checkpoint is not None or args.bypass

    if not run_eval and not run_segdist:
        log.error(
            "Nothing to do.  Provide --checkpoint (codec/correction eval), "
            "--bypass (bypass eval), or a --results-dir / --correction-results-dir "
            "(standalone segment-distance)."
        )
        sys.exit(1)

    # ── Main eval pass ─────────────────────────────────────────────────────────
    if run_eval:
        config = _load_config(args.config)
        device = _resolve_device(args.device)
        log.info("Device: %s", device)

        if args.bypass:
            from eval.evaluate_bypass import evaluate_bypass
            evaluate_bypass(
                config=config,
                output_dir=args.output_dir,
                split=args.split,
                inspection=args.inspection,
            )
        elif args.correction_factor is not None:
            if not args.checkpoint.exists():
                log.error("Checkpoint not found: %s", args.checkpoint)
                sys.exit(1)
            from eval.evaluate_correction import evaluate_correction
            evaluate_correction(
                checkpoint_path=args.checkpoint,
                config=config,
                output_dir=args.output_dir,
                device=device,
                correction_factor=args.correction_factor,
                n_iterations=args.n_iterations,
                split=args.split,
            )
        else:
            if not args.checkpoint.exists():
                log.error("Checkpoint not found: %s", args.checkpoint)
                sys.exit(1)
            evaluate(
                checkpoint_path=args.checkpoint,
                config=config,
                output_dir=args.output_dir,
                device=device,
                split=args.split,
                inspection=args.inspection,
            )

    # ── Segment-distance pass ──────────────────────────────────────────────────
    if run_segdist:
        from eval.evaluate_segment_distance import (
            evaluate_segment_distance,
            _print_comparison,
            _print_summary,
        )
        mask_layer            = args.mask_layer
        spec_nm               = args.spec
        violation_marker_size = args.violation_marker_size_nm

        # Determine effective source dirs, falling back to the eval output when
        # --segment-distance was combined with a main eval pass.
        corr_root  = args.correction_results_dir
        plain_root = args.results_dir
        sd_orig_dir  = args.orig_dir
        sd_recon_dir = args.recon_dir

        if args.segment_distance and run_eval and corr_root is None and plain_root is None and sd_orig_dir is None:
            if args.correction_factor is not None:
                corr_root = args.output_dir
            else:
                plain_root = args.output_dir

        if corr_root is not None:
            _run_segdist_correction(
                corr_root, mask_layer, spec_nm, violation_marker_size,
                evaluate_segment_distance, _print_summary, _print_comparison,
            )
        else:
            if plain_root is not None:
                sd_orig_dir  = plain_root / "original"
                sd_recon_dir = plain_root / "reconstructed"
                default_out  = plain_root / "segment_distance_summary.json"
            elif sd_orig_dir is not None and sd_recon_dir is not None:
                default_out = sd_recon_dir.parent / "segment_distance_summary.json"
            else:
                log.error(
                    "Provide --results-dir, --correction-results-dir, "
                    "or both --orig-dir and --recon-dir for segment-distance."
                )
                sys.exit(1)
            for d, label in ((sd_orig_dir, "orig-dir"), (sd_recon_dir, "recon-dir")):
                if not d.is_dir():
                    log.error("%s does not exist or is not a directory: %s", label, d)
                    sys.exit(1)
            output_json = args.segment_distance_output or default_out
            summary = evaluate_segment_distance(
                orig_dir=sd_orig_dir, recon_dir=sd_recon_dir,
                mask_layer=mask_layer, output_json=output_json,
                spec_nm=spec_nm, violation_marker_size_nm=violation_marker_size,
            )
            if summary:
                _print_summary("Segment-centre distance to original boundary", summary, output_json)
            else:
                log.warning("No results produced.")


if __name__ == "__main__":
    main()

"""Python dispatch layer for the cSDF rasterizer.

Selects between the CPU (OpenMP) and GPU (CUDA) rasterizer backends at
runtime and exposes a unified Python API for converting PWCL polygon +
marker layer geometry into truncated, normalised cSDF patches of shape
``float32 [S, S]``.

Coordinate conventions
──────────────────────
All geometry coordinates are in **physical nanometres** (nm).  Physical
units must be explicit at every call site; never pass raw DBU integers.

cSDF value convention (must never deviate — see csdf.md)
─────────────────────────────────────────────────────────
  1.0  fully inside  (SDF ≤ −t)
  0.5  on boundary   (SDF = 0)  — correct marching-squares iso-level
  0.0  fully outside (SDF ≥ +t)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ─── PWCL geometry types ──────────────────────────────────────────────────────


class SegmentType(IntEnum):
    """PWCL edge-segment types.  Values must match the C++ enum in rasterizer.h."""

    LINE = 0  # 2 control points: [start, end]
    ARC = 1  # 3 control points: [start, on-arc-mid, end]  (3-point arc)
    BEZIER2 = 2  # 3 control points: [p0, ctrl, p2]  (quadratic Bézier)
    BEZIER3 = 3  # 4 control points: [p0, c1, c2, p3]  (cubic Bézier)


@dataclass
class PwclSegment:
    """One directed edge segment in a PWCL contour.

    Args:
        type: Segment kind (LINE / ARC / BEZIER2 / BEZIER3).
        pts: Control-point list in nanometres.  Length must match the type:
             LINE=2, ARC=3, BEZIER2=3, BEZIER3=4.
    """

    type: SegmentType
    pts: list[tuple[float, float]]


@dataclass
class PwclContour:
    """Closed PWCL contour (outer boundary or interior hole).

    Winding convention (enforced by ingest.py):
        is_hole=False → outer boundary, CCW (positive area)
        is_hole=True  → interior hole,  CW  (negative area)

    Args:
        segments: Ordered directed edge segments that close the contour.
        is_hole:  True for interior holes.
    """

    segments: list[PwclSegment] = field(default_factory=list)
    is_hole: bool = False


# ─── Extension import ─────────────────────────────────────────────────────────

try:
    from csdf import csdf_ext as _ext  # compiled pybind11 module

    _HAS_EXT = True
except ImportError:
    _ext = None
    _HAS_EXT = False
    log.warning(
        "csdf_ext C++ extension not found. "
        "Run 'cmake .. -DWITH_CUDA=OFF && make -j$(nproc)' from the build/ "
        "directory before calling rasterize_patch()."
    )

# ─── Patch-size derivation ────────────────────────────────────────────────────


def derive_patch_size_px(
    marker_size_nm: float,
    grid_res_nm_per_px: float,
    compaction_ratio: int,
) -> int:
    """Derive the patch side-length *S* in pixels from a square marker.

    Implements the formula from the spec::

        S = ceil(marker_size_nm / grid_res_nm_per_px)
        S = S + (c - S % c) % c   # round up to next multiple of c

    Args:
        marker_size_nm: Side length of the square marker shape (nm).
        grid_res_nm_per_px: Physical size of one pixel (nm/px).
        compaction_ratio: Spatial downsampling factor *c*; must be a power
            of 2 and must divide the returned *S*.

    Returns:
        S: patch side length in pixels, rounded up to the next multiple of
        ``compaction_ratio``.

    Raises:
        ValueError: If ``compaction_ratio`` is not a positive power of 2.
    """
    if compaction_ratio <= 0 or (compaction_ratio & (compaction_ratio - 1)) != 0:
        raise ValueError(
            f"compaction_ratio must be a positive power of 2, got {compaction_ratio}"
        )
    s = math.ceil(marker_size_nm / grid_res_nm_per_px)
    remainder = s % compaction_ratio
    if remainder != 0:
        s += compaction_ratio - remainder
    return s


# ─── Numpy serialisation helpers ──────────────────────────────────────────────


def _contours_to_arrays(
    contours: list[PwclContour],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of PwclContour objects into flat numpy arrays for C++.

    Args:
        contours: List of closed PWCL contours.

    Returns:
        Tuple of four arrays:

        - ``seg_types``   – ``int32[N]``       segment type per segment
        - ``seg_pts``     – ``float64[N,4,2]`` control points, padded with zeros
        - ``contour_ids`` – ``int32[N]``       contour index per segment (0-based)
        - ``is_hole``     – ``bool[C]``        is_hole flag per contour
    """
    types_list: list[int] = []
    pts_list: list[list] = []
    ids_list: list[int] = []
    holes: list[bool] = []

    for c_idx, contour in enumerate(contours):
        holes.append(contour.is_hole)
        for seg in contour.segments:
            types_list.append(int(seg.type))
            ids_list.append(c_idx)
            # Pad or truncate to exactly 4 control points.
            padded: list[tuple[float, float]] = list(seg.pts)
            while len(padded) < 4:
                padded.append((0.0, 0.0))
            pts_list.append(padded[:4])

    n = len(types_list)
    seg_types = np.array(types_list, dtype=np.int32)
    seg_pts = (
        np.array(pts_list, dtype=np.float64)  # [N, 4, 2]
        if n > 0
        else np.zeros((0, 4, 2), dtype=np.float64)
    )
    contour_ids = np.array(ids_list, dtype=np.int32)
    is_hole = np.array(holes, dtype=bool)

    return seg_types, seg_pts, contour_ids, is_hole


# ─── Public API ───────────────────────────────────────────────────────────────


def rasterize_canvas(
    markers: "list[tuple[list[PwclContour], float, float, int]]",
    canvas_x0_nm: float,
    canvas_y0_nm: float,
    canvas_H: int,
    canvas_W: int,
    grid_res_nm_per_px: float,
    truncation_px: float,
) -> "NDArray[np.float32]":
    """Batch-rasterize all marker regions into a single float32 [H, W] canvas.

    Builds all numpy arrays in one pass and calls C++ once — eliminates
    per-marker Python→C++ call overhead.

    Args:
        markers: List of (contours, origin_x_nm, origin_y_nm, patch_size_px).
        canvas_x0_nm: Canvas bottom-left x (nm).
        canvas_y0_nm: Canvas bottom-left y (nm).
        canvas_H: Canvas height in pixels.
        canvas_W: Canvas width in pixels.
        grid_res_nm_per_px: Physical size of one pixel (nm/px).
        truncation_px: SDF truncation half-width in pixels.

    Returns:
        float32 ndarray of shape [H, W].
    """
    if not _HAS_EXT:
        raise RuntimeError(
            "csdf_ext C++ extension is not available.\n"
            "Build it with:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DWITH_CUDA=OFF && make -j$(nproc)\n"
        )

    # ── Pack all contours into flat arrays ────────────────────────────────────
    all_types: list[int] = []
    all_pts:   list[list] = []
    all_cids:  list[int] = []
    all_holes: list[bool] = []

    marker_c_start: list[int] = []
    marker_c_end:   list[int] = []
    marker_ox:      list[float] = []
    marker_oy:      list[float] = []
    marker_sizes:   list[int] = []

    global_contour_idx = 0

    for contours, ox, oy, patch_px in markers:
        c_start = global_contour_idx
        for contour in contours:
            all_holes.append(contour.is_hole)
            for seg in contour.segments:
                all_types.append(int(seg.type))
                all_cids.append(global_contour_idx)
                padded = list(seg.pts)
                while len(padded) < 4:
                    padded.append((0.0, 0.0))
                all_pts.append(padded[:4])
            global_contour_idx += 1
        marker_c_start.append(c_start)
        marker_c_end.append(global_contour_idx)
        marker_ox.append(ox)
        marker_oy.append(oy)
        marker_sizes.append(patch_px)

    N = len(all_types)
    C = global_contour_idx
    M = len(markers)

    seg_types   = np.array(all_types, dtype=np.int32)
    seg_pts     = np.array(all_pts,   dtype=np.float64) if N > 0 else np.zeros((0, 4, 2), dtype=np.float64)
    contour_ids = np.array(all_cids,  dtype=np.int32)
    is_hole     = np.array(all_holes, dtype=bool)
    c_start_arr = np.array(marker_c_start, dtype=np.int32)
    c_end_arr   = np.array(marker_c_end,   dtype=np.int32)
    ox_arr      = np.array(marker_ox,      dtype=np.float64)
    oy_arr      = np.array(marker_oy,      dtype=np.float64)
    sz_arr      = np.array(marker_sizes,   dtype=np.int32)

    return _ext.rasterize_csdf_canvas(
        seg_types, seg_pts, contour_ids, is_hole,
        c_start_arr, c_end_arr, ox_arr, oy_arr, sz_arr,
        float(canvas_x0_nm), float(canvas_y0_nm),
        int(canvas_H), int(canvas_W),
        float(grid_res_nm_per_px), float(truncation_px),
    )


def rasterize_patch(
    contours: list[PwclContour],
    origin_x_nm: float,
    origin_y_nm: float,
    patch_size_px: int,
    grid_res_nm_per_px: float,
    truncation_px: float,
    device: str = "cpu",
) -> "NDArray[np.float32]":
    """Rasterize PWCL contours into a float32 cSDF patch.

    Implements the three-step rasterization pipeline:

    1. **SDF**          — signed Euclidean distance to the nearest PWCL edge
                          (negative inside, positive outside).
    2. **Truncation**   — clamp to ``[−t, +t]`` where ``t = truncation_px``.
    3. **Normalisation** — ``csdf = (t − clamp(sdf, −t, +t)) / (2·t)`` → ``[0, 1]``.

    Note: Because the raw SDF is negative inside, the normalisation formula is
    ``(t − clamped) / (2t)`` (not ``(clamped + t) / (2t)``), which correctly
    maps deeply-inside pixels to **1.0** and deeply-outside pixels to **0.0**.

    Args:
        contours: Closed PWCL contours (outer CCW boundaries + optional CW
            holes).  All coordinates in nanometres.
        origin_x_nm: x-coordinate of the patch bottom-left corner (nm).
        origin_y_nm: y-coordinate of the patch bottom-left corner (nm).
        patch_size_px: Side length *S* of the output patch in pixels.
        grid_res_nm_per_px: Physical size of one pixel (nm/px).
        truncation_px: Half-width of the SDF truncation band in pixels.
        device: ``"cpu"`` (default) — OpenMP CPU path.  ``"cuda"`` raises
            ``NotImplementedError`` until the GPU path is implemented.

    Returns:
        ``float32`` ndarray of shape ``[S, S]``.  Row 0 = ``y_min`` (bottom
        of the marker box).

        Value map:

        ====  ==========================================================
        1.0   Fully inside the polygon  (SDF ≤ −t_nm)
        0.5   Exactly on the boundary   (SDF = 0)
        0.0   Fully outside             (SDF ≥ +t_nm)
        ====  ==========================================================

    Raises:
        NotImplementedError: If ``device="cuda"`` is requested.
        RuntimeError: If the C++ extension has not been compiled.
    """
    if device == "cuda":
        raise NotImplementedError("The CUDA rasterizer path is not yet implemented.")
    if device != "cpu":
        raise ValueError(f"Unknown device '{device}'. Choose 'cpu' or 'cuda'.")
    if not _HAS_EXT:
        raise RuntimeError(
            "csdf_ext C++ extension is not available.\n"
            "Build it with:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DWITH_CUDA=OFF && make -j$(nproc)\n"
        )

    seg_types, seg_pts, contour_ids, is_hole = _contours_to_arrays(contours)

    patch: NDArray[np.float32] = _ext.rasterize_csdf_cpu(
        seg_types,
        seg_pts,
        contour_ids,
        is_hole,
        float(origin_x_nm),
        float(origin_y_nm),
        int(patch_size_px),
        float(grid_res_nm_per_px),
        float(truncation_px),
    )
    return patch

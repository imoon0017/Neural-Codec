"""Visualize cSDF patches for all PWCL segment types.

Renders six geometry examples — LINE square, ARC circle, BEZIER2 lens,
BEZIER3 rounded rectangle, polygon with hole, and a composite shape —
each shown as a heatmap with the iso=0.5 contour overlaid.

Usage:
    python scripts/visualize_geometries.py
    python scripts/visualize_geometries.py --output csdf_geometries.png
    python scripts/visualize_geometries.py --patch-size 128 --trunc 8
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csdf.csdf_utils import (
    PwclContour,
    PwclSegment,
    SegmentType,
    rasterize_patch,
)

log = logging.getLogger(__name__)

# ─── Geometry builders ────────────────────────────────────────────────────────


def _pt(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    th = math.radians(angle_deg)
    return (cx + r * math.cos(th), cy + r * math.sin(th))


def make_square(
    cx_nm: float, cy_nm: float, half_nm: float, is_hole: bool = False
) -> PwclContour:
    """Axis-aligned square with four LINE segments (CCW outer / CW hole)."""
    if not is_hole:
        corners = [
            (cx_nm - half_nm, cy_nm - half_nm),
            (cx_nm + half_nm, cy_nm - half_nm),
            (cx_nm + half_nm, cy_nm + half_nm),
            (cx_nm - half_nm, cy_nm + half_nm),
        ]
    else:
        corners = [
            (cx_nm - half_nm, cy_nm - half_nm),
            (cx_nm - half_nm, cy_nm + half_nm),
            (cx_nm + half_nm, cy_nm + half_nm),
            (cx_nm + half_nm, cy_nm - half_nm),
        ]
    segments = [
        PwclSegment(SegmentType.LINE, [corners[i], corners[(i + 1) % 4]])
        for i in range(4)
    ]
    return PwclContour(segments=segments, is_hole=is_hole)


def make_circle(cx_nm: float, cy_nm: float, r_nm: float) -> PwclContour:
    """Full circle approximated by four 90° CCW ARC segments."""
    def arc_seg(a0: float, a1: float) -> PwclSegment:
        return PwclSegment(
            SegmentType.ARC,
            [_pt(cx_nm, cy_nm, r_nm, a0),
             _pt(cx_nm, cy_nm, r_nm, (a0 + a1) / 2),
             _pt(cx_nm, cy_nm, r_nm, a1)],
        )

    return PwclContour(
        segments=[arc_seg(0, 90), arc_seg(90, 180), arc_seg(180, 270), arc_seg(270, 360)],
        is_hole=False,
    )


def make_lens(cx_nm: float, cy_nm: float, half_nm: float) -> PwclContour:
    """Lens shape: two quadratic Bézier arcs (BEZIER2)."""
    left  = (cx_nm - half_nm, cy_nm)
    right = (cx_nm + half_nm, cy_nm)
    bot   = (cx_nm, cy_nm - half_nm)
    top   = (cx_nm, cy_nm + half_nm)
    return PwclContour(
        segments=[
            PwclSegment(SegmentType.BEZIER2, [left, bot, right]),
            PwclSegment(SegmentType.BEZIER2, [right, top, left]),
        ],
        is_hole=False,
    )


def make_rounded_rect(cx_nm: float, cy_nm: float, half_nm: float) -> PwclContour:
    """Rounded-rectangle-like shape: two cubic Bézier arcs (BEZIER3)."""
    h = half_nm
    inset = half_nm * 0.25
    return PwclContour(
        segments=[
            PwclSegment(
                SegmentType.BEZIER3,
                [(cx_nm - h, cy_nm),
                 (cx_nm - h + inset, cy_nm - h),
                 (cx_nm + h - inset, cy_nm - h),
                 (cx_nm + h, cy_nm)],
            ),
            PwclSegment(
                SegmentType.BEZIER3,
                [(cx_nm + h, cy_nm),
                 (cx_nm + h - inset, cy_nm + h),
                 (cx_nm - h + inset, cy_nm + h),
                 (cx_nm - h, cy_nm)],
            ),
        ],
        is_hole=False,
    )


def make_donut(
    cx_nm: float, cy_nm: float, outer_half_nm: float, inner_half_nm: float
) -> list[PwclContour]:
    """Square with a square hole (outer CCW + inner CW)."""
    return [
        make_square(cx_nm, cy_nm, outer_half_nm, is_hole=False),
        make_square(cx_nm, cy_nm, inner_half_nm, is_hole=True),
    ]


def make_composite(cx_nm: float, cy_nm: float, scale: float) -> list[PwclContour]:
    """Square outer boundary with a circular hole cut from its centre."""
    outer = make_square(cx_nm, cy_nm, scale * 0.45, is_hole=False)
    # CW circle = four 90° arcs in reverse order
    r = scale * 0.20
    def arc_seg_cw(a0: float, a1: float) -> PwclSegment:
        return PwclSegment(
            SegmentType.ARC,
            [_pt(cx_nm, cy_nm, r, a0),
             _pt(cx_nm, cy_nm, r, (a0 + a1) / 2),
             _pt(cx_nm, cy_nm, r, a1)],
        )

    hole = PwclContour(
        segments=[arc_seg_cw(360, 270), arc_seg_cw(270, 180),
                  arc_seg_cw(180, 90),  arc_seg_cw(90, 0)],
        is_hole=True,
    )
    return [outer, hole]


# ─── Scene definitions ────────────────────────────────────────────────────────


class Scene(NamedTuple):
    title: str
    contours: list[PwclContour]


def build_scenes(S: int) -> list[Scene]:
    cx = cy = S / 2.0
    scale = S * 0.35

    return [
        Scene("LINE — square",        [make_square(cx, cy, scale)]),
        Scene("ARC — circle",         [make_circle(cx, cy, scale)]),
        Scene("BEZIER2 — lens",       [make_lens(cx, cy, scale)]),
        Scene("BEZIER3 — rounded rect", [make_rounded_rect(cx, cy, scale)]),
        Scene("Hole — square donut",  make_donut(cx, cy, scale, scale * 0.4)),
        Scene("Composite — sq + arc hole", make_composite(cx, cy, S)),
    ]


# ─── Rendering ────────────────────────────────────────────────────────────────


def render(
    scenes: list[Scene],
    patch_size_px: int,
    trunc_px: float,
    grid_res: float,
    output: Path | None,
) -> None:
    n = len(scenes)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_flat = np.array(axes).ravel()

    cmap = "RdBu"

    for ax, scene in zip(axes_flat, scenes):
        patch = rasterize_patch(
            scene.contours,
            origin_x_nm=0.0,
            origin_y_nm=0.0,
            patch_size_px=patch_size_px,
            grid_res_nm_per_px=grid_res,
            truncation_px=trunc_px,
        )
        im = ax.imshow(patch, vmin=0.0, vmax=1.0, cmap=cmap, origin="lower")
        ax.contour(patch, levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(scene.title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"cSDF patches  |  S={patch_size_px}px  grid={grid_res}nm/px  trunc={trunc_px}px\n"
        "Red=inside (1.0)  Blue=outside (0.0)  Black line=iso 0.5 boundary",
        fontsize=11,
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        log.info("Saved to %s", output)
        print(f"Saved → {output}")
    else:
        plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Save figure to this path instead of showing interactively (e.g. out.png)")
    p.add_argument("--patch-size", type=int, default=128, metavar="S",
                   help="Patch side length in pixels (default: 128)")
    p.add_argument("--trunc", type=float, default=8.0, metavar="T",
                   help="cSDF truncation half-width in pixels (default: 8.0)")
    p.add_argument("--grid-res", type=float, default=1.0, metavar="G",
                   help="Grid resolution nm/px (default: 1.0)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    scenes = build_scenes(args.patch_size)
    render(scenes, args.patch_size, args.trunc, args.grid_res, args.output)


if __name__ == "__main__":
    main()

"""Quickly display a .npy file as an image.

Usage:
    python scripts/show_npy.py file.npy
    python scripts/show_npy.py file.npy --output out.png
    python scripts/show_npy.py file.npy --index 3        # for [N, H, W] arrays
    python scripts/show_npy.py file.npy --cmap gray
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npy_file", type=Path)
    p.add_argument("--index", "-i", type=int, default=None,
                   help="Slice index for [N, H, W] arrays (default: 0)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Save to file instead of showing interactively")
    p.add_argument("--cmap", default="RdBu",
                   help="Matplotlib colormap (default: RdBu)")
    p.add_argument("--no-contour", action="store_true",
                   help="Skip iso=0.5 contour overlay")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.npy_file.exists():
        print(f"Error: {args.npy_file} not found", file=sys.stderr)
        sys.exit(1)

    arr = np.load(args.npy_file)
    print(f"shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

    # Handle different array shapes
    if arr.ndim == 2:
        image = arr
    elif arr.ndim == 3:
        idx = args.index if args.index is not None else 0
        # Support [N, H, W] and [H, W, C]
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            # Likely [N, H, W] — slice on first axis
            image = arr[idx]
            print(f"Showing slice [{idx}] of {arr.shape[0]}")
        else:
            # Likely [H, W, C]
            image = arr
    elif arr.ndim == 4:
        # [N, C, H, W] or [N, H, W, C]
        idx = args.index if args.index is not None else 0
        image = arr[idx, 0] if arr.shape[1] < arr.shape[2] else arr[idx, :, :, 0]
        print(f"Showing slice [{idx}, 0]")
    else:
        print(f"Error: cannot display array with {arr.ndim} dimensions", file=sys.stderr)
        sys.exit(1)

    vmin = args.vmin if args.vmin is not None else image.min()
    vmax = args.vmax if args.vmax is not None else image.max()

    fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / image.shape[1]))
    im = ax.imshow(image, cmap=args.cmap, origin="lower", vmin=vmin, vmax=vmax)
    if not args.no_contour and 0.0 <= 0.5 <= 1.0 and vmin <= 0.5 <= vmax:
        ax.contour(image, levels=[0.5], colors="black", linewidths=0.7)
    ax.set_title(f"{args.npy_file.name}  {image.shape}", fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

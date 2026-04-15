"""Visualize training results from a CurveCodec checkpoint.

Produces up to four figures:

1. **Loss curves** — train and validation loss over epochs (from metrics.jsonl)
2. **Model summary** — parameter counts per sub-module, quantizer scale factors
3. **Reconstructions** — side-by-side input vs reconstructed cSDF patches
4. **Error map** — per-pixel absolute difference for each sample

Usage::

    python scripts/visualize_training.py
    python scripts/visualize_training.py --checkpoint checkpoints/baseline_v1/best.pt
    python scripts/visualize_training.py --checkpoint best.pt --samples 8 --no-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


# ─── Helpers ─────────────────────────────────────────────────────────────────


def load_checkpoint(ckpt_path: Path) -> dict:
    return torch.load(ckpt_path, map_location="cpu")


def load_metrics(ckpt_dir: Path) -> list[dict]:
    path = ckpt_dir / "metrics.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── Figure 1: Loss curves ────────────────────────────────────────────────────


def plot_loss_curves(metrics: list[dict], save_path: Path | None = None) -> None:
    if not metrics:
        print("No metrics.jsonl found — skipping loss curves.")
        return

    epochs = [m["epoch"] + 1 for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    val_epochs = [m["epoch"] + 1 for m in metrics if "val_loss" in m]
    val_losses = [m["val_loss"] for m in metrics if "val_loss" in m]
    lrs = [m["lr"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Training curves", fontsize=13)

    ax1.plot(epochs, train_losses, label="train", color="steelblue", linewidth=1.5)
    if val_losses:
        ax1.plot(val_epochs, val_losses, label="val", color="tomato",
                 linewidth=1.5, marker="o", markersize=3)
        best_idx = int(np.argmin(val_losses))
        best_epoch = val_epochs[best_idx]
        best_val = val_losses[best_idx]
        ax1.plot(best_epoch, best_val, marker="*", markersize=10,
                 color="tomato", zorder=5, label=f"best val={best_val:.5f} (ep {best_epoch})")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(epochs, lrs, color="seagreen", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _show_or_save(fig, save_path, "loss_curves.png")


# ─── Figure 2: Model summary ──────────────────────────────────────────────────


def plot_model_summary(ckpt: dict, save_path: Path | None = None) -> None:
    from codec.model import CurveCodec

    cfg = ckpt["config"]
    model = CurveCodec(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Parameter counts
    modules = {
        "encoder": model.encoder,
        "quantizer": model.quantizer,
        "decoder": model.decoder,
    }
    names = list(modules.keys())
    counts = [sum(p.numel() for p in m.parameters()) for m in modules.values()]
    total = sum(counts)

    # Quantizer scale factors
    scales = model.quantizer.scale_factors.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Model summary  —  {total:,} params  |  "
        f"epoch {ckpt['epoch']}  |  best val {ckpt['best_val_loss']:.6f}",
        fontsize=12,
    )

    # Param pie
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        counts, labels=names, autopct="%1.1f%%",
        colors=["steelblue", "orange", "tomato"], startangle=90,
    )
    ax.set_title(f"Parameter distribution\n({total:,} total)")

    # Scale factors histogram
    ax = axes[1]
    ax.hist(scales, bins=min(50, len(scales)), color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Scale factor s_d (exp(log_scale))")
    ax.set_ylabel("Count")
    ax.set_title(f"Quantizer scale factors  (D={len(scales)})\n"
                 f"min={scales.min():.4f}  max={scales.max():.4f}  mean={scales.mean():.4f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _show_or_save(fig, save_path, "model_summary.png")


# ─── Figure 3 & 4: Reconstructions ───────────────────────────────────────────


def plot_reconstructions(
    ckpt: dict,
    n_samples: int = 4,
    save_path: Path | None = None,
) -> None:
    from codec.model import CurveCodec
    from dataset.dataset import make_dataloaders

    cfg = ckpt["config"]
    model = CurveCodec(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    try:
        loaders = make_dataloaders(cfg, splits=["validation"], num_workers=0)
        val_loader = loaders["validation"]
    except Exception as exc:
        print(f"Could not load dataset ({exc}) — skipping reconstructions.")
        return

    # Collect samples
    patches: list[np.ndarray] = []
    recons: list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["csdf"]           # [B, 1, S, S]
            x_hat = model(x)
            for i in range(x.shape[0]):
                patches.append(x[i, 0].numpy())
                recons.append(x_hat[i, 0].numpy())
                if len(patches) >= n_samples:
                    break
        if len(patches) >= n_samples:
            pass  # done

    n = min(n_samples, len(patches))
    if n == 0:
        print("No validation samples — skipping reconstructions.")
        return

    # Shared scales across all samples
    vmin, vmax = 0.0, 1.0
    # Clip error colormap at 99th percentile so small errors are visible
    all_errors = np.concatenate(
        [np.abs(patches[i] - recons[i]).ravel() for i in range(n)]
    )
    err_max = float(np.percentile(all_errors, 99))

    row_labels = ["Input cSDF", "Reconstructed", "|Error|"]
    row_cmaps  = ["gray", "gray", "hot"]
    row_ranges = [(vmin, vmax), (vmin, vmax), (0.0, err_max)]

    # Leave a right margin column for colorbars
    fig = plt.figure(figsize=(4 * n + 0.6, 10))
    gs = gridspec.GridSpec(
        3, n + 1,
        hspace=0.35, wspace=0.08,
        width_ratios=[4] * n + [0.25],
    )

    # Keep last image object per row to attach its colorbar
    last_im = [None, None, None]

    for col in range(n):
        orig = patches[col]
        rec  = recons[col]
        err  = np.abs(orig - rec)
        mse  = float(np.mean((orig - rec) ** 2))

        for row, (img, cmap, (mn, mx)) in enumerate(
            zip([orig, rec, err], row_cmaps, row_ranges)
        ):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=cmap, vmin=mn, vmax=mx, interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_title(row_labels[row], loc="left", fontsize=9, pad=4)
            if row == 2:
                ax.set_xlabel(f"MSE={mse:.5f}", fontsize=8)
            last_im[row] = im

    # Colorbars in the right margin column
    for row in range(3):
        cax = fig.add_subplot(gs[row, n])
        cb = fig.colorbar(last_im[row], cax=cax)
        cb.ax.tick_params(labelsize=7)
        if row == 2:
            cb.set_label("abs error\n(p99 clipped)", fontsize=7)

    fig.suptitle(f"Reconstructions — epoch {ckpt['epoch']}", fontsize=12)
    _show_or_save(fig, save_path, "reconstructions.png")


# ─── Utility ──────────────────────────────────────────────────────────────────


def _show_or_save(fig: plt.Figure, save_dir: Path | None, filename: str) -> None:
    if save_dir is not None:
        out = save_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)
    else:
        plt.show()


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
        "--samples", type=int, default=4,
        help="Number of validation patches to reconstruct (default: 4)",
    )
    p.add_argument(
        "--save-dir", type=Path, default=None,
        help="Save figures to this directory instead of displaying them",
    )
    p.add_argument(
        "--no-dataset", action="store_true",
        help="Skip reconstruction figures (useful when dataset is unavailable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    ckpt_dir = args.checkpoint.parent
    metrics = load_metrics(ckpt_dir)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {ckpt['epoch']}")
    print(f"Best val   : {ckpt['best_val_loss']:.6f}")
    print(f"Metrics    : {len(metrics)} epoch(s) recorded")
    print()

    plot_loss_curves(metrics, save_dir)
    plot_model_summary(ckpt, save_dir)
    if not args.no_dataset:
        plot_reconstructions(ckpt, n_samples=args.samples, save_path=save_dir)


if __name__ == "__main__":
    main()

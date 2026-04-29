"""Training entry point for the CurveCodec.

Loads config from a YAML file, builds the dataset, model, optimizer, and
loss, then runs the training loop with periodic validation.  Saves
checkpoints to ``checkpoints/<run_id>/`` and emits per-epoch metrics to
``checkpoints/<run_id>/metrics.jsonl``.

Usage::

    python train/train.py --config train/config/baseline.yaml
    python train/train.py --config train/config/baseline.yaml --resume
    python train/train.py --config train/config/baseline.yaml --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.loss import ReconLoss
from codec.model import CurveCodec
from codec.model_attn import CurveCodecAttn
from codec.model_v2 import CurveCodecV2
from dataset.dataset import make_dataloaders

log = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# ─── Config helpers ───────────────────────────────────────────────────────────


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    bits = int(cfg["model"]["quantizer_bits"])
    if bits not in (8, 16):
        raise ValueError(f"model.quantizer_bits must be 8 or 16, got {bits}")

    c = int(cfg["model"]["compaction_ratio"])
    import math
    n = round(math.log2(c))
    if 2 ** n != c:
        raise ValueError(f"model.compaction_ratio must be a power of 2, got {c}")

    return cfg


# ─── Optimizer / scheduler builders ──────────────────────────────────────────


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the optimizer from config.

    Supported values for ``training.optimizer``: ``adam``, ``adamw``, ``sgd``.
    """
    train_cfg = cfg["training"]
    lr: float = float(train_cfg["lr"])
    wd: float = float(train_cfg.get("weight_decay", 0.0))
    opt_name: str = str(train_cfg.get("optimizer", "adamw")).lower()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )
    raise ValueError(
        f"training.optimizer must be 'adam', 'adamw', or 'sgd', got {opt_name!r}"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the LR scheduler from config.

    Supported values for ``training.lr_scheduler``:
    ``cosine``, ``step``, ``none``.
    """
    train_cfg = cfg["training"]
    sched_name: str = str(train_cfg.get("lr_scheduler", "cosine")).lower()
    epochs: int = int(train_cfg["epochs"])
    warmup: int = int(train_cfg.get("warmup_epochs", 0))

    if sched_name == "none":
        return None

    if sched_name == "cosine":
        lr_min: float = float(train_cfg.get("lr_min", 1e-7))
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1), eta_min=lr_min
        )
        if warmup > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup
            )
            return SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
            )
        return cosine

    if sched_name == "step":
        step_size: int = int(train_cfg.get("lr_step_size", 50))
        gamma: float = float(train_cfg.get("lr_step_gamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(
        f"training.lr_scheduler must be 'cosine', 'step', or 'none', got {sched_name!r}"
    )


# ─── Train / val loops ────────────────────────────────────────────────────────


def train_one_epoch(
    model: CurveCodec,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ReconLoss,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch.

    Returns:
        Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        x: torch.Tensor = batch["csdf"].to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_one_epoch(
    model: CurveCodec,
    loader: torch.utils.data.DataLoader,
    criterion: ReconLoss,
    device: torch.device,
) -> float:
    """Run one validation epoch.

    Returns:
        Mean loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x: torch.Tensor = batch["csdf"].to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────


def _ckpt_dir(cfg: dict[str, Any]) -> Path:
    run_id: str = cfg["paths"]["run_id"]
    return _PROJECT_ROOT / cfg["paths"]["checkpoint_dir"] / run_id


def save_checkpoint(
    path: Path,
    model: CurveCodec,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CurveCodec,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[int, float]:
    """Load a checkpoint in-place.  Returns ``(start_epoch, best_val_loss)``."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    log.info("Resumed from %s (epoch %d)", path, ckpt["epoch"])
    return int(ckpt["epoch"]) + 1, float(ckpt["best_val_loss"])


# ─── Metrics log ─────────────────────────────────────────────────────────────


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─── Main training function ───────────────────────────────────────────────────


def train(cfg: dict[str, Any], device: torch.device, resume: bool = False) -> None:
    """Run the full training procedure.

    Args:
        cfg: Parsed YAML config dict.
        device: Target device for training.
        resume: If ``True``, resume from the last checkpoint in the run dir.
    """
    train_cfg = cfg["training"]
    epochs: int = int(train_cfg["epochs"])
    val_every: int = int(train_cfg.get("val_every_n_epochs", 1))
    grad_clip: float = float(train_cfg.get("grad_clip", 0.0))

    ckpt_dir = _ckpt_dir(cfg)
    metrics_path = ckpt_dir / "metrics.jsonl"

    # ── CUDA warm-up ─────────────────────────────────────────────────────────
    # Force full CUDA context + allocator initialisation in the main process
    # before DataLoader workers spawn.  Prevents NVML assertion failures in
    # containerised environments where NVML initialises lazily.
    if device.type == "cuda":
        _warmup = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        del _warmup

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = cfg.get("model", {}).get("arch", "curve_codec")
    _ARCH_MAP = {"curve_codec": CurveCodec, "curve_codec_v2": CurveCodecV2, "curve_codec_attn": CurveCodecAttn}
    if arch not in _ARCH_MAP:
        raise ValueError(f"model.arch must be one of {list(_ARCH_MAP)}, got '{arch}'")
    model = _ARCH_MAP[arch](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s: %d trainable parameters", arch, n_params)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = ReconLoss(cfg)
    log.info("Loss: %s", criterion)

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers: int = int(train_cfg.get("num_workers", 2))
    loaders = make_dataloaders(
        cfg,
        splits=["train", "validation"],
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" and num_workers > 0),
        persistent_workers=(num_workers > 0),
    )
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    log.info(
        "Dataset: %d train / %d val patches",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            start_epoch, best_val_loss = load_checkpoint(
                last_ckpt, model, optimizer, scheduler
            )
        else:
            log.warning("--resume requested but no checkpoint found at %s", last_ckpt)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip
        )
        elapsed = time.time() - t0

        # Get current LR (first param group)
        current_lr = optimizer.param_groups[0]["lr"]

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": current_lr,
            "elapsed_s": round(elapsed, 2),
        }

        val_loss: float | None = None
        if (epoch + 1) % val_every == 0:
            val_loss = val_one_epoch(model, val_loader, criterion, device)
            record["val_loss"] = val_loss

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer, scheduler,
                    epoch, best_val_loss, cfg,
                )

            log.info(
                "Epoch %4d/%d  train=%.6f  val=%.6f%s  lr=%.2e  %.1fs",
                epoch + 1, epochs, train_loss, val_loss,
                " ✓" if is_best else "  ",
                current_lr, elapsed,
            )
        else:
            log.info(
                "Epoch %4d/%d  train=%.6f  lr=%.2e  %.1fs",
                epoch + 1, epochs, train_loss, current_lr, elapsed,
            )

        # Always save the last checkpoint
        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, scheduler,
            epoch, best_val_loss, cfg,
        )

        # Append metrics record
        _append_metrics(metrics_path, record)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

    log.info(
        "Training complete.  Best val loss: %.6f  Checkpoints: %s",
        best_val_loss, ckpt_dir,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config", type=Path, default=Path("train/config/baseline.yaml"),
        help="YAML config file (default: train/config/baseline.yaml)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Training device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/<run_id>/last.pt",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    cfg = load_config(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train(cfg, device=device, resume=args.resume)


if __name__ == "__main__":
    main()

"""Loss functions for training the CurveCodec.

Defines the reconstruction loss used to train the autoencoder on cSDF
patches.  Pixels within ``erosion_margin_px`` of every edge are excluded
from the loss because they lie inside the encoder's receptive-field
boundary and may be contaminated by zero-padding artefacts.

Supported reconstruction losses (``training.loss.recon`` in config):

* ``"mse"``   — mean squared error (default)
* ``"l1"``    — mean absolute error
* ``"huber"`` — Huber / smooth-L1; delta from ``training.loss.huber_delta``
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_VALID_RECON: frozenset[str] = frozenset({"mse", "l1", "huber"})


class ReconLoss(nn.Module):
    """Pixel-wise reconstruction loss with receptive-field erosion mask.

    Args:
        config: Parsed YAML config dict.  Reads ``training.loss.recon``,
            ``training.loss.huber_delta``, and
            ``training.loss.erosion_margin_px``.

    Raises:
        ValueError: If ``training.loss.recon`` is not a recognised value.
        ValueError: If ``erosion_margin_px`` is negative.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        loss_cfg: dict[str, Any] = config.get("training", {}).get("loss", {})

        recon: str = str(loss_cfg.get("recon", "mse"))
        if recon not in _VALID_RECON:
            raise ValueError(
                f"training.loss.recon must be one of {sorted(_VALID_RECON)}, got {recon!r}"
            )
        self._recon: str = recon
        self._huber_delta: float = float(loss_cfg.get("huber_delta", 1.0))

        margin: int = int(loss_cfg.get("erosion_margin_px", 0))
        if margin < 0:
            raise ValueError(f"training.loss.erosion_margin_px must be ≥ 0, got {margin}")
        self._margin: int = margin

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """Compute reconstruction loss, ignoring border pixels.

        Args:
            x_hat: Reconstructed cSDF ``[B, 1, S, S]``.
            x:     Ground-truth cSDF ``[B, 1, S, S]``.

        Returns:
            Scalar loss tensor.

        Raises:
            RuntimeError: If the eroded patch has zero spatial extent.
        """
        m = self._margin
        if m > 0:
            h, w = x.shape[-2], x.shape[-1]
            if h - 2 * m <= 0 or w - 2 * m <= 0:
                raise RuntimeError(
                    f"erosion_margin_px={m} exceeds patch size ({h}×{w}); "
                    "reduce training.loss.erosion_margin_px"
                )
            x_hat = x_hat[..., m : h - m, m : w - m]
            x = x[..., m : h - m, m : w - m]

        if self._recon == "mse":
            return F.mse_loss(x_hat, x)
        if self._recon == "l1":
            return F.l1_loss(x_hat, x)
        # huber
        return F.huber_loss(x_hat, x, delta=self._huber_delta)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def margin(self) -> int:
        """Erosion margin in pixels."""
        return self._margin

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"recon={self._recon!r}, "
            f"margin={self._margin}px"
            + (f", huber_delta={self._huber_delta}" if self._recon == "huber" else "")
            + ")"
        )

"""Loss functions for training the CurveCodec.

Defines the reconstruction loss used to train the autoencoder on cSDF
patches.  Pixels within ``erosion_margin_px`` of every edge are excluded
from the loss because they lie inside the encoder's receptive-field
boundary and may be contaminated by zero-padding artefacts.

## Uniform mode (default, ``flat_loss_weight = 0``)

Applies a single reconstruction loss over all pixels in the eroded patch:

* ``"mse"``   — mean squared error
* ``"l1"``    — mean absolute error
* ``"huber"`` — Huber / smooth-L1; delta from ``training.loss.huber_delta``

## Split mode (``flat_loss_weight > 0``)

The cSDF truncation gives a natural pixel split:

* **Boundary band** — ``flat_threshold < gt < 1 − flat_threshold``
  Pixels within the truncation distance of the edge.  The cSDF value here
  is a continuous signal that encodes subpixel contour position → MSE/L1/Huber.

* **Flat region** — ``gt ≤ flat_threshold`` or ``gt ≥ 1 − flat_threshold``
  Pixels beyond the truncation distance, clipped to exactly 0 (exterior) or
  1 (interior) by the rasteriser.  The task here is binary classification:
  is this pixel inside or outside? → BCE on the decoder output.

Total loss:

    L = mean_boundary(recon_loss) + flat_loss_weight · mean_flat(BCE)

BCE is computed directly on the decoder's output — no model change is
required.  ``F.binary_cross_entropy`` takes ``pred ∈ (0, 1)`` and binary
targets, and is numerically stable with the internal clamp that PyTorch
applies.  The explicit ``.clamp(ε, 1−ε)`` below is a second-layer guard
against rare floating-point extremes.  Note: split mode requires decoder
output in ``(0, 1)``; it is incompatible with a Tanh output head.

``flat_loss_weight`` (λ) balances the two terms.  BCE on flat pixels is
naturally larger in magnitude than MSE on boundary pixels at comparable
convergence quality, so λ < 1 is typical (default 0.1).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_VALID_RECON: frozenset[str] = frozenset({"mse", "l1", "huber"})

_BCE_EPS: float = 1e-7   # clamp guard for BCE on decoder outputs


class ReconLoss(nn.Module):
    """Pixel-wise reconstruction loss with receptive-field erosion mask.

    Args:
        config: Parsed YAML config dict.  Reads the following keys under
            ``training.loss``:

            * ``recon``               — uniform/boundary term: mse | l1 | huber (default mse)
            * ``huber_delta``         — delta for huber (default 1.0)
            * ``erosion_margin_px``   — border pixels to exclude (default 0)
            * ``flat_loss_weight``    — λ for BCE flat-region term (default 0.0 = uniform mode)
            * ``flat_threshold``      — epsilon to detect gt ≈ 0 or gt ≈ 1 (default 1e-6)

    Raises:
        ValueError: On invalid config values.
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

        flat_loss_weight: float = float(loss_cfg.get("flat_loss_weight", 0.0))
        if flat_loss_weight < 0.0:
            raise ValueError(
                f"training.loss.flat_loss_weight must be ≥ 0, got {flat_loss_weight}"
            )
        self._flat_loss_weight: float = flat_loss_weight

        flat_threshold: float = float(loss_cfg.get("flat_threshold", 1e-6))
        if not (0.0 < flat_threshold < 0.5):
            raise ValueError(
                f"training.loss.flat_threshold must be in (0, 0.5), got {flat_threshold}"
            )
        self._flat_threshold: float = flat_threshold

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """Compute reconstruction loss, ignoring border pixels.

        Args:
            x_hat: Reconstructed cSDF ``[B, 1, S, S]`` in ``[0, 1]``.
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

        if self._flat_loss_weight > 0.0:
            return self._split_forward(x_hat, x)

        # ── uniform mode (original behaviour) ────────────────────────────────
        if self._recon == "mse":
            return F.mse_loss(x_hat, x)
        if self._recon == "l1":
            return F.l1_loss(x_hat, x)
        return F.huber_loss(x_hat, x, delta=self._huber_delta)

    def _split_forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        eps = self._flat_threshold
        boundary_mask = (x > eps) & (x < 1.0 - eps)
        flat_mask = ~boundary_mask

        # ── boundary: regression on continuous cSDF (contour position) ───────
        if boundary_mask.any():
            xh_b, x_b = x_hat[boundary_mask], x[boundary_mask]
            if self._recon == "mse":
                boundary_loss = F.mse_loss(xh_b, x_b)
            elif self._recon == "l1":
                boundary_loss = F.l1_loss(xh_b, x_b)
            else:
                boundary_loss = F.huber_loss(xh_b, x_b, delta=self._huber_delta)
        else:
            boundary_loss = x_hat.new_tensor(0.0)

        # ── flat: binary classification (interior vs exterior) ────────────────
        if flat_mask.any():
            flat_loss = F.binary_cross_entropy(
                x_hat[flat_mask].clamp(_BCE_EPS, 1.0 - _BCE_EPS),
                x[flat_mask],
            )
        else:
            flat_loss = x_hat.new_tensor(0.0)

        return boundary_loss + self._flat_loss_weight * flat_loss

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def margin(self) -> int:
        """Erosion margin in pixels."""
        return self._margin

    def __repr__(self) -> str:
        parts = [f"recon={self._recon!r}", f"margin={self._margin}px"]
        if self._recon == "huber":
            parts.append(f"huber_delta={self._huber_delta}")
        if self._flat_loss_weight > 0.0:
            parts.append(f"flat_loss_weight={self._flat_loss_weight}")
            parts.append(f"flat_threshold={self._flat_threshold}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

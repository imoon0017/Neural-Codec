"""QuantizerLayer for the CurveCodec.

Implements straight-through estimation (STE) during training and rounding
during evaluation.  Supports integer export (``int8`` when
``quantizer_bits=8``, ``int16`` when ``quantizer_bits=16``) as determined
by config.  Must only be called from within ``CurveCodec.forward()``.

Quantization scheme (per-channel scalar uniform):

    q_{d,i,j} = round(clamp(z_{d,i,j} / s_d, -2^(B-1), 2^(B-1)-1))
    z_hat_{d,i,j} = q_{d,i,j} * s_d

``s_d`` are ``nn.Parameter`` values learned jointly with AE weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class _STERound(torch.autograd.Function):
    """Round with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:  # noqa: ANN001
        return x.round()

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:  # noqa: ANN001
        return grad  # identity — straight-through


class QuantizerLayer(nn.Module):
    """Per-channel scalar uniform quantizer with STE training.

    Args:
        latent_dim: Number of latent channels ``D``.
        bits: Quantization bit-width (8 or 16).
    """

    def __init__(self, latent_dim: int, bits: int) -> None:
        super().__init__()
        if bits not in (8, 16):
            raise ValueError(f"quantizer_bits must be 8 or 16, got {bits}")
        self._bits = bits
        self._q_min: int = -(2 ** (bits - 1))
        self._q_max: int = 2 ** (bits - 1) - 1
        # Learned per-channel scale factors; stored as log to keep them positive
        self.log_scale = nn.Parameter(torch.zeros(latent_dim))

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, z: Tensor) -> Tensor:
        """Quantize-dequantize with STE (train) or true round (eval).

        Args:
            z: Continuous latent map ``[B, D, H, W]``.

        Returns:
            Dequantized latent map ``[B, D, H, W]`` in ``float32``.
        """
        s = self.log_scale.exp().view(1, -1, 1, 1)  # [1, D, 1, 1]
        z_scaled = z / s

        if self.training:
            q = _STERound.apply(z_scaled).clamp(self._q_min, self._q_max)
        else:
            with torch.no_grad():
                q = z_scaled.round().clamp(self._q_min, self._q_max)

        return q * s

    # ── export ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def export_dna(self, z: Tensor) -> Tensor:
        """Quantize and cast to integer dtype for ``.cdna`` export.

        Args:
            z: Continuous latent map ``[B, D, H, W]``.

        Returns:
            Integer curve-DNA tensor; dtype ``int8`` or ``int16``.
        """
        s = self.log_scale.exp().view(1, -1, 1, 1)
        q = (z / s).round().clamp(self._q_min, self._q_max)
        dtype = torch.int8 if self._bits == 8 else torch.int16
        return q.to(dtype)

    @property
    def scale_factors(self) -> Tensor:
        """Per-channel scale factors ``s_d`` as a ``[D]`` float32 tensor."""
        return self.log_scale.exp().detach()

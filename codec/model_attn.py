"""CurveCodecAttn: autoencoder with windowed self-attention at the encoder bottleneck.

Composes ``EncoderAttn`` → ``QuantizerLayer`` → ``Decoder``.  The only
architectural difference from ``CurveCodec`` is that the encoder finishes
with a ``WindowedSelfAttentionBlock`` at the ``S/c × S/c`` latent resolution,
giving the model global spatial context before quantization.

The public interface (``forward``, ``encode``, ``decode``, ``export_dna``,
``quantize`` flag) is identical to ``CurveCodec``.

Config keys (in addition to the base set)::

    model.attn_heads        — attention heads (default 4)
    model.attn_window_size  — window side length w (default 7)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.decoder import Decoder
from codec.encoder_attn import EncoderAttn
from codec.quantizer import QuantizerLayer


class CurveCodecAttn(nn.Module):
    """AE with bottleneck attention: cSDF → EncoderAttn → Quantizer → Decoder → cSDF.

    Args:
        config: Parsed YAML config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        bits: int = int(config["model"]["quantizer_bits"])
        if bits not in (8, 16):
            raise ValueError(f"model.quantizer_bits must be 8 or 16, got {bits}")

        D: int = int(config["model"]["latent_dim"])
        self.quantize: bool = bool(config["model"].get("quantize", True))
        self.encoder = EncoderAttn(config)
        self.quantizer = QuantizerLayer(latent_dim=D, bits=bits)
        self.decoder = Decoder(config)

    def forward(self, x: Tensor) -> Tensor:
        """Encode (with attention), quantize-dequantize, and decode.

        Args:
            x: cSDF patches ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Reconstructed cSDF patches ``[B, 1, S, S]`` in ``[0, 1]``.
        """
        z = self.encoder(x)
        z_q = self.quantizer(z) if self.quantize else z
        return self.decoder(z_q)

    def encode(self, x: Tensor) -> Tensor:
        """Encode to continuous latent map (attention applied).

        Args:
            x: cSDF patches ``[B, 1, S, S]``.

        Returns:
            Continuous latent map ``[B, D, S/c, S/c]``.
        """
        return self.encoder(x)

    def decode(self, z_q: Tensor) -> Tensor:
        """Decode from (dequantized) latent map.

        Args:
            z_q: Latent map ``[B, D, S/c, S/c]`` in ``float32``.

        Returns:
            Reconstructed cSDF patches ``[B, 1, S, S]``.
        """
        return self.decoder(z_q)

    def export_dna(self, x: Tensor) -> Tensor:
        """Encode (with attention) and export integer curve DNA.

        Args:
            x: cSDF patches ``[B, 1, S, S]``.

        Returns:
            Integer curve-DNA ``[B, D, S/c, S/c]``; dtype ``int8`` or ``int16``.
        """
        z = self.encoder(x)
        return self.quantizer.export_dna(z)

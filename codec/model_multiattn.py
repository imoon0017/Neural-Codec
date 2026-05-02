"""CurveCodecMultiAttn: autoencoder with windowed self-attention after every encoder stage.

Identical public interface to ``CurveCodec`` and ``CurveCodecAttn``; the only
difference is that the encoder is ``EncoderMultiAttn`` instead of ``Encoder`` or
``EncoderAttn``.

Set ``model.arch: curve_codec_multiattn`` in your config to use this model.
See ``codec/encoder_multiattn.py`` for the full architecture description and
config keys (``attn_heads``, ``attn_window_sizes``).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.decoder import Decoder
from codec.encoder_multiattn import EncoderMultiAttn
from codec.quantizer import QuantizerLayer


class CurveCodecMultiAttn(nn.Module):
    """AE with per-stage attention: cSDF → EncoderMultiAttn → Quantizer → Decoder → cSDF.

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
        self.encoder = EncoderMultiAttn(config)
        self.quantizer = QuantizerLayer(latent_dim=D, bits=bits)
        self.decoder = Decoder(config)

    def forward(self, x: Tensor) -> Tensor:
        """Encode (with per-stage attention), quantize-dequantize, and decode.

        Args:
            x: cSDF patches ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Reconstructed cSDF patches ``[B, 1, S, S]`` in ``[0, 1]``.
        """
        z = self.encoder(x)
        z_q = self.quantizer(z) if self.quantize else z
        return self.decoder(z_q)

    def encode(self, x: Tensor) -> Tensor:
        """Encode to continuous latent map.

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
        """Encode and export integer curve DNA for ``.cdna`` archiving.

        Args:
            x: cSDF patches ``[B, 1, S, S]``.

        Returns:
            Integer curve-DNA ``[B, D, S/c, S/c]``; dtype ``int8`` or ``int16``.
        """
        z = self.encoder(x)
        return self.quantizer.export_dna(z)

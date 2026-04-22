"""CurveCodecV2: autoencoder with anti-aliased encoder and transposed-conv decoder.

Composes ``EncoderAA`` → ``QuantizerLayer`` → ``DecoderTC``.  Compared to
``CurveCodec``:

* **EncoderAA** — stride-2 down-blocks replaced by stride-1 conv +
  ``BlurPool2d``, eliminating sub-sampling aliasing.
* **DecoderTC** — pixel-shuffle up-blocks replaced by ``ConvTranspose2d``
  up-blocks, eliminating the chessboard artefacts introduced by the
  channel-to-pixel shuffle when the mapping is imperfectly learned.

The public interface (``forward``, ``encode``, ``decode``, ``export_dna``) is
identical to ``CurveCodec``.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.decoder_tc import DecoderTC
from codec.encoder_aa import EncoderAA
from codec.quantizer import QuantizerLayer


class CurveCodecV2(nn.Module):
    """Anti-aliased autoencoder: cSDF → EncoderAA → Quantizer → DecoderTC → cSDF.

    Args:
        config: Parsed YAML config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        bits: int = int(config["model"]["quantizer_bits"])
        if bits not in (8, 16):
            raise ValueError(f"model.quantizer_bits must be 8 or 16, got {bits}")

        D: int = int(config["model"]["latent_dim"])
        self.encoder = EncoderAA(config)
        self.quantizer = QuantizerLayer(latent_dim=D, bits=bits)
        self.decoder = DecoderTC(config)

    def forward(self, x: Tensor) -> Tensor:
        """Encode, quantize-dequantize, and decode.

        Args:
            x: cSDF patches ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Reconstructed cSDF patches ``[B, 1, S, S]`` in ``[0, 1]``.
        """
        z = self.encoder(x)
        z_q = self.quantizer(z)
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

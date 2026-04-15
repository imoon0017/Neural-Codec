"""Unpack a .cdna archive and dequantize to float32 latent tensors.

Usage::

    from io.unpack import load_cdna, dequantize

    dna, meta = load_cdna(Path("foo.cdna"))
    z = dequantize(dna, meta["scale_factors"])   # float32 [N, D, Sc, Sc]
    x_hat = model.decode(z.to(device))           # [N, 1, S, S]
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_cdna(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a ``.cdna`` archive from disk.

    Args:
        path: Path to the ``.cdna`` zip file.

    Returns:
        A ``(dna, meta)`` tuple where:

        * ``dna`` — integer numpy array of shape ``[N, D, Sc, Sc]``, dtype
          ``int8`` (``quantizer_bits=8``) or ``int16``
          (``quantizer_bits=16``).
        * ``meta`` — parsed ``meta.json`` dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the archive is missing required entries or the
                    reconstructed shape is inconsistent.
    """
    if not path.exists():
        raise FileNotFoundError(f".cdna file not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if "meta.json" not in names:
            raise ValueError(f".cdna missing meta.json: {path}")
        if "dna.bin" not in names:
            raise ValueError(f".cdna missing dna.bin: {path}")
        meta: dict[str, Any] = json.loads(zf.read("meta.json").decode())
        raw_bytes = zf.read("dna.bin")

    bits: int = int(meta["quantizer_bits"])
    if bits not in (8, 16):
        raise ValueError(f"Unsupported quantizer_bits={bits} in {path}")
    dtype = np.int8 if bits == 8 else np.int16

    n_markers: int = len(meta["markers"])
    D: int = int(meta["latent_dim"])
    c: int = int(meta["compaction_ratio"])
    S: int = int(meta["patch_size_px"])
    Sc: int = S // c

    expected_bytes = n_markers * D * Sc * Sc * dtype().itemsize
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"dna.bin size mismatch: expected {expected_bytes} bytes "
            f"({n_markers}×{D}×{Sc}×{Sc} {dtype.__name__}), "
            f"got {len(raw_bytes)} in {path}"
        )

    dna = np.frombuffer(raw_bytes, dtype=dtype).reshape(n_markers, D, Sc, Sc).copy()
    return dna, meta


def dequantize(
    dna: np.ndarray,
    scale_factors: list[float] | np.ndarray,
) -> torch.Tensor:
    """Dequantize integer curve DNA to a float32 latent map.

    Applies the per-channel inverse quantization:
    ``z_hat[b, d, h, w] = dna[b, d, h, w] * scale_factors[d]``

    Args:
        dna: Integer array ``[N, D, Sc, Sc]`` (int8 or int16).
        scale_factors: Per-channel scale factors ``s_d``, length ``D``.

    Returns:
        Float32 tensor ``[N, D, Sc, Sc]`` ready to pass to
        :meth:`~codec.model.CurveCodec.decode`.
    """
    q = torch.from_numpy(dna.astype(np.float32))  # [N, D, Sc, Sc]
    s = torch.tensor(scale_factors, dtype=torch.float32).view(1, -1, 1, 1)
    return q * s

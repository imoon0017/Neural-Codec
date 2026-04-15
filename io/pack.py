"""Pack integer curve DNA and metadata into a .cdna zip archive.

The archive contains two entries:

* ``dna.bin`` — raw C-contiguous little-endian integer array, shape
  ``[N, D, S//c, S//c]``, dtype int8 (quantizer_bits=8) or int16 (quantizer_bits=16).
* ``meta.json`` — JSON object with model hyperparameters, per-channel scale
  factors, per-marker spatial coordinates, and a checkpoint fingerprint.

Usage::

    from io.pack import save_cdna

    dna = model.export_dna(patches)           # [N, D, Sc, Sc] int8/int16 tensor
    save_cdna(
        path=output_dir / "foo.cdna",
        dna=dna,
        config=cfg,
        scale_factors=model.quantizer.scale_factors,
        markers=[{"x_nm": ..., "y_nm": ..., "size_nm": ...}, ...],
        checkpoint_path=ckpt_path,
    )
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_MODEL_VERSION = "v1.0.0"


def _checkpoint_hash(ckpt_path: Path) -> str:
    """SHA-256 hex digest (first 16 chars) of the checkpoint file."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_cdna(
    path: Path,
    dna: np.ndarray | torch.Tensor,
    config: dict[str, Any],
    scale_factors: np.ndarray | torch.Tensor | list[float],
    markers: list[dict[str, float]],
    checkpoint_path: Path | None = None,
) -> None:
    """Write a ``.cdna`` zip archive.

    Args:
        path: Destination path (created or overwritten).
        dna: Integer curve-DNA of shape ``[N, D, Sc, Sc]``, dtype ``int8``
             or ``int16``.  ``N`` must equal ``len(markers)``.
        config: Parsed YAML config dict; provides ``model.*`` and ``csdf.*``
                hyperparameters.
        scale_factors: Per-channel dequantization scale factors, length ``D``
                       (``float32``).
        markers: List of ``N`` dicts, each with keys ``x_nm``, ``y_nm``,
                 ``size_nm`` (bottom-left corner and side length of the
                 rasterization region, in nanometres).
        checkpoint_path: Optional; if given, its SHA-256 is embedded in
                         ``meta.json`` for reproducibility.

    Raises:
        ValueError: If ``dna`` has wrong dtype or rank, or ``len(dna) !=
                    len(markers)``.
    """
    # ── Normalise inputs ─────────────────────────────────────────────────────
    if isinstance(dna, torch.Tensor):
        dna_np = dna.cpu().numpy()
    else:
        dna_np = np.asarray(dna)
    if dna_np.dtype not in (np.int8, np.int16):
        raise ValueError(f"dna dtype must be int8 or int16, got {dna_np.dtype}")
    if dna_np.ndim != 4:
        raise ValueError(f"dna must be 4D [N, D, Sc, Sc], got shape {dna_np.shape}")
    if len(dna_np) != len(markers):
        raise ValueError(
            f"len(dna)={len(dna_np)} != len(markers)={len(markers)}"
        )

    if isinstance(scale_factors, torch.Tensor):
        sf = scale_factors.cpu().numpy().astype(np.float32)
    else:
        sf = np.asarray(scale_factors, dtype=np.float32)

    # ── Compute patch_size_px from dna shape and compaction_ratio ────────────
    c = int(config["model"]["compaction_ratio"])
    patch_size_px = dna_np.shape[3] * c  # Sc * c = S

    meta: dict[str, Any] = {
        "latent_dim": int(config["model"]["latent_dim"]),
        "compaction_ratio": c,
        "patch_size_px": patch_size_px,
        "quantizer_bits": int(config["model"]["quantizer_bits"]),
        "grid_res_nm_per_px": float(config["csdf"]["grid_res_nm_per_px"]),
        "truncation_px": float(config["csdf"]["truncation_px"]),
        "scale_factors": sf.tolist(),
        "markers": [
            {"x_nm": float(m["x_nm"]), "y_nm": float(m["y_nm"]),
             "size_nm": float(m["size_nm"])}
            for m in markers
        ],
        "model_version": _MODEL_VERSION,
        "checkpoint_hash": _checkpoint_hash(checkpoint_path) if checkpoint_path else "",
    }

    # ── Write zip ────────────────────────────────────────────────────────────
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure C-contiguous little-endian bytes
    dna_c = np.ascontiguousarray(dna_np)
    if dna_c.dtype.byteorder not in ("<", "=", "|"):
        dna_c = dna_c.byteswap().newbyteorder("<")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dna.bin", dna_c.tobytes())
        zf.writestr("meta.json", json.dumps(meta, indent=2))

"""Evaluation metrics for the CurveCodec.

Two official metrics:

* **Compression ratio** — file-size ratio between the original OASIS layout
  and the ``.cdna`` archive.  Both files must exist on disk before calling.
* **Area difference ratio** — fraction of geometry that was gained or lost
  during the encode → decode round-trip, computed via XOR area.

Both functions operate exclusively on files saved to disk; never on
in-memory tensors.

Usage::

    from eval.metrics import compression_ratio, area_difference_ratio

    cr  = compression_ratio(original_oas, cdna_path)
    adr = area_difference_ratio(original_oas, reconstructed_oas, mask_layer=1)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def compression_ratio(original_oas_path: Path, cdna_path: Path) -> float:
    """Return ``filesize(original) / filesize(.cdna)``.

    Both files must exist on disk at call time.

    Args:
        original_oas_path: Path to the original OASIS layout file.
        cdna_path: Path to the ``.cdna`` archive produced by the encoder.

    Returns:
        Compression ratio ≥ 0.0.  Values > 1 mean the ``.cdna`` is smaller
        than the original.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If the ``.cdna`` file is empty.
    """
    if not original_oas_path.exists():
        raise FileNotFoundError(f"Original OASIS not found: {original_oas_path}")
    if not cdna_path.exists():
        raise FileNotFoundError(f".cdna archive not found: {cdna_path}")

    orig_size = original_oas_path.stat().st_size
    cdna_size = cdna_path.stat().st_size
    if cdna_size == 0:
        raise ValueError(f".cdna file is empty: {cdna_path}")

    return orig_size / cdna_size


def area_difference_ratio(
    original_oas_path: Path,
    reconstructed_oas_path: Path,
    mask_layer: int = 1,
) -> float:
    """Return ``Area(original XOR reconstructed) / Area(original)``.

    Uses ``klayout.db.Region`` boolean XOR to compute the symmetric
    difference between the two geometry sets.  Areas are computed in DBU²
    units of the original layout (no conversion to nm² needed for a ratio).

    The result is clamped to ``[0, 1]``; a value > 1 (XOR area exceeds
    original) is physically possible when the reconstructed geometry extends
    outside the original mask and is logged as a warning but never discarded.

    Args:
        original_oas_path: Path to the original OASIS file.
        reconstructed_oas_path: Path to the reconstructed OASIS file.
        mask_layer: GDS layer number of the mask polygons (default ``1``).

    Returns:
        Area difference ratio in ``[0, 1]``.  ``0`` = perfect
        reconstruction; ``1`` = completely wrong.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    import klayout.db as db

    if not original_oas_path.exists():
        raise FileNotFoundError(f"Original OASIS not found: {original_oas_path}")
    if not reconstructed_oas_path.exists():
        raise FileNotFoundError(
            f"Reconstructed OASIS not found: {reconstructed_oas_path}"
        )

    def _load_region(oas_path: Path) -> "db.Region":
        layout = db.Layout()
        layout.read(str(oas_path))
        li = layout.layer(mask_layer, 0)
        region = db.Region()
        for cell in layout.each_cell():
            for shape in cell.shapes(li).each():
                if shape.is_polygon():
                    region.insert(shape.polygon)
                elif shape.is_box():
                    region.insert(db.Polygon(shape.box))
        return region

    orig_region = _load_region(original_oas_path)
    recon_region = _load_region(reconstructed_oas_path)

    orig_area: int = orig_region.area()
    if orig_area == 0:
        log.debug("Original region has zero area: %s", original_oas_path)
        return 0.0

    xor_region = orig_region ^ recon_region
    xor_area: int = xor_region.area()

    ratio = xor_area / orig_area
    if ratio > 1.0:
        log.warning(
            "XOR area exceeds original area (ratio=%.4f) for %s vs %s — clamping to 1.0",
            ratio,
            original_oas_path.name,
            reconstructed_oas_path.name,
        )
        ratio = 1.0

    return float(ratio)

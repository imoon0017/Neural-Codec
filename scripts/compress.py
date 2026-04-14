"""End-to-end compression script: OASIS layout → .cdna archive.

Reads an OASIS file, rasterizes each marked region to a cSDF patch,
encodes and quantizes to curve DNA via the CurveCodec, then serializes
the result to a ``.cdna`` zip archive containing ``dna.bin`` and
``meta.json``.
"""

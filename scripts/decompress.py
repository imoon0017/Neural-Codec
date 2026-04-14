"""End-to-end decompression script: .cdna archive → OASIS layout.

Reads a ``.cdna`` zip archive, dequantizes and decodes curve DNA to cSDF
patches via the CurveCodec, applies marching squares (iso=0.5) to recover
PWCL contours, then writes the result as an OASIS file via klayout.db.
"""

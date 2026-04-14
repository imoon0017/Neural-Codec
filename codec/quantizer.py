"""QuantizerLayer for the CurveCodec.

Implements straight-through estimation (STE) during training and rounding
during evaluation.  Supports integer export (``int8`` when
``quantizer_bits=8``, ``int16`` when ``quantizer_bits=16``) as determined
by config.  Must only be called from within ``CurveCodec.forward()``.
"""

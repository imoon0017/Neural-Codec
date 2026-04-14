"""Decoder network for the CurveCodec autoencoder.

Accepts a latent map of shape ``[B, D, S/c, S/c]`` and reconstructs a
cSDF patch of shape ``[B, 1, S, S]``.  ``D`` is ``model.latent_dim`` and
``c`` is ``model.compaction_ratio`` from config.
"""

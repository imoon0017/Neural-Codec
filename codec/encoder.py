"""Encoder network for the CurveCodec autoencoder.

Accepts a batch of cSDF patches of shape ``[B, 1, S, S]`` and produces a
continuous latent map of shape ``[B, D, S/c, S/c]``, where ``D`` is
``model.latent_dim`` and ``c`` is ``model.compaction_ratio`` from config.
"""

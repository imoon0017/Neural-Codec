"""CurveCodec: top-level autoencoder model.

Composes Encoder → QuantizerLayer → Decoder into a single ``nn.Module``.
Manages device placement via ``model.to(device)``; never hardcodes
``"cuda"`` or ``"cpu"``.  All hyperparameters (``D``, ``c``, ``B``) are
read from the YAML config.
"""

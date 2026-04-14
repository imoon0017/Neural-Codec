"""PyTorch Dataset for cSDF patches.

Supports two modes controlled by ``dataset.mode`` in config:

* ``cached`` — loads pre-rasterized ``.npy`` patches from
  ``dataset/cache/<split>/``.  Verifies ``manifest.yaml`` matches the
  current config before use.
* ``on_the_fly`` — rasterizes PWCL geometry on demand from raw OASIS
  files in ``dataset/raw/<split>/``.
"""

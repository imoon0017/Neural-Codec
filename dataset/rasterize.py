"""Dataset rasterization script.

Iterates over raw OASIS files listed in ``catalog.csv`` and converts each
PWCL polygon + marker layer into a pre-rasterized ``.npy`` cSDF patch
cached under ``dataset/cache/<split>/``.  Writes a ``manifest.yaml``
recording the config hash used, so training can verify cache freshness.
"""

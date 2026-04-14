"""OASIS layout ingest script.

The only sanctioned way to add new OASIS files to the dataset.  Validates
each source file (OASIS only — GDSII is rejected), copies it into the
appropriate ``dataset/raw/<split>/`` subdirectory, and updates
``dataset/catalog.csv``.  Reads rasterization config (mask layer, marker
layer, ``grid_res_nm_per_px``) from the supplied YAML config file.
"""

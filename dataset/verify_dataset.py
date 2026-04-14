"""Dataset integrity verification script.

Checks that every entry in ``catalog.csv`` has a corresponding raw OASIS
file, that cached ``.npy`` patches exist and match the manifest, and that
patch shapes are consistent with the current config
(``patch_size_px % compaction_ratio == 0``).
"""

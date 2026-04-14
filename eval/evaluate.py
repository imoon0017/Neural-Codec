"""Evaluation script for a trained CurveCodec checkpoint.

Runs the full pipeline — encode test patches to ``.cdna``, reload and
decode, reconstruct PWCL via marching squares, then compute and report
metrics.  Writes per-sample results and aggregate statistics to
``eval/results/<run_id>/``.
"""

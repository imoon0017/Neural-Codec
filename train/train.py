"""Training entry point for the CurveCodec.

Loads config from a YAML file, builds the dataset, model, optimizer, and
loss, then runs the training loop with periodic validation.  Saves
checkpoints to ``checkpoints/<run_id>/`` and logs metrics for
inspection.
"""

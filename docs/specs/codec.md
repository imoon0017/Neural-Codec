# codec/ — Neural Autoencoder (CurveCodec)

## Overview

Standard deterministic AE (not VAE). Full end-to-end graph lives in `CurveCodec` (`model.py`):

```
cSDF [B,1,S,S] → Encoder → QuantizerLayer → Decoder → reconstructed cSDF [B,1,S,S]
```

All operators must be **shift-invariant** (see below). CPU and GPU inference both supported
via `model.to(device)`.

## Tensor Shapes

| Stage | Shape | dtype |
|---|---|---|
| cSDF input | `[B, 1, S, S]` | `float32` |
| Continuous latent map | `[B, D, S/c, S/c]` | `float32` |
| Quantized latent (train/eval) | `[B, D, S/c, S/c]` | `float32` (rounded) |
| Curve DNA (export / archive) | `[B, D, S/c, S/c]` | `int8` or `int16` (from `quantizer_bits`) |
| Reconstructed cSDF | `[B, 1, S, S]` | `float32` |

- `D` = `model.latent_dim` — configurable, never hardcoded
- `c` = `model.compaction_ratio` — must be power of 2, must divide `S`
- `S/c` is always derived: `patch_size_px // compaction_ratio`

## QuantizerLayer — Three Operational Modes

| Mode | Trigger | Behaviour | Output dtype |
|---|---|---|---|
| **Train** | `model.train()` | STE: `round()` forward, identity gradient backward | `float32` |
| **Eval** | `model.eval()` | True `round()`, no STE | `float32` |
| **Export** | `model.export_dna(x)` | True `round()`, cast to integer dtype | `int8` or `int16` (from `quantizer_bits`) |

The decoder **always receives `float32`**. Integer codes exist only in the `.cdna` archive.
Dequantization (`q * s_d`) is applied in `io/unpack.py` before decoder inference.

### Quantization Scheme

Per-channel scalar uniform quantization (channel `d` shared across all spatial positions):

```
q_{d,i,j} = round(clamp(z_{d,i,j} / s_d, -2^(B-1), 2^(B-1)-1))
z_hat_{d,i,j} = q_{d,i,j} * s_d
```

- `B` = `model.quantizer_bits`, default 16 — maps to dtype: `8 → int8`, `16 → int16`
- Supported values: `8` and `16`. Reject any other value at config load time with a clear error.
- `s_d` are `nn.Parameter` values — learned jointly with AE weights
- At export: `s_d` written to `meta.json` in `.cdna` archive
- At decode: `s_d` loaded from `meta.json` — **never from the model checkpoint**
- **Do not use VQ/VQ-VAE** — scalar per-channel quantization only

### STE Implementation

```python
class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad):
        return grad  # straight-through
```

Never use `.detach()` hacks — they break decoder gradient flow. Verify with
`torch.autograd.gradcheck` that gradients reach encoder weights.

## Shift-Invariance Constraint

**Allowed:**
- `nn.Conv2d` with `padding='same'` or symmetric padding
- `nn.ConvTranspose2d` with consistent output padding
- `nn.AvgPool2d`, `nn.MaxPool2d` with `ceil_mode=False`
- `nn.BatchNorm2d`, `nn.GroupNorm`, `nn.LayerNorm` (spatial)
- Pointwise activations: `ReLU`, `GELU`, `SiLU`, etc.
- Pixel-shuffle / depth-to-space (when stride divides evenly)

**Prohibited:**
- `nn.Linear` on flattened spatial features
- Global average pooling (`nn.AdaptiveAvgPool2d(1)`) — collapses spatial structure
- Positional encodings (sinusoidal, learned, RoPE)
- Spatial self-attention with absolute position bias
- Any op indexing spatial coordinates `(i, j)` as a feature

Translating the input by `(dx, dy)` must translate the latent map by `(dx/c, dy/c)`.
Add a unit test: shift input by `(dx, dy)`, confirm curve DNA output shifts identically.

## Contouring (`contouring/`)

After decoding:
1. Apply `sigmoid` or `clamp(0, 1)` to decoder output — values outside `[0, 1]` cause spurious contours
2. Marching squares at **iso-level 0.5** → polygon vertices
3. PWCL fitting: re-classify curve segments (arc vs. Bézier) — lossy by design
4. Validate: no self-intersections, closed polygons, CCW outer / CW holes

## Gotchas

- **Never apply `QuantizerLayer` outside `CurveCodec.forward()`** — decoder trains on unquantized latents otherwise
- **Train/eval symmetry** — unit test: `model.train()` and `model.eval()` must produce identical outputs for the same input
- **`S/c` hardcoding is a bug** — always derive as `patch_size_px // compaction_ratio`
- **Scale factors must come from the archive** — loading from checkpoint produces silently wrong geometry
- **Avoid CPU-pathological ops** — large BatchNorm with small batch is slow on CPU; prefer GroupNorm

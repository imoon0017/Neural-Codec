# CLAUDE.md — Neural Codec: AE-Based Curved Mask Compressor

## Project Overview

Neural autoencoder codec for compressing IC/chip layout curved masks in EDA workflows.

Pipeline:
1. **Ingest** — OASIS curved mask layouts added via `dataset/ingest.py`
2. **Rasterize** — PWCL polygon + marker layer → cSDF patch `float32 [S,S]` (truncated normalized SDF; interior=1, exterior=0, boundary∈(0,1))
3. **Encode** — cSDF → continuous latent map `[D, S/c, S/c]` → quantized integer **curve DNA** `[D, S/c, S/c]` (dtype determined by `model.quantizer_bits`: 8→`int8`, 16→`int16`)
4. **Store** — curve DNA serialized to `.cdna` zip archive
5. **Decode on demand** — `.cdna` → dequantize → decoder → reconstructed cSDF → marching squares (iso=0.5) → PWCL

Use cases: layout visualization, DRC/simulation, layout manipulation.
Layout I/O exclusively via **KLayout** (`klayout.db`). OASIS only — reject GDSII.

**Detailed module specs**: `docs/specs/`

---

## Repository Structure

```
neural_codec/
├── csdf/                   # cSDF rasterization — C++/CUDA + pybind11
│   ├── rasterizer.cu       # GPU path (CUDA)
│   ├── rasterizer.cpp      # CPU path (OpenMP)
│   ├── rasterizer.h        # Shared interface
│   └── csdf_utils.py       # Python dispatch (cpu/cuda)
├── codec/                  # Neural AE (PyTorch)
│   ├── encoder.py          # cSDF [B,1,S,S] → latent [B,D,S/c,S/c]
│   ├── quantizer.py        # QuantizerLayer (STE train / round eval / intB export, B from config)
│   ├── decoder.py          # latent [B,D,S/c,S/c] → cSDF [B,1,S,S]
│   ├── model.py            # CurveCodec: encoder → quantizer → decoder
│   └── loss.py
├── contouring/             # cSDF → PWCL (marching squares + PWCL fitting)
├── io/                     # .cdna pack/unpack + OASIS I/O via klayout.db
├── dataset/                # Layout repository + data pipeline
│   ├── raw/train|validation|test/
│   ├── cache/train|validation|test/   # pre-rasterized .npy (DVC)
│   ├── catalog.csv
│   ├── ingest.py           # ONLY way to add OASIS files
│   ├── rasterize.py        # PWCL → .npy cache
│   ├── dataset.py          # PyTorch Dataset (cached / on-the-fly)
│   └── verify_dataset.py
├── eval/                   # encode → save → decode → save → metrics
├── train/
│   ├── train.py
│   └── config/baseline.yaml
├── scripts/
│   ├── compress.py         # OASIS → .cdna
│   └── decompress.py       # .cdna → OASIS
├── docs/specs/             # Module-level design specs
│   ├── csdf.md
│   ├── codec.md
│   ├── io.md
│   ├── dataset.md
│   └── eval.md
├── CMakeLists.txt
└── pyproject.toml
```

---

## Key Concepts

| Term | Definition |
|---|---|
| **PWCL** | Piecewise Curve Linear — mask geometry (control points + segment type: line/arc/Bézier) |
| **cSDF** | Custom SDF — truncated normalized distance field, `float32 [S,S]`, range `[0,1]` |
| **Curve DNA** | Integer latent map `[D, S/c, S/c]` (dtype: `int8`/`int16` per `quantizer_bits`) — the compressed artifact in `.cdna` |
| **Marker layer** | Square OASIS shape that defines the rasterization region for each polygon |
| **`S`** | Patch size in pixels — derived from marker size and `grid_res_nm_per_px`; one fixed value per dataset |
| **`c`** | Compaction ratio — spatial downsampling factor; power of 2, divides `S` |
| **`D`** | Latent feature channels — configurable via `model.latent_dim` |
| **`.cdna`** | Curve DNA archive — zip with `dna.bin` (integer dtype from `quantizer_bits`) + `meta.json` |

---

## Coding Standards

- Python ≥ 3.10, type annotations on all public APIs, Google-style docstrings
- Formatter: `black` (line length 100); linter: `ruff` — zero warnings before commit
- No `print()` in library code — use `logging.getLogger(__name__)`
- C++17, OpenMP on CPU path, `CUDA_CHECK()` macro on every CUDA call
- All file paths via `pathlib.Path`; physical units explicit in variable names (`pitch_nm`, `grid_res_nm_per_px`)
- `D`, `c`, `S`, `B` (quantizer bits) always from config — never hardcoded
- All PyTorch models: `model.to(device)` dispatch — never hardcode `"cuda"` or `"cpu"`

---

## Config (YAML)

```yaml
model:
  latent_dim: 32           # D
  compaction_ratio: 8      # c — power of 2, must divide S
  quantizer_bits: 16       # B — bits per code: 8 → int8 (4× smaller than float32), 16 → int16 (2× smaller)

csdf:
  grid_res_nm_per_px: 1.0  # primary rasterization parameter (nm)
  marker_layer: 10         # GDS layer of square marker shapes
  mask_layer: 1            # GDS layer of mask polygons
  truncation_px: 3.0       # band half-width t
  # patch_size_px is DERIVED: ceil(marker_size_nm / grid_res_nm_per_px), rounded to multiple of c

dataset:
  mode: cached             # "cached" or "on_the_fly"
  cache_dir: dataset/cache/

training:
  batch_size: 32
  lr: 1.0e-4
  epochs: 200
  val_every_n_epochs: 1

paths:
  dataset_root: dataset/raw/
  checkpoint_dir: checkpoints/
  run_id: baseline_v1
```

---

## Build & Environment

```bash
# Python environment
conda create -n neural_codec python=3.11
conda activate neural_codec
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install klayout
pip install -e ".[dev]"

# C++/CUDA extension
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON   # or OFF for CPU-only
make -j$(nproc)

# Dataset setup
dvc pull dataset/raw/
python dataset/ingest.py \
    --source /path/to/layouts/ --split train \
    --mask-layer 1 --marker-layer 10 \
    --config train/config/baseline.yaml --workers 16
python dataset/verify_dataset.py

# Training
python train/train.py --config train/config/baseline.yaml

# Evaluation
python eval/evaluate.py \
    --checkpoint checkpoints/<run_id>/best.pt \
    --config train/config/baseline.yaml \
    --data-dir dataset/raw/test/ \
    --output-dir eval/results/<run_id>/ \
    --device cuda

# End-to-end compress / decompress
python scripts/compress.py   --input layout.oas --output compressed.cdna --device cpu
python scripts/decompress.py --input compressed.cdna --output recovered.oas --device cpu

# Tests
pytest csdf/tests/ codec/tests/ contouring/tests/ io/tests/ eval/tests/ -v
```

---

## Review Checklist

- [ ] `ruff check .` and `black --check .` pass
- [ ] Type annotations and docstrings on all new public functions
- [ ] Unit tests added or updated; `pytest` passes in full
- [ ] `patch_size_px % compaction_ratio == 0` enforced at config load time
- [ ] `compaction_ratio` is a power of 2 — enforced at config load time
- [ ] `QuantizerLayer` only called inside `CurveCodec.forward()`
- [ ] `torch.autograd.gradcheck` passes through `QuantizerLayer` in train mode
- [ ] CPU and GPU paths both tested after rasterizer or model changes
- [ ] No positional encodings or absolute-position ops added to the model
- [ ] `eval/evaluate.py` runs to completion on a test subset after model changes
- [ ] `.cdna` `meta.json` version bumped if archive format changes
- [ ] New OASIS files added via `ingest.py` only — never copied manually
- [ ] `cache/manifest.yaml` matches config before training in cached mode
- [ ] No binary files committed to git — DVC for OASIS/cache, external storage for checkpoints

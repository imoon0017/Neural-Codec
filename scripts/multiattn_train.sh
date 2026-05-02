#!/usr/bin/env bash
# multiattn_train.sh — Train CurveCodecMultiAttn (D=8, c=16, B=8)
#
# Fixed:    arch=curve_codec_multiattn, attn_window_sizes=[7,5,3,3,3],
#           latent_dim=8, compaction_ratio=16, quantizer_bits=8,
#           encoder_channels=64, batch_size=1, epochs=200
#
# Usage:
#   ./scripts/multiattn_train.sh                 # cuda
#   ./scripts/multiattn_train.sh --device cpu
#   ./scripts/multiattn_train.sh --dry-run
#   ./scripts/multiattn_train.sh --resume
#
# Outputs:
#   checkpoints/multiattn_D8_c16_B8/train.log
#   checkpoints/multiattn_D8_c16_B8/best.pt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

RUN_ID="multiattn_D8_c16_B8"
CFG_PATH="${PROJECT_ROOT}/train/config/multiattn_split/${RUN_ID}.yaml"
CKPT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_ID}"
LOG_PATH="${CKPT_DIR}/train.log"

# ── Parse flags ───────────────────────────────────────────────────────────────

DEVICE=""
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            [[ "${DEVICE}" == "gpu" ]] && DEVICE="cuda"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

[[ -z "${DEVICE}" ]] && DEVICE="cuda"

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Print header ──────────────────────────────────────────────────────────────

echo "============================================================"
echo "  CurveCodecMultiAttn Training"
echo "  run_id            : ${RUN_ID}"
echo "  config            : ${CFG_PATH}"
echo "  device            : ${DEVICE}"
echo "  dry run           : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  resume            : $([ ${RESUME} -eq 1 ] && echo yes || echo no)"
echo "  arch              : curve_codec_multiattn  (fixed)"
echo "  attn_window_sizes : [7, 5, 3, 3, 3]  (fixed)"
echo "  latent_dim        : 8  (fixed)"
echo "  compaction_ratio  : 16  (fixed)"
echo "  quantizer_bits    : 8  (fixed)"
echo "  Date              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "ERROR — config not found: ${CFG_PATH}" >&2
    exit 1
fi

TRAIN_ARGS=("--config" "${CFG_PATH}" "--device" "${DEVICE}")
[[ ${RESUME} -eq 1 ]] && TRAIN_ARGS+=("--resume")

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] python train/train.py ${TRAIN_ARGS[*]}"
    exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
mkdir -p "${CKPT_DIR}"
echo "log → ${LOG_PATH}"
echo ""

cd "${PROJECT_ROOT}"
T_START=$(date +%s)
EXIT_CODE=0
python train/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_PATH}" || EXIT_CODE=$?
T_END=$(date +%s)
DURATION=$(( T_END - T_START ))

echo ""
echo "============================================================"
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "  ✗ Training failed (exit=${EXIT_CODE}, ${DURATION}s)"
    echo "  Log: ${LOG_PATH}"
else
    echo "  ✓ Training complete (${DURATION}s)"
    echo "  Checkpoint: ${CKPT_DIR}/best.pt"
    # Print best val loss from checkpoint
    python - "${CKPT_DIR}/best.pt" <<'PYEOF'
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
if p.exists():
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    bvl = ckpt.get("best_val_loss")
    ep  = ckpt.get("epoch")
    if bvl is not None:
        print(f"  Best val loss : {bvl:.6f}  (epoch {ep})")
PYEOF
fi
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
exit ${EXIT_CODE}

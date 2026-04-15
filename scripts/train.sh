#!/usr/bin/env bash
# train.sh — Launch CurveCodec training
#
# Usage:
#   ./scripts/train.sh                          # default config, cuda
#   ./scripts/train.sh --config path/to/cfg.yaml
#   ./scripts/train.sh --device cpu
#   ./scripts/train.sh --resume
#   ./scripts/train.sh --config my.yaml --resume --device cuda
#
# All unrecognised flags are forwarded directly to train/train.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

# ── Parse known flags (the rest are forwarded) ───────────────────────────────
FORWARD_ARGS=()
CONFIG="${PROJECT_ROOT}/train/config/baseline.yaml"
DEVICE=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";    FORWARD_ARGS+=("--config" "$2"); shift 2 ;;
        --device)   DEVICE="$2";    FORWARD_ARGS+=("--device" "$2"); shift 2 ;;
        --resume)   RESUME="1";     FORWARD_ARGS+=("--resume");      shift   ;;
        *)          FORWARD_ARGS+=("$1");                             shift   ;;
    esac
done

# If --config was not explicitly passed, inject the default
if [[ -z "$DEVICE" ]]; then
    FORWARD_ARGS+=("--device" "cuda")
fi

# ── Resolve conda ─────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Print run header ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  CurveCodec Training"
echo "  Config : ${CONFIG}"
echo "  Device : ${DEVICE:-cuda}"
echo "  Resume : ${RESUME:-no}"
echo "  Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Extract run_id from config for log naming ─────────────────────────────────
RUN_ID=$(python -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c['paths']['run_id'])" 2>/dev/null || echo "run")
LOG_DIR="${PROJECT_ROOT}/checkpoints/${RUN_ID}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"

echo "  Log    : ${LOG_FILE}"
echo "============================================================"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

python train/train.py "${FORWARD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

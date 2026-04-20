#!/usr/bin/env bash
# split_eval.sh — Evaluate all split-test CurveCodec checkpoints with --inspection
#
# Runs eval/evaluate.py sequentially for each split_D{D}_c{c}_B{B} checkpoint,
# writing per-marker inspection .oas files alongside the standard CSV/JSON.
#
# Usage:
#   ./scripts/split_eval.sh                    # default: cuda, test split
#   ./scripts/split_eval.sh --device cpu
#   ./scripts/split_eval.sh --split validation
#   ./scripts/split_eval.sh --dry-run          # print commands, don't execute
#
# Outputs (per run):
#   eval/results/split_D{D}_c{c}_B{B}/results.csv
#   eval/results/split_D{D}_c{c}_B{B}/summary.json
#   eval/results/split_D{D}_c{c}_B{B}/inspection/*.oas

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
RESULTS_ROOT="${PROJECT_ROOT}/eval/results"
SPLIT_CONFIG_DIR="${PROJECT_ROOT}/train/config/split_test"

LATENT_DIMS=(32 16 8)
COMPACTION_RATIOS=(8 16 32)
QUANTIZER_BITS=(16 8)

# ── Parse flags ────────────────────────────────────────────────────────────────

DEVICE=""
SPLIT="test"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            [[ "${DEVICE}" == "gpu" ]] && DEVICE="cuda"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${DEVICE}" ]]; then
    DEVICE="cuda"
fi

# ── Activate conda ─────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Print header ───────────────────────────────────────────────────────────────

TOTAL=$(( ${#LATENT_DIMS[@]} * ${#COMPACTION_RATIOS[@]} * ${#QUANTIZER_BITS[@]} ))

echo "============================================================"
echo "  CurveCodec Split-Test Evaluation (--inspection)"
echo "  Device       : ${DEVICE}"
echo "  Split        : ${SPLIT}"
echo "  Dry run      : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  Total runs   : ${TOTAL}"
echo "  Date         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Evaluate sequentially ──────────────────────────────────────────────────────

IDX=0
N_OK=0
N_SKIP=0
N_FAIL=0

cd "${PROJECT_ROOT}"

for D in "${LATENT_DIMS[@]}"; do
for c in "${COMPACTION_RATIOS[@]}"; do
for B in "${QUANTIZER_BITS[@]}"; do
    IDX=$(( IDX + 1 ))
    RUN_ID="split_D${D}_c${c}_B${B}"
    CKPT="${CHECKPOINT_ROOT}/${RUN_ID}/best.pt"
    CFG="${SPLIT_CONFIG_DIR}/${RUN_ID}.yaml"
    OUT_DIR="${RESULTS_ROOT}/${RUN_ID}"

    echo "── [${IDX}/${TOTAL}] ${RUN_ID} ──────────────────────────"

    if [[ ! -f "${CKPT}" ]]; then
        echo "  SKIP — checkpoint not found: ${CKPT}"
        N_SKIP=$(( N_SKIP + 1 ))
        echo ""
        continue
    fi

    if [[ ! -f "${CFG}" ]]; then
        echo "  SKIP — config not found: ${CFG}"
        N_SKIP=$(( N_SKIP + 1 ))
        echo ""
        continue
    fi

    echo "  checkpoint : ${CKPT}"
    echo "  config     : ${CFG}"
    echo "  output     : ${OUT_DIR}"

    CMD=(
        python eval/evaluate.py
        --checkpoint "${CKPT}"
        --config     "${CFG}"
        --output-dir "${OUT_DIR}"
        --device     "${DEVICE}"
        --split      "${SPLIT}"
        --inspection
    )

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] ${CMD[*]}"
        echo ""
        continue
    fi

    T_START=$(date +%s)
    EXIT_CODE=0
    "${CMD[@]}" 2>&1 | sed 's/^/  /' || EXIT_CODE=$?
    T_END=$(date +%s)
    DURATION=$(( T_END - T_START ))

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "  ✗ failed (exit=${EXIT_CODE}, ${DURATION}s)"
        N_FAIL=$(( N_FAIL + 1 ))
    else
        echo "  ✓ done (${DURATION}s)"
        N_OK=$(( N_OK + 1 ))
    fi
    echo ""
done
done
done

# ── Summary ────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  OK: ${N_OK}  Skipped: ${N_SKIP}  Failed: ${N_FAIL}  Total: ${TOTAL}"
echo "  Results root: ${RESULTS_ROOT}"
echo "============================================================"

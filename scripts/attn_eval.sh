#!/usr/bin/env bash
# attn_eval.sh — Evaluate the CurveCodecAttn checkpoint (with quantization)
#
# Model : CurveCodecAttn — conv encoder + windowed self-attention bottleneck
# Config: train/config/attn_v1.yaml
#         D=32, c=8, B=16, heads=4, window=7, batch=1, quantize=true
#
# Results written to:
#   eval/results/attn_v1/           best checkpoint
#   eval/results/attn_v1_last/      last checkpoint (optional, --last flag)
#
# Usage:
#   ./scripts/attn_eval.sh                        # default: cuda, validation split
#   ./scripts/attn_eval.sh --device cpu
#   ./scripts/attn_eval.sh --split test
#   ./scripts/attn_eval.sh --last                 # also eval last.pt
#   ./scripts/attn_eval.sh --inspection           # write per-marker .oas files
#   ./scripts/attn_eval.sh --dry-run

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

RUN_ID="attn_v1"
CONFIG="${PROJECT_ROOT}/train/config/${RUN_ID}.yaml"
CKPT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_ID}"
RESULTS_ROOT="${PROJECT_ROOT}/eval/results"

# ── Parse flags ───────────────────────────────────────────────────────────────

DEVICE=""
SPLIT="validation"
INSPECTION=0
EVAL_LAST=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            [[ "${DEVICE}" == "gpu" ]] && DEVICE="cuda"
            shift 2 ;;
        --split)
            SPLIT="$2"
            shift 2 ;;
        --inspection)
            INSPECTION=1
            shift ;;
        --last)
            EVAL_LAST=1
            shift ;;
        --dry-run)
            DRY_RUN=1
            shift ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1 ;;
    esac
done

[[ -z "${DEVICE}" ]] && DEVICE="cuda"

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  CurveCodecAttn Evaluation  (quantized)"
echo "  Run ID     : ${RUN_ID}"
echo "  Config     : ${CONFIG}"
echo "  Ckpt dir   : ${CKPT_DIR}"
echo "  Split      : ${SPLIT}"
echo "  Device     : ${DEVICE}"
echo "  Inspection : $([ ${INSPECTION} -eq 1 ] && echo yes || echo no)"
echo "  Eval last  : $([ ${EVAL_LAST}  -eq 1 ] && echo yes || echo no)"
echo "  Dry run    : $([ ${DRY_RUN}    -eq 1 ] && echo yes || echo no)"
echo "  Date       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Helper ───────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
cd "${PROJECT_ROOT}"

run_eval() {
    local LABEL="$1"
    local CKPT="$2"
    local OUT_DIR="$3"

    echo "── ${LABEL} ──────────────────────────────────────────────"
    echo "  checkpoint : ${CKPT}"
    echo "  output     : ${OUT_DIR}"

    if [[ ! -f "${CKPT}" ]]; then
        echo "  SKIP — checkpoint not found"
        echo ""
        return 0
    fi

    CMD=(
        python eval/evaluate.py
        --checkpoint "${CKPT}"
        --config     "${CONFIG}"
        --output-dir "${OUT_DIR}"
        --device     "${DEVICE}"
        --split      "${SPLIT}"
    )
    [[ ${INSPECTION} -eq 1 ]] && CMD+=("--inspection")

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] ${CMD[*]}"
        echo ""
        return 0
    fi

    T_START=$(date +%s)
    EXIT_CODE=0
    "${CMD[@]}" 2>&1 | sed 's/^/  /' || EXIT_CODE=$?
    T_END=$(date +%s)

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "  ✗ failed (exit=${EXIT_CODE}, $(( T_END - T_START ))s)"
    else
        echo "  ✓ done ($(( T_END - T_START ))s)"
    fi
    echo ""
}

# ── Evaluate ─────────────────────────────────────────────────────────────────

run_eval "best checkpoint" \
    "${CKPT_DIR}/best.pt" \
    "${RESULTS_ROOT}/${RUN_ID}"

if [[ ${EVAL_LAST} -eq 1 ]]; then
    run_eval "last checkpoint" \
        "${CKPT_DIR}/last.pt" \
        "${RESULTS_ROOT}/${RUN_ID}_last"
fi

# ── Print summary ─────────────────────────────────────────────────────────────

if [[ ${DRY_RUN} -eq 0 ]]; then
    python - "${RESULTS_ROOT}" "${RUN_ID}" <<'PYEOF'
import json, sys
from pathlib import Path

results_root = Path(sys.argv[1])
run_id       = sys.argv[2]

def print_summary(label, run_dir):
    p = results_root / run_dir / "summary.json"
    if not p.exists():
        print(f"  {label}: no summary.json found")
        return
    with open(p) as f:
        s = json.load(f)
    adr_m  = s.get("area_difference_ratio_mean")
    adr_p  = s.get("area_difference_ratio_p95")
    cr_m   = s.get("compression_ratio_mean")
    n      = s.get("n_samples", "?")
    passed = s.get("targets_passed", {})
    print(f"  {label}")
    print(f"    n_samples : {n}")
    print(f"    ADR mean  : {adr_m:.6f}" if adr_m is not None else "    ADR mean  : —")
    print(f"    ADR p95   : {adr_p:.6f}" if adr_p is not None else "    ADR p95   : —")
    print(f"    CR  mean  : {cr_m:.4f}"  if cr_m  is not None else "    CR  mean  : N/A (no-quant path)")
    print(f"    ADR<=0.02 : {'PASS' if passed.get('area_difference_ratio_le002') else 'FAIL'}")

print()
print_summary("best.pt", run_id)
PYEOF
fi

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: ${RESULTS_ROOT}/${RUN_ID}/"
echo "============================================================"

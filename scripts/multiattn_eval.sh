#!/usr/bin/env bash
# multiattn_eval.sh — Evaluate CurveCodecMultiAttn checkpoint (D=8, c=16, B=8)
#
# Usage:
#   ./scripts/multiattn_eval.sh                      # cuda, validation split
#   ./scripts/multiattn_eval.sh --device cpu
#   ./scripts/multiattn_eval.sh --split test
#   ./scripts/multiattn_eval.sh --inspection         # write per-marker .oas files
#   ./scripts/multiattn_eval.sh --dry-run
#
# Outputs:
#   eval/results/multiattn_D8_c16_B8/results.csv
#   eval/results/multiattn_D8_c16_B8/summary.json
#   eval/results/multiattn_D8_c16_B8/inspection/*.oas  (--inspection only)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

RUN_ID="multiattn_D8_c16_B8"
CKPT="${PROJECT_ROOT}/checkpoints/${RUN_ID}/best.pt"
CFG="${PROJECT_ROOT}/train/config/multiattn_split/${RUN_ID}.yaml"
OUT_DIR="${PROJECT_ROOT}/eval/results/${RUN_ID}"

# ── Parse flags ───────────────────────────────────────────────────────────────

DEVICE=""
SPLIT="validation"
INSPECTION=0
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
        --inspection)
            INSPECTION=1
            shift
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

[[ -z "${DEVICE}" ]] && DEVICE="cuda"

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Print header ──────────────────────────────────────────────────────────────

echo "============================================================"
echo "  CurveCodecMultiAttn Evaluation"
echo "  run_id     : ${RUN_ID}"
echo "  checkpoint : ${CKPT}"
echo "  config     : ${CFG}"
echo "  output     : ${OUT_DIR}"
echo "  device     : ${DEVICE}"
echo "  split      : ${SPLIT}"
echo "  inspection : $([ ${INSPECTION} -eq 1 ] && echo yes || echo no)"
echo "  dry run    : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  Date       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

CMD=(
    python eval/evaluate.py
    --checkpoint "${CKPT}"
    --config     "${CFG}"
    --output-dir "${OUT_DIR}"
    --device     "${DEVICE}"
    --split      "${SPLIT}"
)
[[ ${INSPECTION} -eq 1 ]] && CMD+=("--inspection")

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}"
    exit 0
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR — checkpoint not found: ${CKPT}" >&2
    echo "Run ./scripts/multiattn_train.sh first." >&2
    exit 1
fi

if [[ ! -f "${CFG}" ]]; then
    echo "ERROR — config not found: ${CFG}" >&2
    exit 1
fi

# ── Evaluate ──────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

T_START=$(date +%s)
EXIT_CODE=0
"${CMD[@]}" || EXIT_CODE=$?
T_END=$(date +%s)
DURATION=$(( T_END - T_START ))

echo ""
echo "============================================================"
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "  ✗ Evaluation failed (exit=${EXIT_CODE}, ${DURATION}s)"
else
    echo "  ✓ Evaluation complete (${DURATION}s)"
    # Print key metrics and compare to prior c=16 results
    python - "${OUT_DIR}/summary.json" <<'PYEOF'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("  summary.json not found")
    sys.exit(0)

s = json.load(open(p))

cr   = s.get("compression_ratio_mean")
adr  = s.get("area_difference_ratio_mean")
adrp = s.get("area_difference_ratio_p95")
enc  = s.get("encode_ms_mean")
dec  = s.get("decode_ms_mean")
tp   = s.get("targets_passed", {})

print(f"  compression_ratio_mean       : {cr:.5f}  (target ≥10  → {'PASS' if cr and cr>=10 else 'FAIL'})")
print(f"  area_difference_ratio_mean   : {adr:.6f}  (target ≤0.02 → {'PASS' if adr and adr<=0.02 else 'FAIL'})")
print(f"  area_difference_ratio_p95    : {adrp:.6f}")
print(f"  encode_ms_mean               : {enc:.1f} ms")
print(f"  decode_ms_mean               : {dec:.1f} ms")
print()

# Comparison table
prior = {
    "attn_D8_c16_B8 (windowed w=7)": {"cr": 8.366, "adr": 0.02575},
    "sliding_D8_c16_B8 (sliding w=3)": {"cr": 8.278, "adr": 0.02685},
}
print(f"  {'run':<40}  {'CR':>8}  {'ADR mean':>10}")
print(f"  {'-'*62}")
for name, v in prior.items():
    print(f"  {name:<40}  {v['cr']:>8.3f}  {v['adr']:>10.6f}")
print(f"  {'multiattn_D8_c16_B8 (tiling, per-stage)':<40}  {cr:>8.3f}  {adr:>10.6f}")
PYEOF
fi
echo "  Results: ${OUT_DIR}"
echo "  Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
exit ${EXIT_CODE}

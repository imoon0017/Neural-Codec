#!/usr/bin/env bash
# bypass_eval.sh — Bypass round-trip reference evaluation (no codec)
#
# Measures the theoretical minimum ADR from the rasterize→marching-squares
# step alone.  No checkpoint is required.  Use the reported ADR as the
# floor against which trained-model results should be compared.
#
# Usage:
#   ./scripts/bypass_eval.sh                         # test split, default config
#   ./scripts/bypass_eval.sh --split validation
#   ./scripts/bypass_eval.sh --config train/config/multiattn_split/multiattn_D8_c16_B8.yaml
#   ./scripts/bypass_eval.sh --output-dir eval/results/my_bypass/
#   ./scripts/bypass_eval.sh --inspection            # write per-marker .oas files
#   ./scripts/bypass_eval.sh --dry-run
#
# --config reads only the [csdf] and [dataset] sections (grid resolution,
# layer numbers, cache_dir / manifest).  The [model] section is not used.
#
# Outputs:
#   eval/results/bypass_reference/results.csv
#   eval/results/bypass_reference/summary.json
#   eval/results/bypass_reference/inspection/*.oas   (--inspection only)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

CFG="${PROJECT_ROOT}/train/config/baseline.yaml"
OUT_DIR="${PROJECT_ROOT}/eval/results/bypass_reference"

# ── Parse flags ───────────────────────────────────────────────────────────────

SPLIT="test"
INSPECTION=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CFG="$(realpath "$2")"
            shift 2
            ;;
        --output-dir)
            OUT_DIR="$(realpath "$2")"
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
            echo "Usage: $0 [--config <yaml>] [--output-dir <dir>] [--split test|validation|train] [--inspection] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Validate ─────────────────────────────────────────────────────────────────

if [[ ! -f "${CFG}" ]]; then
    echo "ERROR — config not found: ${CFG}" >&2
    exit 1
fi

# ── Print header ──────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Bypass Round-Trip Reference Evaluation"
echo "  config     : ${CFG}"
echo "  output     : ${OUT_DIR}"
echo "  split      : ${SPLIT}"
echo "  inspection : $([ ${INSPECTION} -eq 1 ] && echo yes || echo no)"
echo "  dry run    : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  date       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

CMD=(
    python eval/evaluate_bypass.py
    --config     "${CFG}"
    --output-dir "${OUT_DIR}"
    --split      "${SPLIT}"
)
[[ ${INSPECTION} -eq 1 ]] && CMD+=("--inspection")

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}"
    exit 0
fi

# ── Run ───────────────────────────────────────────────────────────────────────

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
    echo "  ✓ Bypass evaluation complete (${DURATION}s)"
    python - "${OUT_DIR}/summary.json" <<'PYEOF'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("  summary.json not found")
    sys.exit(0)

s = json.load(open(p))

adr  = s.get("area_difference_ratio_mean")
adrp = s.get("area_difference_ratio_p95")
adrx = s.get("area_difference_ratio_max")
cms  = s.get("contour_ms_mean")
n    = s.get("n_samples", 0)

print(f"  n_samples                    : {n}")
print(f"  area_difference_ratio_mean   : {adr:.6f}  ← bypass floor")
print(f"  area_difference_ratio_p95    : {adrp:.6f}")
print(f"  area_difference_ratio_max    : {adrx:.6f}")
print(f"  contour_ms_mean              : {cms:.2f} ms  (marching squares only)")
print()
print("  NOTE: ADR above is the rasterize→marching-squares floor.")
print("  Trained models should produce ADR close to this value.")
PYEOF
fi
echo "  Results: ${OUT_DIR}"
echo "  Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
exit ${EXIT_CODE}

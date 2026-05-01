#!/usr/bin/env bash
# sliding_split_test.sh — Train CurveCodecAttn with sliding-window attention
#
# Fixed:    arch=curve_codec_attn, attn_type=sliding, attn_window_size=3,
#           attn_heads=4, encoder_channels=64, latent_dim=8, quantizer_bits=8,
#           batch_size=1, epochs=200
# Variable: compaction_ratio  (8, 16)
#           Total: 2 combinations
#
# Usage:
#   ./scripts/sliding_split_test.sh                 # cuda
#   ./scripts/sliding_split_test.sh --device cpu
#   ./scripts/sliding_split_test.sh --dry-run       # print commands only
#   ./scripts/sliding_split_test.sh --resume        # resume from last.pt
#
# Outputs:
#   checkpoints/sliding_D8_c{c}_B8/train.log        per-run log
#   checkpoints/sliding_D8_c{c}_B8/best.pt          best checkpoint
#   checkpoints/sliding_split_summary.json           results table

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

SPLIT_CONFIG_DIR="${PROJECT_ROOT}/train/config/sliding_split"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
SUMMARY_PATH="${CHECKPOINT_ROOT}/sliding_split_summary.json"
RESULTS_FILE="${CHECKPOINT_ROOT}/.sliding_split_results.jsonl"

# Fixed axes
LATENT_DIM=8
QUANTIZER_BITS=8

# Variable axis
COMPACTION_RATIOS=(8 16)

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

# ── Print run header ──────────────────────────────────────────────────────────

TOTAL=${#COMPACTION_RATIOS[@]}

echo "============================================================"
echo "  CurveCodecAttn Sliding-Window Split Test"
echo "  Device            : ${DEVICE}"
echo "  Dry run           : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  Resume            : $([ ${RESUME} -eq 1 ] && echo yes || echo no)"
echo "  arch              : curve_codec_attn  (fixed)"
echo "  attn_type         : sliding  (fixed)"
echo "  attn_window_size  : 3  (fixed)"
echo "  latent_dim        : ${LATENT_DIM}  (fixed)"
echo "  quantizer_bits    : ${QUANTIZER_BITS}  (fixed)"
echo "  compaction_ratio  : ${COMPACTION_RATIOS[*]}"
echo "  Total combinations: ${TOTAL}"
echo "  Date              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Train sequentially ────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

mkdir -p "${CHECKPOINT_ROOT}"
: > "${RESULTS_FILE}"

IDX=0

for c in "${COMPACTION_RATIOS[@]}"; do
    IDX=$(( IDX + 1 ))
    RUN_ID="sliding_D${LATENT_DIM}_c${c}_B${QUANTIZER_BITS}"
    CFG_PATH="${SPLIT_CONFIG_DIR}/${RUN_ID}.yaml"
    CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
    LOG_PATH="${CKPT_DIR}/train.log"

    echo "── [${IDX}/${TOTAL}] D=${LATENT_DIM}  c=${c}  B=${QUANTIZER_BITS} ──────────────────────────────"

    if [[ ! -f "${CFG_PATH}" ]]; then
        echo "  ERROR — config not found: ${CFG_PATH}" >&2
        echo ""
        continue
    fi

    echo "  config : ${CFG_PATH}"
    echo "  log    : ${LOG_PATH}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        TRAIN_CMD="python train/train.py --config ${CFG_PATH} --device ${DEVICE}"
        [[ ${RESUME} -eq 1 ]] && TRAIN_CMD+=" --resume"
        echo "  [dry-run] ${TRAIN_CMD}"
        echo ""
        continue
    fi

    mkdir -p "${CKPT_DIR}"

    TRAIN_ARGS=("--config" "${CFG_PATH}" "--device" "${DEVICE}")
    [[ ${RESUME} -eq 1 ]] && TRAIN_ARGS+=("--resume")

    T_START=$(date +%s)
    EXIT_CODE=0

    cd "${PROJECT_ROOT}"
    python train/train.py "${TRAIN_ARGS[@]}" > "${LOG_PATH}" 2>&1 || EXIT_CODE=$?

    T_END=$(date +%s)
    DURATION=$(( T_END - T_START ))
    STATUS="success"
    [[ ${EXIT_CODE} -ne 0 ]] && STATUS="failed"

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "  ✗ exit=${EXIT_CODE}  (${DURATION}s) — see ${LOG_PATH}"
    else
        echo "  ✓ done  (${DURATION}s)"
    fi

    python - "${CKPT_DIR}/best.pt" "${RUN_ID}" \
             "${LATENT_DIM}" "${c}" "${QUANTIZER_BITS}" \
             "${STATUS}" "${EXIT_CODE}" "${DURATION}" \
             "${RESULTS_FILE}" \
        <<'PYEOF'
import json, sys, torch
from pathlib import Path

best_pt      = Path(sys.argv[1])
run_id       = sys.argv[2]
D, c, B      = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
status       = sys.argv[6]
exit_code    = int(sys.argv[7])
duration     = int(sys.argv[8])
results_file = Path(sys.argv[9])

best_val_loss = None
if best_pt.exists():
    try:
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=True)
        best_val_loss = float(ckpt.get("best_val_loss", float("nan")))
    except Exception:
        pass

record = {
    "run_id": run_id, "D": D, "c": c, "B": B,
    "status": status, "exit_code": exit_code,
    "duration_s": duration, "best_val_loss": best_val_loss,
    "config": f"train/config/sliding_split/{run_id}.yaml",
    "log": f"checkpoints/{run_id}/train.log",
}
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
PYEOF

    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────

if [[ ${DRY_RUN} -eq 0 ]]; then
    python - "${RESULTS_FILE}" "${SUMMARY_PATH}" "${TOTAL}" <<'PYEOF'
import json, sys
from pathlib import Path

results_file = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
total        = int(sys.argv[3])

results = [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()]
summary = {
    "total":     total,
    "completed": len(results),
    "success":   sum(1 for r in results if r["status"] == "success"),
    "failed":    sum(1 for r in results if r["status"] != "success"),
    "runs":      results,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

header = f"{'run_id':<28}  {'D':>3}  {'c':>3}  {'B':>3}  {'status':<8}  {'best_val_loss':>14}  {'duration':>10}"
sep    = "-" * len(header)
print(sep)
print(header)
print(sep)
for r in results:
    bvl = f"{r['best_val_loss']:.6f}" if r.get("best_val_loss") is not None else "—"
    dur = f"{r['duration_s']}s"
    print(f"{r['run_id']:<28}  {r['D']:>3}  {r['c']:>3}  {r['B']:>3}  {r['status']:<8}  {bvl:>14}  {dur:>10}")
print(sep)
success = sum(1 for r in results if r["status"] == "success")
print(f"{success}/{len(results)} runs succeeded")
print(f"Summary -> {summary_path}")
PYEOF
fi

echo ""
echo "============================================================"
echo "  Sliding split test complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Summary: ${SUMMARY_PATH}"
echo "============================================================"

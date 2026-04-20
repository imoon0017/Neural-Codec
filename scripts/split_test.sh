#!/usr/bin/env bash
# split_test.sh — Split-test CurveCodec across model configurations
#
# Generates one YAML config per combination of (latent_dim, compaction_ratio,
# quantizer_bits), then trains each sequentially.  All other hyperparameters
# are inherited from a base config.
#
# Split conditions
#   latent_dim        : 32  16  8
#   compaction_ratio  :  8  16  32
#   quantizer_bits    : 16   8
#   Total             : 18 combinations
#
# Usage:
#   ./scripts/split_test.sh                                # default base config, cuda
#   ./scripts/split_test.sh --base-config path/to/cfg.yaml
#   ./scripts/split_test.sh --device cpu
#   ./scripts/split_test.sh --dry-run                      # generate configs only
#   ./scripts/split_test.sh --resume                       # resume each run from last.pt
#
# Outputs:
#   train/config/split_test/split_D{D}_c{c}_B{B}.yaml     generated configs
#   checkpoints/split_D{D}_c{c}_B{B}/train.log             per-run log
#   checkpoints/split_D{D}_c{c}_B{B}/best.pt               best checkpoint
#   checkpoints/split_test_summary.json                    final results table
#
# Note: compaction_ratio=32 requires the cached patch size S to be divisible
# by 32.  Failing runs are logged and skipped; remaining runs continue.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

SPLIT_CONFIG_DIR="${PROJECT_ROOT}/train/config/split_test"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
SUMMARY_PATH="${CHECKPOINT_ROOT}/split_test_summary.json"
RESULTS_FILE="${CHECKPOINT_ROOT}/.split_test_results.jsonl"

LATENT_DIMS=(32 16 8)
COMPACTION_RATIOS=(8 16 32)
QUANTIZER_BITS=(16 8)

# ── Parse flags ───────────────────────────────────────────────────────────────

BASE_CONFIG="${PROJECT_ROOT}/train/config/baseline.yaml"
DEVICE=""
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-config)
            BASE_CONFIG="$(realpath "$2")"
            shift 2
            ;;
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

if [[ -z "${DEVICE}" ]]; then
    DEVICE="cuda"
fi

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Validate base config ──────────────────────────────────────────────────────

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "ERROR: base config not found: ${BASE_CONFIG}" >&2
    exit 1
fi

# ── Print run header ──────────────────────────────────────────────────────────

TOTAL=$(( ${#LATENT_DIMS[@]} * ${#COMPACTION_RATIOS[@]} * ${#QUANTIZER_BITS[@]} ))

echo "============================================================"
echo "  CurveCodec Split Test"
echo "  Base config       : ${BASE_CONFIG}"
echo "  Device            : ${DEVICE}"
echo "  Dry run           : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  Resume            : $([ ${RESUME} -eq 1 ] && echo yes || echo no)"
echo "  latent_dim        : ${LATENT_DIMS[*]}"
echo "  compaction_ratio  : ${COMPACTION_RATIOS[*]}"
echo "  quantizer_bits    : ${QUANTIZER_BITS[*]}"
echo "  Total combinations: ${TOTAL}"
echo "  Date              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Phase 1: generate configs ─────────────────────────────────────────────────

echo "Phase 1 — generating ${TOTAL} config files …"
mkdir -p "${SPLIT_CONFIG_DIR}"

python - \
    "${PROJECT_ROOT}" \
    "${BASE_CONFIG}" \
    "${SPLIT_CONFIG_DIR}" \
    "${LATENT_DIMS[*]}" \
    "${COMPACTION_RATIOS[*]}" \
    "${QUANTIZER_BITS[*]}" \
    <<'PYEOF'
import copy, sys, yaml
from itertools import product
from pathlib import Path

project_root      = Path(sys.argv[1])
base_config       = Path(sys.argv[2])
split_cfg_dir     = Path(sys.argv[3])
latent_dims       = [int(x) for x in sys.argv[4].split()]
compaction_ratios = [int(x) for x in sys.argv[5].split()]
quantizer_bits    = [int(x) for x in sys.argv[6].split()]

with open(base_config) as f:
    base_cfg = yaml.safe_load(f)

for D, c, B in product(latent_dims, compaction_ratios, quantizer_bits):
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["latent_dim"]       = D
    cfg["model"]["compaction_ratio"] = c
    cfg["model"]["quantizer_bits"]   = B
    cfg["paths"]["run_id"]           = f"split_D{D}_c{c}_B{B}"

    out = split_cfg_dir / f"split_D{D}_c{c}_B{B}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {out.relative_to(project_root)}")

print(f"Generated {len(latent_dims) * len(compaction_ratios) * len(quantizer_bits)} configs.")
PYEOF

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo ""
    echo "--dry-run: configs generated, skipping training."
    exit 0
fi

# ── Phase 2: train sequentially ───────────────────────────────────────────────

echo ""
echo "Phase 2 — training ${TOTAL} configurations sequentially …"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

mkdir -p "${CHECKPOINT_ROOT}"
# Start fresh results file for this run
: > "${RESULTS_FILE}"

IDX=0

for D in "${LATENT_DIMS[@]}"; do
for c in "${COMPACTION_RATIOS[@]}"; do
for B in "${QUANTIZER_BITS[@]}"; do
    IDX=$(( IDX + 1 ))
    RUN_ID="split_D${D}_c${c}_B${B}"
    CFG_PATH="${SPLIT_CONFIG_DIR}/${RUN_ID}.yaml"
    CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
    LOG_PATH="${CKPT_DIR}/train.log"

    mkdir -p "${CKPT_DIR}"

    echo "── [${IDX}/${TOTAL}] D=${D}  c=${c}  B=${B} ──────────────────────────"
    echo "  config : ${CFG_PATH}"
    echo "  log    : ${LOG_PATH}"

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

    # Append one JSON line to the results file
    python - "${CKPT_DIR}/best.pt" "${RUN_ID}" "${D}" "${c}" "${B}" \
             "${STATUS}" "${EXIT_CODE}" "${DURATION}" \
             "${RESULTS_FILE}" \
             "train/config/split_test/${RUN_ID}.yaml" \
             "checkpoints/${RUN_ID}/train.log" \
        <<'PYEOF'
import json, sys, torch
from pathlib import Path

best_pt    = Path(sys.argv[1])
run_id     = sys.argv[2]
D, c, B    = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
status     = sys.argv[6]
exit_code  = int(sys.argv[7])
duration   = int(sys.argv[8])
results_file = Path(sys.argv[9])
config_rel = sys.argv[10]
log_rel    = sys.argv[11]

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
    "config": config_rel, "log": log_rel,
}
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")
PYEOF

    # Rolling summary JSON from the accumulated results file
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
PYEOF

    echo ""
done
done
done

# ── Summary table ─────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Split-test complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Summary: ${SUMMARY_PATH}"
echo "============================================================"

python - "${RESULTS_FILE}" <<'PYEOF'
import json, sys
from pathlib import Path

results = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
header = f"{'run_id':<30}  {'status':<8}  {'best_val_loss':>14}  {'duration':>10}"
sep    = "-" * len(header)
print(sep)
print(header)
print(sep)
for r in results:
    bvl = f"{r['best_val_loss']:.6f}" if r["best_val_loss"] is not None else "—"
    dur = f"{r['duration_s']}s"
    print(f"{r['run_id']:<30}  {r['status']:<8}  {bvl:>14}  {dur:>10}")
print(sep)
success = sum(1 for r in results if r["status"] == "success")
print(f"{success}/{len(results)} runs succeeded")
PYEOF

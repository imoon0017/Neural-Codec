#!/usr/bin/env bash
# rasterize.sh — Rebuild the dataset cache (.npy files)
#
# Uses marker_margin_nm=188 so the canvas is exactly 896px at 6nm/px:
#   (5000 + 2×188) / 6 = 5376 / 6 = 896  (exact integer, no rounding)
#   896 is divisible by 8, 16, and 32.
#
# Usage:
#   ./scripts/rasterize.sh                    # all splits, 4 workers
#   ./scripts/rasterize.sh --splits train     # single split
#   ./scripts/rasterize.sh --workers 8
#   ./scripts/rasterize.sh --dry-run
#
# NOTE: dataset/catalog.csv is NOT updated by this script.
#       After rasterization, re-run ingest.py with the appropriate
#       compaction_ratio config to update patch_size_px in the catalog.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

CONFIG_OUT="${PROJECT_ROOT}/train/config/rasterize.yaml"
BASE_CONFIG="${PROJECT_ROOT}/train/config/attn_v1.yaml"
MARKER_MARGIN_NM=188

# ── Parse flags ───────────────────────────────────────────────────────────────

SPLITS="train validation test"
WORKERS=4
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --splits)   SPLITS="$2";  shift 2 ;;
        --workers)  WORKERS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1;    shift   ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Header ───────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Dataset Rasterization"
echo "  marker_margin_nm : ${MARKER_MARGIN_NM} nm"
echo "  canvas           : 896 px  (5376 nm / 6 nm·px⁻¹, exact)"
echo "  config           : ${CONFIG_OUT}"
echo "  splits           : ${SPLITS}"
echo "  workers          : ${WORKERS}"
echo "  dry run          : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  date             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Generate config ───────────────────────────────────────────────────────────

python - "${BASE_CONFIG}" "${CONFIG_OUT}" "${MARKER_MARGIN_NM}" <<'PYEOF'
import copy, sys, yaml
from pathlib import Path

base_config      = Path(sys.argv[1])
config_out       = Path(sys.argv[2])
marker_margin_nm = int(sys.argv[3])

with open(base_config) as f:
    cfg = yaml.safe_load(f)

cfg["csdf"]["marker_margin_nm"] = marker_margin_nm
cfg["paths"]["run_id"]          = "rasterize"

with open(config_out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"Config written to {config_out}")
PYEOF

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "--dry-run: config generated, skipping rasterization."
    exit 0
fi

# ── Rasterize ─────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"
T_START=$(date +%s)

# shellcheck disable=SC2086
python dataset/rasterize.py \
    --config  "${CONFIG_OUT}" \
    --splits  ${SPLITS} \
    --workers "${WORKERS}" \
    --force

T_END=$(date +%s)

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')  ($(( T_END - T_START ))s)"
echo "  NOTE: catalog patch_size_px is NOT updated by this script."
echo "        Re-run ingest.py to update it."
echo "============================================================"

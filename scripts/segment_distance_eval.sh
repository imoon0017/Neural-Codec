#!/usr/bin/env bash
# segment_distance_eval.sh — Segment-center boundary-distance metric over eval results
#
# For every segment in a reconstructed PWCL layer, measures the nearest
# Euclidean distance (nm) from the segment's centre point to the original
# polygon boundary, then reports statistics (mean, std, median, p95, p99).
#
# Automatically detects the output layout:
#   standard   — <results-dir>/original/ + <results-dir>/reconstructed/
#   correction — <results-dir>/original/ + <results-dir>/baseline/reconstructed/
#                                        + <results-dir>/corrected/reconstructed/
#
# With no --run-id the script iterates over every subdirectory of eval/results/
# and skips those whose output summary already exists (pass --force to re-run).
#
# Usage:
#   ./scripts/segment_distance_eval.sh
#   ./scripts/segment_distance_eval.sh --run-id attn_D8_c16_B8
#   ./scripts/segment_distance_eval.sh --run-id correction_attn_D8_c16_B8_a0p2_n5
#   ./scripts/segment_distance_eval.sh --mask-layer 1 --jobs 4
#   ./scripts/segment_distance_eval.sh --spec 2.0
#   ./scripts/segment_distance_eval.sh --spec 2.0 --violation-marker-size-nm 4.0
#   ./scripts/segment_distance_eval.sh --force
#   ./scripts/segment_distance_eval.sh --dry-run
#
# Defaults:
#   run-id                  (all subdirectories of eval/results/)
#   mask-layer              1
#   jobs                    1
#   spec                    (none — no violation markers written)
#   violation-marker-size   2.0 nm
#
# Outputs (per results directory):
#   segment_distance_summary.json       standard mode
#   segment_distance_summary.csv
#   segment_distance_baseline.json      correction mode
#   segment_distance_baseline.csv
#   segment_distance_corrected.json
#   segment_distance_corrected.csv
#   segment_distance.log                per-directory run log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

# ── Defaults ──────────────────────────────────────────────────────────────────

RUN_ID=""
MASK_LAYER=1
JOBS=1
FORCE=0
DRY_RUN=0
SPEC=""
VIOLATION_MARKER_SIZE="2.0"

# ── Parse flags ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)                   RUN_ID="$2";               shift 2 ;;
        --mask-layer)               MASK_LAYER="$2";            shift 2 ;;
        --jobs)                     JOBS="$2";                  shift 2 ;;
        --spec)                     SPEC="$2";                  shift 2 ;;
        --violation-marker-size-nm) VIOLATION_MARKER_SIZE="$2"; shift 2 ;;
        --force)                    FORCE=1;                    shift   ;;
        --dry-run)                  DRY_RUN=1;                  shift   ;;
        --help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--run-id ID] [--mask-layer N] [--jobs N] [--spec NM] [--violation-marker-size-nm NM] [--force] [--dry-run]" >&2
            exit 1 ;;
    esac
done

RESULTS_ROOT="${PROJECT_ROOT}/eval/results"
EVAL_SCRIPT="${PROJECT_ROOT}/eval/evaluate.py"

# ── Activate conda ────────────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── Collect directories ───────────────────────────────────────────────────────

if [[ -n "${RUN_ID}" ]]; then
    DIRS=("${RESULTS_ROOT}/${RUN_ID}")
    if [[ ! -d "${DIRS[0]}" ]]; then
        echo "ERROR — results directory not found: ${DIRS[0]}" >&2
        exit 1
    fi
else
    mapfile -t DIRS < <(find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

# ── Mode detection ────────────────────────────────────────────────────────────

detect_mode() {
    local dir="$1"
    if [[ -d "${dir}/baseline/reconstructed" && \
          -d "${dir}/corrected/reconstructed" && \
          -d "${dir}/original" ]]; then
        echo "correction"
    elif [[ -d "${dir}/original" && -d "${dir}/reconstructed" ]]; then
        echo "standard"
    else
        echo "unknown"
    fi
}

sentinel_path() {
    local dir="$1" mode="$2"
    if [[ "${mode}" == "correction" ]]; then
        echo "${dir}/segment_distance_corrected.json"
    else
        echo "${dir}/segment_distance_summary.json"
    fi
}

# ── Print header ──────────────────────────────────────────────────────────────

TOTAL=${#DIRS[@]}
echo "============================================================"
echo "  Segment-Distance Evaluation"
echo "  results root  : ${RESULTS_ROOT}"
echo "  run_id        : ${RUN_ID:-<all>}"
echo "  mask_layer    : ${MASK_LAYER}"
echo "  spec (nm)     : ${SPEC:-<none>}"
echo "  marker size   : ${VIOLATION_MARKER_SIZE} nm"
echo "  jobs          : ${JOBS}"
echo "  force         : $([ ${FORCE} -eq 1 ] && echo yes || echo no)"
echo "  directories   : ${TOTAL}"
echo "  dry run       : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  date          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Per-directory runner ──────────────────────────────────────────────────────

_spec_flags() {
    if [[ -n "${SPEC}" ]]; then
        echo "--spec ${SPEC} --violation-marker-size-nm ${VIOLATION_MARKER_SIZE}"
    fi
}

run_one() {
    local dir="$1"
    local name
    name="$(basename "${dir}")"
    local mode
    mode="$(detect_mode "${dir}")"
    local sentinel
    sentinel="$(sentinel_path "${dir}" "${mode}")"

    if [[ "${mode}" == "unknown" ]]; then
        echo "[$(date '+%H:%M:%S')] –  ${name}  (no recognised output structure — skipping)"
        return 0
    fi

    if [[ ${FORCE} -eq 0 && -f "${sentinel}" ]]; then
        echo "[$(date '+%H:%M:%S')] –  ${name}  [${mode}] (already done — use --force to re-run)"
        return 0
    fi

    local spec_flags
    spec_flags="$(_spec_flags)"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        if [[ "${mode}" == "correction" ]]; then
            echo "[dry-run]  python eval/evaluate.py --correction-results-dir ${dir} --mask-layer ${MASK_LAYER}${spec_flags:+ ${spec_flags}}"
        else
            echo "[dry-run]  python eval/evaluate.py --results-dir ${dir} --mask-layer ${MASK_LAYER}${spec_flags:+ ${spec_flags}}"
        fi
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START  ${name}  [${mode}]"
    local t0
    t0=$(date +%s)
    local log_file="${dir}/segment_distance.log"

    if [[ "${mode}" == "correction" ]]; then
        # shellcheck disable=SC2086
        python "${EVAL_SCRIPT}" \
            --correction-results-dir "${dir}" \
            --mask-layer "${MASK_LAYER}" \
            ${spec_flags} \
            2>&1 | tee "${log_file}"
    else
        # shellcheck disable=SC2086
        python "${EVAL_SCRIPT}" \
            --results-dir "${dir}" \
            --mask-layer "${MASK_LAYER}" \
            ${spec_flags} \
            2>&1 | tee "${log_file}"
    fi
    local exit_code=${PIPESTATUS[0]}

    local elapsed=$(( $(date +%s) - t0 ))
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] ✓  ${name}  [${elapsed}s]"
        _print_summary "${dir}" "${mode}"
    else
        echo "[$(date '+%H:%M:%S')] ✗  ${name}  (exit=${exit_code}, ${elapsed}s — see ${log_file})" >&2
        return 1
    fi
}

_print_summary() {
    local dir="$1" mode="$2"
    local json
    if [[ "${mode}" == "correction" ]]; then
        # Print side-by-side from the two JSON files via the script's stdout
        # (already printed by evaluate.py to the log; echo key lines)
        local bj="${dir}/segment_distance_baseline.json"
        local cj="${dir}/segment_distance_corrected.json"
        [[ -f "${bj}" && -f "${cj}" ]] || return 0
        python - "${bj}" "${cj}" <<'PYEOF'
import json, sys
b = json.load(open(sys.argv[1]))
c = json.load(open(sys.argv[2]))
keys   = ["mean_nm", "median_nm", "p95_nm", "p99_nm"]
labels = ["mean  ", "median", "p95   ", "p99   "]
print(f"  {'metric':<10}  {'baseline':>10}  {'corrected':>10}  {'delta':>10}")
print("  " + "─" * 46)
for label, key in zip(labels, keys):
    bv = b.get(key, float("nan"))
    cv = c.get(key, float("nan"))
    delta = cv - bv
    sign = "+" if delta > 0 else ""
    print(f"  {label}  {bv:>10.4f}  {cv:>10.4f}  {sign}{delta:>9.4f} nm")
PYEOF
    else
        json="${dir}/segment_distance_summary.json"
        [[ -f "${json}" ]] || return 0
        python - "${json}" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
for key, label in [("mean_nm","mean"),("median_nm","median"),("p95_nm","p95"),("p99_nm","p99")]:
    v = s.get(key, float("nan"))
    print(f"  {label:<8} {v:.4f} nm")
if "spec_nm" in s:
    vf = s.get("violation_fraction_mean", 0.0) * 100.0
    print(f"  spec     {s['spec_nm']:.4f} nm  violations={s.get('n_violations_total',0)}  ({vf:.1f}% mean)")
PYEOF
    fi
}

export -f run_one detect_mode sentinel_path _print_summary _spec_flags
export FORCE DRY_RUN MASK_LAYER EVAL_SCRIPT SPEC VIOLATION_MARKER_SIZE

# ── Run ───────────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"

ERRORS=0
if [[ "${JOBS}" -gt 1 && ${DRY_RUN} -eq 0 ]]; then
    printf '%s\n' "${DIRS[@]}" | \
        xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {} || ERRORS=$?
else
    for dir in "${DIRS[@]}"; do
        run_one "${dir}" || ERRORS=$((ERRORS + 1))
    done
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
if [[ ${ERRORS} -eq 0 ]]; then
    echo "  ✓ All done.  ${TOTAL} director$([ "${TOTAL}" -eq 1 ] && echo y || echo ies) processed."
else
    echo "  ✗ Done with ${ERRORS} error(s).  Check per-directory segment_distance.log files."
fi
echo "  Date : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
[[ ${ERRORS} -eq 0 ]]

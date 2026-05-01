#!/usr/bin/env bash
# attn_split_eval.sh — Evaluate all CurveCodecAttn split-test checkpoints
#
# Runs eval/evaluate.py sequentially for each attn_D{D}_c{c}_B{B} checkpoint
# produced by attn_split_test.sh.
#
# Fixed:    arch=curve_codec_attn, attn_heads=4, attn_window_size=7,
#           encoder_channels=64, batch_size=1
# Variable: latent_dim (8, 16, 32) × compaction_ratio (8, 16, 32) × quantizer_bits (8, 16)
#           Total: 18 combinations (skips any missing checkpoint / config)
#
# Usage:
#   ./scripts/attn_split_eval.sh                      # default: cuda, validation split
#   ./scripts/attn_split_eval.sh --device cpu
#   ./scripts/attn_split_eval.sh --split test
#   ./scripts/attn_split_eval.sh --inspection         # write per-marker .oas files
#   ./scripts/attn_split_eval.sh --dry-run            # print commands, don't execute
#
# Outputs (per run):
#   eval/results/attn_D{D}_c{c}_B{B}/results.csv
#   eval/results/attn_D{D}_c{c}_B{B}/summary.json
#   eval/results/attn_D{D}_c{c}_B{B}/inspection/*.oas  (--inspection only)
#   eval/results/attn_split_eval_summary.json           aggregate comparison

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="neural_codec"

CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
RESULTS_ROOT="${PROJECT_ROOT}/eval/results"
SPLIT_CONFIG_DIR="${PROJECT_ROOT}/train/config/attn_split"
AGGREGATE_SUMMARY="${RESULTS_ROOT}/attn_split_eval_summary.json"

LATENT_DIMS=(8 16 32)
COMPACTION_RATIOS=(8 16 32)
QUANTIZER_BITS_LIST=(8 16)

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

TOTAL=$(( ${#LATENT_DIMS[@]} * ${#COMPACTION_RATIOS[@]} * ${#QUANTIZER_BITS_LIST[@]} ))

echo "============================================================"
echo "  CurveCodecAttn Split-Test Evaluation"
echo "  Device            : ${DEVICE}"
echo "  Split             : ${SPLIT}"
echo "  Inspection        : $([ ${INSPECTION} -eq 1 ] && echo yes || echo no)"
echo "  Dry run           : $([ ${DRY_RUN} -eq 1 ] && echo yes || echo no)"
echo "  latent_dim        : ${LATENT_DIMS[*]}"
echo "  compaction_ratio  : ${COMPACTION_RATIOS[*]}"
echo "  quantizer_bits    : ${QUANTIZER_BITS_LIST[*]}"
echo "  Total runs        : ${TOTAL}"
echo "  Date              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Evaluate sequentially ─────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"
cd "${PROJECT_ROOT}"

IDX=0
N_OK=0
N_SKIP=0
N_FAIL=0

for D in "${LATENT_DIMS[@]}"; do
for c in "${COMPACTION_RATIOS[@]}"; do
for B in "${QUANTIZER_BITS_LIST[@]}"; do
    IDX=$(( IDX + 1 ))
    RUN_ID="attn_D${D}_c${c}_B${B}"
    CKPT="${CHECKPOINT_ROOT}/${RUN_ID}/best.pt"
    CFG="${SPLIT_CONFIG_DIR}/${RUN_ID}.yaml"
    OUT_DIR="${RESULTS_ROOT}/${RUN_ID}"

    echo "── [${IDX}/${TOTAL}] D=${D}  c=${c}  B=${B}  (${RUN_ID}) ──────────────"

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
    )
    [[ ${INSPECTION} -eq 1 ]] && CMD+=("--inspection")

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

# ── Aggregate summary + comparison table ──────────────────────────────────────

if [[ ${DRY_RUN} -eq 0 ]]; then
    python - \
        "${RESULTS_ROOT}" \
        "${AGGREGATE_SUMMARY}" \
        "${LATENT_DIMS[*]}" \
        "${COMPACTION_RATIOS[*]}" \
        "${QUANTIZER_BITS_LIST[*]}" \
        <<'PYEOF'
import json, sys
from pathlib import Path

results_root      = Path(sys.argv[1])
agg_path          = Path(sys.argv[2])
latent_dims       = [int(x) for x in sys.argv[3].split()]
compaction_ratios = [int(x) for x in sys.argv[4].split()]
quantizer_bits_l  = [int(x) for x in sys.argv[5].split()]

def load_summary(run_id):
    p = results_root / run_id / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def extract(s):
    if s is None:
        return {}
    return {
        "n_samples":         s.get("n_samples"),
        "adr_mean":          s.get("area_difference_ratio_mean"),
        "adr_p95":           s.get("area_difference_ratio_p95"),
        "cr_mean":           s.get("compression_ratio_mean"),
        "encode_ms_mean":    s.get("encode_ms_mean"),
        "decode_ms_mean":    s.get("decode_ms_mean"),
        "target_adr_passed": s.get("targets_passed", {}).get("area_difference_ratio_le002"),
        "target_cr_passed":  s.get("targets_passed", {}).get("compression_ratio_ge10"),
    }

runs = []
for D in latent_dims:
    for c in compaction_ratios:
        for B in quantizer_bits_l:
            run_id = f"attn_D{D}_c{c}_B{B}"
            s = load_summary(run_id)
            entry = {
                "run_id": run_id, "D": D, "c": c, "B": B,
                "status": "ok" if s else "missing",
            }
            entry.update(extract(s))
            runs.append(entry)

with open(agg_path, "w") as f:
    json.dump({"runs": runs}, f, indent=2)

COL_W = 28
def fmt(v, fmt_str):
    return format(v, fmt_str) if v is not None else "—"

header = (
    f"{'run_id':<{COL_W}}  {'D':>3}  {'c':>3}  {'B':>3}  {'n':>5}  "
    f"{'ADR_mean':>10}  {'ADR_p95':>10}  {'CR_mean':>8}  "
    f"{'enc_ms':>8}  {'dec_ms':>8}  {'ADR<=0.02':>10}  {'CR>=10':>8}"
)
sep = "-" * len(header)
print()
print(sep)
print(header)
print(sep)

prev_D = None
for r in runs:
    if r["D"] != prev_D and prev_D is not None:
        print()
    prev_D = r["D"]

    if r["status"] == "ok":
        adr_ok = "PASS" if r.get("target_adr_passed") else ("FAIL" if r.get("target_adr_passed") is False else "—")
        cr_ok  = "PASS" if r.get("target_cr_passed")  else ("FAIL" if r.get("target_cr_passed")  is False else "—")
        print(
            f"{r['run_id']:<{COL_W}}  {r['D']:>3}  {r['c']:>3}  {r['B']:>3}  {fmt(r.get('n_samples'), '>5')}  "
            f"{fmt(r.get('adr_mean'), '10.6f')}  {fmt(r.get('adr_p95'), '10.6f')}  "
            f"{fmt(r.get('cr_mean'), '8.2f')}  "
            f"{fmt(r.get('encode_ms_mean'), '8.1f')}  {fmt(r.get('decode_ms_mean'), '8.1f')}  "
            f"{adr_ok:>10}  {cr_ok:>8}"
        )
    else:
        print(f"{r['run_id']:<{COL_W}}  {r['D']:>3}  {r['c']:>3}  {r['B']:>3}  {'(missing)':>5}")

print(sep)
print(f"Aggregate summary -> {agg_path}")
PYEOF
fi

# ── Final status ──────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Attn split eval complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  OK: ${N_OK}  Skipped: ${N_SKIP}  Failed: ${N_FAIL}  Total: ${TOTAL}"
echo "  Results root: ${RESULTS_ROOT}"
echo "  Aggregate  : ${AGGREGATE_SUMMARY}"
echo "============================================================"

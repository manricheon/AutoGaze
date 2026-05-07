#!/usr/bin/env bash
# Run standard video QA benchmarks with/without AutoGaze.
#
# Usage:
#   bash scripts/run_benchmarks.sh [OPTIONS]
#
# Options:
#   --ratio FLOAT        Gazing ratio for AutoGaze ON runs  (default: 0.75)
#   --frames INT         Frames per video                   (default: 16)
#   --max-samples INT    Cap dataset size (e.g. 100 for smoke test)
#   --tasks LIST         Comma-separated task list          (default: all)
#                          e.g. --tasks videomme,mvbench,hlvid
#   --mllm NAME          MLLM backend (default: nvila)
#                          nvila          NVILA-8B (native AutoGaze processor)
#                          qwen25vl       Qwen2.5-VL-7B, zero-shot hook
#                          qwen25vl_full  Qwen2.5-VL-7B, full ViT integration
#                          vjepa2         V-JEPA2 encoder, zero-shot hook
#                          vjepa2_full    V-JEPA2 encoder, full integration
#                          vjepa2_llm     V-JEPA2 ViT + projector + LLM (MCQ QA)
#                          siglip         Vanilla HF SigLIP, feature extraction only
#   --lm-path P          [vjepa2_llm only] HF ID or local path for causal LLM
#                          e.g. Qwen/Qwen2.5-7B-Instruct
#   --projector-path P   [vjepa2_llm only] path to saved VJEPA2Projector weights
#                          (omit to use randomly-init projector for testing)
#   --baseline-only      Run AutoGaze OFF only
#   --autogaze-only      Run AutoGaze ON only
#   --results-dir DIR    Output directory                   (default: results/YYYYMMDD_HHMM)
#   --hlvid-video-dir D  Local video dir for HLVid          (default: data/HLVid/videos)
#   --model-path P       MLLM model path  (default: weights/NVILA-8B-HD-Video)
#   --autogaze-path P    AutoGaze path    (default: weights/AutoGaze)
#   --no-resume          Do not resume from existing outputs
#   --help, -h           Show this message
#
# Examples:
#   # Full run, all benchmarks, AutoGaze ON + OFF (NVILA)
#   bash scripts/run_benchmarks.sh
#
#   # Smoke test (100 samples each), VideoMME + MVBench only
#   bash scripts/run_benchmarks.sh --max-samples 100 --tasks videomme,mvbench
#
#   # Only AutoGaze ON with ratio=0.5
#   bash scripts/run_benchmarks.sh --autogaze-only --ratio 0.5
#
#   # Qwen2.5-VL-7B with AutoGaze (zero-shot hook — no model modification)
#   bash scripts/run_benchmarks.sh --mllm qwen25vl \
#       --model-path Qwen/Qwen2.5-VL-7B-Instruct \
#       --tasks videomme,mvbench
#
#   # Qwen2.5-VL-7B with AutoGaze (full ViT integration — per-temporal-chunk gaze)
#   bash scripts/run_benchmarks.sh --mllm qwen25vl_full \
#       --model-path Qwen/Qwen2.5-VL-7B-Instruct \
#       --tasks videomme,mvbench
#
#   # V-JEPA2 encoder with AutoGaze (zero-shot hook)
#   bash scripts/run_benchmarks.sh --mllm vjepa2 \
#       --model-path facebook/vjepa2-vitl-fpc64-256 \
#       --tasks videomme,mvbench
#
#   # V-JEPA2 encoder with AutoGaze (full integration — per-temporal-group gaze)
#   bash scripts/run_benchmarks.sh --mllm vjepa2_full \
#       --model-path facebook/vjepa2-vitl-fpc64-256 \
#       --tasks videomme,mvbench
#
#   # V-JEPA2 ViT + projector + LLM (full MCQ video QA; projector must be trained)
#   bash scripts/run_benchmarks.sh --mllm vjepa2_llm \
#       --model-path facebook/vjepa2-vitl-fpc64-256 \
#       --autogaze-path weights/AutoGaze \
#       --lm-path Qwen/Qwen2.5-7B-Instruct \
#       --projector-path weights/vjepa2_projector \
#       --tasks videomme,mvbench
#
#   # Run HLVid (local download required first)
#   bash scripts/download_hlvid.sh data/HLVid
#   bash scripts/run_benchmarks.sh --tasks hlvid --hlvid-video-dir data/HLVid/videos

set -euo pipefail

# ── colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
header()  { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── defaults ───────────────────────────────────────────────────────────────
RATIO="0.75"
FRAMES=16
MAX_SAMPLES=""
TASKS_ARG=""
MLLM="nvila"
RUN_BASELINE=true
RUN_AUTOGAZE=true
RESULTS_DIR=""
HLVID_VIDEO_DIR="data/HLVid/videos"
MODEL_PATH="weights/NVILA-8B-HD-Video"
AG_PATH="weights/AutoGaze"
RESUME="--resume"
LM_PATH=""
PROJECTOR_PATH=""

# ── argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ratio)           RATIO="$2";              shift 2 ;;
        --frames)          FRAMES="$2";             shift 2 ;;
        --max-samples)     MAX_SAMPLES="$2";        shift 2 ;;
        --tasks)           TASKS_ARG="$2";          shift 2 ;;
        --mllm)            MLLM="$2";               shift 2 ;;
        --baseline-only)   RUN_AUTOGAZE=false;      shift ;;
        --autogaze-only)   RUN_BASELINE=false;      shift ;;
        --results-dir)     RESULTS_DIR="$2";        shift 2 ;;
        --hlvid-video-dir) HLVID_VIDEO_DIR="$2";   shift 2 ;;
        --model-path)      MODEL_PATH="$2";         shift 2 ;;
        --autogaze-path)   AG_PATH="$2";            shift 2 ;;
        --lm-path)         LM_PATH="$2";            shift 2 ;;
        --projector-path)  PROJECTOR_PATH="$2";     shift 2 ;;
        --no-resume)       RESUME="";               shift ;;
        --help|-h)
            head -73 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── results directory ──────────────────────────────────────────────────────
if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="results/$(date +%Y%m%d_%H%M)"
fi
mkdir -p "$RESULTS_DIR"

# ── task list ──────────────────────────────────────────────────────────────
ALL_HF_TASKS=(videomme videomme_w_sub mvbench nextqa egoschema mlvu longvideobench)
ALL_LOCAL_TASKS=(hlvid)

if [[ -n "$TASKS_ARG" ]]; then
    IFS=',' read -ra SELECTED_TASKS <<< "$TASKS_ARG"
else
    SELECTED_TASKS=("${ALL_HF_TASKS[@]}" "${ALL_LOCAL_TASKS[@]}")
fi

# ── sanity checks ──────────────────────────────────────────────────────────
[[ -d "$MODEL_PATH" ]] || { warn "Model not found: $MODEL_PATH"; exit 1; }
if $RUN_AUTOGAZE; then
    [[ -d "$AG_PATH" ]] || { warn "AutoGaze weights not found: $AG_PATH"; exit 1; }
fi
if [[ "$MLLM" == "vjepa2_llm" ]]; then
    [[ -n "$LM_PATH" ]] || { warn "--mllm vjepa2_llm requires --lm-path"; exit 1; }
fi

# ── shared eval args ───────────────────────────────────────────────────────
COMMON_ARGS=(
    --model-path    "$MODEL_PATH"
    --autogaze-path "$AG_PATH"
    --num-frames    "$FRAMES"
    --mllm          "$MLLM"
)
[[ -n "$MAX_SAMPLES"    ]] && COMMON_ARGS+=(--max-samples    "$MAX_SAMPLES")
[[ -n "$RESUME"         ]] && COMMON_ARGS+=("$RESUME")
[[ -n "$LM_PATH"        ]] && COMMON_ARGS+=(--lm-path        "$LM_PATH")
[[ -n "$PROJECTOR_PATH" ]] && COMMON_ARGS+=(--projector-path "$PROJECTOR_PATH")

# ── helper: run one task ───────────────────────────────────────────────────
run_task() {
    local task="$1"
    local mode="$2"       # "ag" or "baseline"
    local extra_args=("${@:3}")

    local out_file
    if [[ "$mode" == "ag" ]]; then
        out_file="$RESULTS_DIR/${task}_${MLLM}_ag$(echo "$RATIO" | tr -d '.').json"
    else
        out_file="$RESULTS_DIR/${task}_${MLLM}_baseline.json"
    fi

    local cmd=(python -m autogaze.eval.run_benchmark
               --task "$task"
               "${COMMON_ARGS[@]}"
               "${extra_args[@]}"
               --output "$out_file")

    if [[ "$mode" == "baseline" ]]; then
        cmd+=(--no-autogaze)
    else
        cmd+=(--gazing-ratio "$RATIO")
    fi

    info "Task: ${task}  Mode: ${mode}  → ${out_file}"
    if "${cmd[@]}"; then
        ok "$task ($mode) done"
    else
        warn "$task ($mode) FAILED — check logs"
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}AutoGaze Video QA Benchmark Runner${NC}"
echo "  Results dir : $RESULTS_DIR"
echo "  Tasks       : ${SELECTED_TASKS[*]}"
echo "  MLLM        : $MLLM"
echo "  Gazing ratio: $RATIO"
echo "  Frames      : $FRAMES"
[[ -n "$MAX_SAMPLES" ]] && echo "  Max samples : $MAX_SAMPLES"
echo ""

for task in "${SELECTED_TASKS[@]}"; do

    # Determine extra args (HLVid needs --video-dir)
    extra=()
    if [[ "$task" == "hlvid" ]]; then
        if [[ ! -d "$HLVID_VIDEO_DIR" ]]; then
            warn "HLVid videos not found at $HLVID_VIDEO_DIR — skipping"
            warn "Run: bash scripts/download_hlvid.sh data/HLVid"
            continue
        fi
        extra+=(--video-dir "$HLVID_VIDEO_DIR")
    fi

    header "$task"

    if $RUN_AUTOGAZE; then
        run_task "$task" "ag" "${extra[@]}"
    fi

    if $RUN_BASELINE; then
        run_task "$task" "baseline" "${extra[@]}"
    fi

done

# ── summary table ──────────────────────────────────────────────────────────
header "결과 요약"

python3 - "$RESULTS_DIR" "$RUN_AUTOGAZE" "$RUN_BASELINE" <<'PYEOF'
import json, glob, sys
from pathlib import Path

results_dir  = sys.argv[1]
show_ag      = sys.argv[2] == "true"
show_baseline= sys.argv[3] == "true"

files = sorted(glob.glob(f"{results_dir}/*.json"))
if not files:
    print("결과 파일 없음")
    sys.exit(0)

# Collect results by task
from collections import defaultdict
by_task = defaultdict(dict)
for f in files:
    with open(f) as fp:
        r = json.load(fp)
    task = r.get("task", Path(f).stem)
    key  = "ag" if r.get("autogaze") else "baseline"
    by_task[task][key] = r["metrics"].get("overall_accuracy", 0)

# Print table
print(f"\n{'Task':25} {'AutoGaze ON':>12} {'Baseline':>10} {'차이':>8}")
print("-" * 58)

task_order = [
    "videomme", "videomme_w_sub", "mvbench",
    "nextqa", "egoschema", "mlvu", "longvideobench", "hlvid",
]
for task in task_order:
    d = by_task.get(task, {})
    if not d:
        continue
    ag_acc  = d.get("ag",       None)
    bl_acc  = d.get("baseline", None)
    ag_str  = f"{ag_acc:.2f}%" if ag_acc is not None else "—"
    bl_str  = f"{bl_acc:.2f}%" if bl_acc is not None else "—"
    diff_str= f"{ag_acc-bl_acc:+.2f}%" if (ag_acc and bl_acc) else "—"
    print(f"{task:25} {ag_str:>12} {bl_str:>10} {diff_str:>8}")

print()
PYEOF

echo ""
ok "완료. 결과: $RESULTS_DIR"

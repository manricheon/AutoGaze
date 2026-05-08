#!/usr/bin/env bash
# Download evaluation benchmark datasets for AutoGaze from HuggingFace.
#
# Usage:
#   bash scripts/download_data_eval.sh [TARGET_DIR] [DATASET...]
#
# Arguments:
#   TARGET_DIR : 저장 경로 (기본: ./data/eval)
#   DATASET... : 다운로드할 데이터셋 목록 (기본: hf_bytes)
#
# 개별 데이터셋:
#   videomme       — lmms-lab/Video-MME                        (~73 GB,  900 clips)
#   mvbench        — OpenGVLab/MVBench                         (~12 GB,  3641 clips)
#   nextqa         — lmms-lab/NExTQA                           (~6 GB,   4996 clips)
#   egoschema      — lmms-lab/EgoSchema                        (~30 GB,  5031 clips)
#   mlvu           — MLVU/MLVU                                 (~15 GB,  MCQ subset)
#   longvideobench — longvideobench/LongVideoBench              (~22 GB,  ~1400 clips)
#   hlvid          — bfshi/HLVid (delegates to download_hlvid.sh, ~152 GB)
#
# 그룹 키워드:
#   hf_bytes   — videomme + mvbench + nextqa + egoschema + mlvu + longvideobench
#                (HF 임베디드 바이트 방식 전체, ~158 GB)
#   all        — hf_bytes + hlvid  (~310 GB)
#
# 옵션:
#   --extract-videos   다운로드 후 videos/ 폴더에 mp4 파일 추출
#                      (scripts/extract_hf_videos.py 실행, python + datasets 필요)
#   --hlvid-parts RANGE  HLVid 파트 범위 (기본: 전체).
#                        예: "1-4"  → 파트 1~4 (~34 GB)
#   --annotations-only   HLVid 어노테이션만 다운로드 (비디오 제외)
#
# 예시:
#   bash scripts/download_data_eval.sh                              # 기본 (HF-bytes 전체, ~158 GB)
#   bash scripts/download_data_eval.sh data/eval videomme           # VideoMME만
#   bash scripts/download_data_eval.sh data/eval mvbench nextqa     # MVBench + NExTQA
#   bash scripts/download_data_eval.sh data/eval hf_bytes           # HF-bytes 전체
#   bash scripts/download_data_eval.sh data/eval all                # 전부 (~310 GB)
#   bash scripts/download_data_eval.sh data/eval hlvid --hlvid-parts 1-4  # HLVid 파트 1~4
#   bash scripts/download_data_eval.sh data/eval hf_bytes --extract-videos # 다운로드 + mp4 추출
#
# 다운로드 후 벤치마크 실행 예시:
#   python -m autogaze.eval.run_benchmark \
#       --task videomme \
#       --hf-data-dir data/eval/Video-MME \
#       --mllm nvila \
#       --model-path weights/NVILA-8B-HD-Video \
#       --autogaze-path weights/AutoGaze
#
#   # --extract-videos 사용 시 --video-dir 폴백도 가능:
#   python -m autogaze.eval.run_benchmark \
#       --task videomme \
#       --hf-data-dir data/eval/Video-MME \
#       --video-dir   data/eval/Video-MME/videos \
#       --mllm nvila ...

set -euo pipefail

# ── 색상 출력 ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 인수 파싱 ───────────────────────────────────────────────────────────────
TARGET_DIR="${1:-./data/eval}"
shift || true

DATASETS=()
EXTRACT_VIDEOS=false
HLVID_PARTS=""
ANNOTATIONS_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extract-videos)  EXTRACT_VIDEOS=true;  shift ;;
        --annotations-only) ANNOTATIONS_ONLY=true; shift ;;
        --hlvid-parts)
            [[ -z "${2:-}" ]] && error "--hlvid-parts 옵션에 범위가 필요합니다. 예: --hlvid-parts 1-4"
            HLVID_PARTS="$2"; shift 2 ;;
        --help|-h)
            head -55 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        -*)
            error "알 수 없는 옵션: $1  (--help로 사용법 확인)" ;;
        *)
            DATASETS+=("$1"); shift ;;
    esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("hf_bytes")
fi

# ── 그룹 키워드 확장 ────────────────────────────────────────────────────────
HF_BYTES_DATASETS=("videomme" "mvbench" "nextqa" "egoschema" "mlvu" "longvideobench")

EXPANDED=()
for d in "${DATASETS[@]}"; do
    case "$d" in
        hf_bytes) EXPANDED+=("${HF_BYTES_DATASETS[@]}") ;;
        all)      EXPANDED+=("${HF_BYTES_DATASETS[@]}" "hlvid") ;;
        *)        EXPANDED+=("$d") ;;
    esac
done
# 중복 제거 (순서 유지)
mapfile -t DATASETS < <(printf '%s\n' "${EXPANDED[@]}" | awk '!seen[$0]++')

# ── huggingface-cli 확인 ────────────────────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
    error "huggingface-cli를 찾을 수 없습니다.\n  pip install huggingface_hub 으로 설치하거나 scripts/setup.sh를 먼저 실행하세요."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$TARGET_DIR"

echo ""
info "평가 데이터셋 다운로드 시작"
info "저장 경로  : $TARGET_DIR"
info "대상 데이터셋: ${DATASETS[*]}"
info "mp4 추출   : $EXTRACT_VIDEOS"
echo ""

# ── 다운로드 함수 (HF-bytes 공통) ──────────────────────────────────────────

# $1: task name  $2: HF repo ID  $3: local dir name  $4: size hint
download_hf_dataset() {
    local task="$1"
    local repo="$2"
    local local_name="$3"
    local size="$4"
    local local_dir="$TARGET_DIR/$local_name"

    info "[$task] $repo ($size)"
    huggingface-cli download "$repo" \
        --repo-type dataset \
        --resume-download \
        --local-dir "$local_dir"
    success "$task → $local_dir"

    if [[ "$EXTRACT_VIDEOS" == true ]]; then
        info "  mp4 추출 중 ($task) ..."
        if python3 "$SCRIPT_DIR/extract_hf_videos.py" \
                --task "$task" \
                --hf-data-dir "$local_dir" \
                --out "$local_dir/videos"; then
            success "  mp4 추출 완료 → $local_dir/videos/"
        else
            warn "  mp4 추출 실패. --video-dir 없이 --hf-data-dir만 사용하세요."
        fi
    fi
}

download_videomme() {
    download_hf_dataset "videomme" "lmms-lab/Video-MME" "Video-MME" "~73 GB"
}

download_mvbench() {
    download_hf_dataset "mvbench" "OpenGVLab/MVBench" "MVBench" "~12 GB"
}

download_nextqa() {
    download_hf_dataset "nextqa" "lmms-lab/NExTQA" "NExTQA" "~6 GB"
}

download_egoschema() {
    download_hf_dataset "egoschema" "lmms-lab/EgoSchema" "EgoSchema" "~30 GB"
}

download_mlvu() {
    download_hf_dataset "mlvu" "MLVU/MLVU" "MLVU" "~15 GB"
}

download_longvideobench() {
    download_hf_dataset "longvideobench" "longvideobench/LongVideoBench" "LongVideoBench" "~22 GB"
}

download_hlvid() {
    local hlvid_script="$SCRIPT_DIR/download_hlvid.sh"
    if [[ ! -f "$hlvid_script" ]]; then
        error "download_hlvid.sh를 찾을 수 없습니다: $hlvid_script"
    fi
    local hlvid_dir="$TARGET_DIR/HLVid"
    local extra_args=()
    [[ -n "$HLVID_PARTS"     ]] && extra_args+=("--parts" "$HLVID_PARTS")
    [[ "$ANNOTATIONS_ONLY" == true ]] && extra_args+=("--annotations-only")
    info "[hlvid] bfshi/HLVid (~152 GB, 16 tar 파트)"
    bash "$hlvid_script" "$hlvid_dir" "${extra_args[@]}"
}

# ── 순서대로 실행 ───────────────────────────────────────────────────────────
STEP=0
TOTAL=${#DATASETS[@]}

for dataset in "${DATASETS[@]}"; do
    STEP=$((STEP + 1))
    echo ""
    echo "━━━━ [$STEP/$TOTAL] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    case "$dataset" in
        videomme)       download_videomme       ;;
        mvbench)        download_mvbench        ;;
        nextqa)         download_nextqa         ;;
        egoschema)      download_egoschema      ;;
        mlvu)           download_mlvu           ;;
        longvideobench) download_longvideobench ;;
        hlvid)          download_hlvid          ;;
        *)
            warn "알 수 없는 데이터셋: '$dataset'"
            warn "  선택 가능: videomme mvbench nextqa egoschema mlvu longvideobench hlvid"
            warn "  그룹 키워드: hf_bytes all"
            ;;
    esac
done

# ── 완료 요약 ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
success "다운로드 완료"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "저장 경로: $TARGET_DIR"
echo ""

# 다운로드 상태 확인
declare -A LOCAL_DIRS=(
    [videomme]="Video-MME"
    [mvbench]="MVBench"
    [nextqa]="NExTQA"
    [egoschema]="EgoSchema"
    [mlvu]="MLVU"
    [longvideobench]="LongVideoBench"
    [hlvid]="HLVid"
)

echo "데이터셋 상태:"
for dataset in "${DATASETS[@]}"; do
    dir="${LOCAL_DIRS[$dataset]:-}"
    [[ -z "$dir" ]] && continue
    local_path="$TARGET_DIR/$dir"
    if [[ -d "$local_path" ]]; then
        n_files=$(find "$local_path" -name "*.parquet" -o -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  ✓ $dataset  → $local_path  ($n_files 파일)"
        if [[ "$EXTRACT_VIDEOS" == true && -d "$local_path/videos" ]]; then
            n_videos=$(find "$local_path/videos" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
            echo "             videos/ : $n_videos mp4"
        fi
    else
        echo "  ✗ $dataset  (다운로드 실패 또는 경로 없음)"
    fi
done

# ── 벤치마크 실행 예시 ──────────────────────────────────────────────────────
echo ""
echo "벤치마크 실행 예시 (--hf-data-dir 사용):"
echo ""

declare -A RUN_TASKS=(
    [videomme]="videomme"
    [mvbench]="mvbench"
    [nextqa]="nextqa"
    [egoschema]="egoschema"
    [mlvu]="mlvu"
    [longvideobench]="longvideobench"
    [hlvid]="hlvid"
)

for dataset in "${DATASETS[@]}"; do
    dir="${LOCAL_DIRS[$dataset]:-}"
    task="${RUN_TASKS[$dataset]:-$dataset}"
    [[ -z "$dir" ]] && continue

    if [[ "$dataset" == "hlvid" ]]; then
        echo "  # $dataset"
        echo "  python -m autogaze.eval.run_benchmark \\"
        echo "      --task $task \\"
        echo "      --video-dir $TARGET_DIR/HLVid/videos \\"
        echo "      --mllm nvila \\"
        echo "      --model-path weights/NVILA-8B-HD-Video \\"
        echo "      --autogaze-path weights/AutoGaze"
    else
        echo "  # $dataset"
        echo "  python -m autogaze.eval.run_benchmark \\"
        echo "      --task $task \\"
        echo "      --hf-data-dir $TARGET_DIR/$dir \\"
        if [[ "$EXTRACT_VIDEOS" == true ]]; then
        echo "      --video-dir   $TARGET_DIR/$dir/videos \\"
        fi
        echo "      --mllm nvila \\"
        echo "      --model-path weights/NVILA-8B-HD-Video \\"
        echo "      --autogaze-path weights/AutoGaze"
    fi
    echo ""
done

echo "  # AutoGaze OFF (기준선) 비교:"
echo "  python -m autogaze.eval.run_benchmark \\"
echo "      --task videomme \\"
echo "      --hf-data-dir $TARGET_DIR/Video-MME \\"
echo "      --mllm nvila \\"
echo "      --model-path weights/NVILA-8B-HD-Video \\"
echo "      --no-autogaze"
echo ""
echo "  # run_benchmarks.sh (ON/OFF 자동 비교, --hf-data-dir 전달):"
echo "  bash scripts/run_benchmarks.sh \\"
echo "      --tasks videomme,mvbench \\"
echo "      --hf-data-dir $TARGET_DIR \\"
echo "      --model-path weights/NVILA-8B-HD-Video \\"
echo "      --autogaze-path weights/AutoGaze"

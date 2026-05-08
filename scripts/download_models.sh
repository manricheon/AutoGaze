#!/usr/bin/env bash
# Download model weights for AutoGaze from HuggingFace.
#
# Usage:
#   bash scripts/download_models.sh [TARGET_DIR] [MODEL...]
#
# Arguments:
#   TARGET_DIR  : 가중치 저장 경로 (기본: ./weights)
#   MODEL...    : 다운로드할 모델 목록 (기본: autogaze videomae)
#
# 개별 모델:
#   autogaze    — nvidia/AutoGaze                        (~50 MB,  패치 선택 모델)
#   videomae    — bfshi/VideoMAE_AutoGaze                (~2 GB,   학습용 보상 모델)
#   nvila       — nvidia/NVILA-8B-HD-Video               (~16 GB,  AutoGaze 통합 MLLM)
#   vjepa2      — facebook/vjepa2-vitl-fpc64-256         (~2 GB,   Video-native ViT-L)
#   qwen25vl    — Qwen/Qwen2.5-VL-7B-Instruct            (~16 GB,  VL MLLM)
#   qwen25      — Qwen/Qwen2.5-7B-Instruct               (~15 GB,  LM, vjepa2_llm용)
#   siglip      — google/siglip-base-patch16-224         (~400 MB, CV tasks)
#   siglip2     — google/siglip2-base-patch16-224        (~400 MB, CV tasks)
#   dinov2      — facebook/dinov2-base-imagenet1k-1-layer (~350 MB, CV tasks)
#   yolos       — hustvl/yolos-tiny                      (~30 MB,  CV object detection)
#   segformer   — nvidia/segformer-b2-finetuned-ade-512-512 (~100 MB, CV segmentation)
#   depthanything — depth-anything/Depth-Anything-V2-Small-hf (~100 MB, CV depth)
#
# 그룹 키워드:
#   mllm        — nvila + vjepa2 + qwen25vl + qwen25    (MLLM 벤치마크 전체)
#   cv          — siglip + siglip2 + dinov2 + yolos + segformer + depthanything
#   all         — autogaze + videomae + mllm + cv        (전부)
#
# 예시:
#   bash scripts/download_models.sh                         # 기본 (autogaze + videomae)
#   bash scripts/download_models.sh weights mllm            # MLLM 벤치마크용 전체
#   bash scripts/download_models.sh weights vjepa2 qwen25vl # V-JEPA2 + Qwen25VL만
#   bash scripts/download_models.sh weights cv              # CV 태스크 소형 모델
#   bash scripts/download_models.sh weights all             # 전부 (~52 GB)
#
# 오프라인 실행 시 로컬 경로:
#   nvila     → weights/NVILA-8B-HD-Video
#   vjepa2    → weights/vjepa2-vitl-fpc64-256
#   qwen25vl  → weights/Qwen2.5-VL-7B-Instruct
#   qwen25    → weights/Qwen2.5-7B-Instruct
#   CV 모델   → HuggingFace 캐시 (TRANSFORMERS_OFFLINE=1 시 자동 사용)

set -euo pipefail

# ── 색상 출력 ───────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 인수 파싱 ───────────────────────────────────────────────
TARGET_DIR="${1:-./weights}"
shift || true

if [[ $# -eq 0 ]]; then
    MODELS=("autogaze" "videomae")
else
    MODELS=("$@")
fi

# 그룹 키워드 확장
EXPANDED=()
for m in "${MODELS[@]}"; do
    case "$m" in
        mllm) EXPANDED+=("nvila" "vjepa2" "qwen25vl" "qwen25") ;;
        cv)   EXPANDED+=("siglip" "siglip2" "dinov2" "yolos" "segformer" "depthanything") ;;
        all)  EXPANDED+=("autogaze" "videomae" "nvila" "vjepa2" "qwen25vl" "qwen25"
                         "siglip" "siglip2" "dinov2" "yolos" "segformer" "depthanything") ;;
        *)    EXPANDED+=("$m") ;;
    esac
done
# 중복 제거 (순서 유지)
mapfile -t MODELS < <(printf '%s\n' "${EXPANDED[@]}" | awk '!seen[$0]++')

# ── huggingface-cli 확인 ────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
    error "huggingface-cli를 찾을 수 없습니다.\n  pip install huggingface_hub 으로 설치하거나 scripts/setup.sh를 먼저 실행하세요."
fi

mkdir -p "$TARGET_DIR"

echo ""
info "모델 다운로드 시작"
info "저장 경로 : $TARGET_DIR"
info "대상 모델 : ${MODELS[*]}"
echo ""

# ── 다운로드 함수 ───────────────────────────────────────────

download_autogaze() {
    info "[AutoGaze] nvidia/AutoGaze (~50 MB)"
    info "  용도: inference + 학습 (항상 필요)"
    huggingface-cli download nvidia/AutoGaze \
        --resume-download \
        --local-dir "$TARGET_DIR/AutoGaze"
    success "AutoGaze → $TARGET_DIR/AutoGaze"
}

download_videomae() {
    info "[VideoMAE] bfshi/VideoMAE_AutoGaze (~2 GB)"
    info "  용도: Stage1 NTP / Stage2 RL 학습용 보상 모델 (inference 불필요)"
    huggingface-cli download bfshi/VideoMAE_AutoGaze \
        --resume-download \
        --local-dir "$TARGET_DIR/VideoMAE_AutoGaze"
    success "VideoMAE → $TARGET_DIR/VideoMAE_AutoGaze"
}

download_nvila() {
    info "[NVILA] nvidia/NVILA-8B-HD-Video (~16.2 GB)"
    info "  용도: AutoGaze 통합 8B MLLM (고해상도·장형 비디오 이해)"
    info "  라이선스: CC-BY-NC-4.0 (비상업적 용도)"
    huggingface-cli download nvidia/NVILA-8B-HD-Video \
        --resume-download \
        --local-dir "$TARGET_DIR/NVILA-8B-HD-Video"
    success "NVILA → $TARGET_DIR/NVILA-8B-HD-Video"
}

download_vjepa2() {
    info "[V-JEPA2] facebook/vjepa2-vitl-fpc64-256 (~2 GB)"
    info "  용도: video-native ViT-L 특징 추출, AutoGaze 통합 벤치마크"
    info "  주의: transformers >= 4.53 필요"
    huggingface-cli download facebook/vjepa2-vitl-fpc64-256 \
        --resume-download \
        --local-dir "$TARGET_DIR/vjepa2-vitl-fpc64-256"
    success "V-JEPA2 → $TARGET_DIR/vjepa2-vitl-fpc64-256"
}

download_qwen25vl() {
    info "[Qwen2.5-VL] Qwen/Qwen2.5-VL-7B-Instruct (~16 GB)"
    info "  용도: Qwen2.5-VL MLLM 벤치마크 (qwen25vl / qwen25vl_full)"
    info "  주의: transformers >= 4.45 필요"
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
        --resume-download \
        --local-dir "$TARGET_DIR/Qwen2.5-VL-7B-Instruct"
    success "Qwen2.5-VL → $TARGET_DIR/Qwen2.5-VL-7B-Instruct"
}

download_qwen25() {
    info "[Qwen2.5] Qwen/Qwen2.5-7B-Instruct (~15 GB)"
    info "  용도: vjepa2_llm 파이프라인의 causal LM"
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
        --resume-download \
        --local-dir "$TARGET_DIR/Qwen2.5-7B-Instruct"
    success "Qwen2.5 → $TARGET_DIR/Qwen2.5-7B-Instruct"
}

# CV 소형 모델은 HuggingFace 캐시에 저장 (코드에서 HF ID로 참조)
download_cv_model() {
    local hf_id="$1"
    local label="$2"
    local size="$3"
    info "[$label] $hf_id ($size) → HF 캐시"
    huggingface-cli download "$hf_id" --resume-download
    success "$label cached"
}

download_siglip() {
    download_cv_model "google/siglip-base-patch16-224" "SigLIP-base" "~400 MB"
}

download_siglip2() {
    download_cv_model "google/siglip2-base-patch16-224" "SigLIP2-base" "~400 MB"
}

download_dinov2() {
    download_cv_model "facebook/dinov2-base-imagenet1k-1-layer" "DINOv2-base" "~350 MB"
}

download_yolos() {
    download_cv_model "hustvl/yolos-tiny" "YOLOS-tiny" "~30 MB"
}

download_segformer() {
    download_cv_model "nvidia/segformer-b2-finetuned-ade-512-512" "SegFormer-B2" "~100 MB"
}

download_depthanything() {
    download_cv_model "depth-anything/Depth-Anything-V2-Small-hf" "DepthAnything-V2-Small" "~100 MB"
}

# ── 순서대로 실행 ───────────────────────────────────────────
STEP=0
TOTAL=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    STEP=$((STEP + 1))
    echo ""
    echo "━━━━ [$STEP/$TOTAL] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    case "$model" in
        autogaze)      download_autogaze      ;;
        videomae)      download_videomae      ;;
        nvila)         download_nvila         ;;
        vjepa2)        download_vjepa2        ;;
        qwen25vl)      download_qwen25vl      ;;
        qwen25)        download_qwen25        ;;
        siglip)        download_siglip        ;;
        siglip2)       download_siglip2       ;;
        dinov2)        download_dinov2        ;;
        yolos)         download_yolos         ;;
        segformer)     download_segformer     ;;
        depthanything) download_depthanything ;;
        *)
            warn "알 수 없는 모델: '$model'"
            warn "  선택 가능: autogaze videomae nvila vjepa2 qwen25vl qwen25"
            warn "             siglip siglip2 dinov2 yolos segformer depthanything"
            warn "  그룹 키워드: mllm cv all"
            ;;
    esac
done

# ── 완료 요약 ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
success "다운로드 완료"
echo "════════════════════════════════════════════════════════"
echo ""
echo "저장 경로: $TARGET_DIR"
echo ""

# 로컬 경로 모델 상태 확인
declare -A LOCAL_PATHS=(
    [autogaze]="$TARGET_DIR/AutoGaze"
    [nvila]="$TARGET_DIR/NVILA-8B-HD-Video"
    [vjepa2]="$TARGET_DIR/vjepa2-vitl-fpc64-256"
    [qwen25vl]="$TARGET_DIR/Qwen2.5-VL-7B-Instruct"
    [qwen25]="$TARGET_DIR/Qwen2.5-7B-Instruct"
)

echo "로컬 저장 모델:"
for model in "${MODELS[@]}"; do
    path="${LOCAL_PATHS[$model]:-}"
    if [[ -n "$path" ]]; then
        [[ -d "$path" ]] \
            && echo "  ✓ $model  → $path" \
            || echo "  ✗ $model  (다운로드 실패)"
    elif [[ "$model" == "videomae" ]]; then
        [[ -f "$TARGET_DIR/VideoMAE_AutoGaze/videomae.pt" ]] \
            && echo "  ✓ videomae  → $TARGET_DIR/VideoMAE_AutoGaze/" \
            || echo "  ✗ videomae  (다운로드 실패)"
    fi
done

# CV 캐시 모델 확인
CV_MODELS=("siglip" "siglip2" "dinov2" "yolos" "segformer" "depthanything")
declare -A CV_HF_IDS=(
    [siglip]="google/siglip-base-patch16-224"
    [siglip2]="google/siglip2-base-patch16-224"
    [dinov2]="facebook/dinov2-base-imagenet1k-1-layer"
    [yolos]="hustvl/yolos-tiny"
    [segformer]="nvidia/segformer-b2-finetuned-ade-512-512"
    [depthanything]="depth-anything/Depth-Anything-V2-Small-hf"
)

HAS_CV=false
for model in "${MODELS[@]}"; do
    if [[ -n "${CV_HF_IDS[$model]:-}" ]]; then
        HAS_CV=true; break
    fi
done

if $HAS_CV; then
    echo ""
    echo "HF 캐시 모델 (TRANSFORMERS_OFFLINE=1 시 자동 사용):"
    for model in "${MODELS[@]}"; do
        hf_id="${CV_HF_IDS[$model]:-}"
        [[ -z "$hf_id" ]] && continue
        cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--$(echo "$hf_id" | tr '/' '--')"
        [[ -d "$cache_dir" ]] \
            && echo "  ✓ $model  ($hf_id)" \
            || echo "  ✗ $model  ($hf_id) — 캐시 미확인"
    done
fi

echo ""
echo "벤치마크 실행 예시 (로컬 경로 사용):"
echo ""
echo "  # NVILA"
echo "  bash scripts/run_benchmarks.sh --mllm nvila \\"
echo "      --model-path $TARGET_DIR/NVILA-8B-HD-Video"
echo ""
echo "  # Qwen2.5-VL"
echo "  bash scripts/run_benchmarks.sh --mllm qwen25vl \\"
echo "      --model-path $TARGET_DIR/Qwen2.5-VL-7B-Instruct"
echo ""
echo "  # V-JEPA2 + LLM"
echo "  bash scripts/run_benchmarks.sh --mllm vjepa2_llm \\"
echo "      --model-path $TARGET_DIR/vjepa2-vitl-fpc64-256 \\"
echo "      --lm-path    $TARGET_DIR/Qwen2.5-7B-Instruct \\"
echo "      --projector-path $TARGET_DIR/vjepa2_projector"
echo ""
echo "학습 시작 명령:"
echo "  NTP : bash scripts/train_ntp_single_gpu.sh <data_dir> $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt"
echo "  RL  : bash scripts/train_rl_single_gpu.sh  <data_dir> $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt <ntp_ckpt>"

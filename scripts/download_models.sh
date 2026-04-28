#!/usr/bin/env bash
# Download model weights for AutoGaze from HuggingFace.
#
# Usage:
#   bash scripts/download_models.sh [TARGET_DIR] [MODEL...]
#
# Arguments:
#   TARGET_DIR  : 가중치 저장 경로 (기본: ./weights)
#   MODEL...    : 다운로드할 모델 목록 (기본: autogaze videomae)
#                 선택 가능한 모델:
#                   autogaze  — nvidia/AutoGaze          (~50 MB,  패치 선택 모델)
#                   videomae  — bfshi/VideoMAE_AutoGaze  (~2 GB,   학습용 보상 모델)
#                   nvila     — nvidia/NVILA-8B-HD-Video (~16 GB,  통합 MLLM 데모용)
#                   all       — 위 세 가지 모두
#
# 예시:
#   bash scripts/download_models.sh                        # 기본 (autogaze + videomae)
#   bash scripts/download_models.sh weights all            # 전부 (~18 GB)
#   bash scripts/download_models.sh weights nvila          # NVILA만
#   bash scripts/download_models.sh weights autogaze nvila # AutoGaze + NVILA
#
# 모델별 용도:
#   autogaze  : inference / NTP 학습 / RL 학습 모두 필요 (항상 필요)
#   videomae  : Stage 1 NTP 및 Stage 2 RL 학습 시 필요 (inference 불필요)
#   nvila     : AutoGaze가 통합된 8B MLLM. 고해상도·장형 비디오 이해 데모용.
#               학습/평가 시 VILA 리포지터리별도 필요.

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

# 나머지 인수가 없으면 기본값 사용
if [[ $# -eq 0 ]]; then
    MODELS=("autogaze" "videomae")
else
    MODELS=("$@")
fi

# "all" 확장
EXPANDED=()
for m in "${MODELS[@]}"; do
    if [[ "$m" == "all" ]]; then
        EXPANDED+=("autogaze" "videomae" "nvila")
    else
        EXPANDED+=("$m")
    fi
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
info "저장 경로: $TARGET_DIR"
info "대상 모델: ${MODELS[*]}"
echo ""

# ── 다운로드 함수 ───────────────────────────────────────────
download_autogaze() {
    info "[AutoGaze] nvidia/AutoGaze (~50 MB)"
    info "  용도: inference + 학습 (항상 필요)"
    huggingface-cli download nvidia/AutoGaze \
        --local-dir "$TARGET_DIR/AutoGaze"
    success "AutoGaze → $TARGET_DIR/AutoGaze"
}

download_videomae() {
    info "[VideoMAE] bfshi/VideoMAE_AutoGaze (~2 GB)"
    info "  용도: Stage1 NTP / Stage2 RL 학습용 보상 모델 (inference 불필요)"
    huggingface-cli download bfshi/VideoMAE_AutoGaze \
        --local-dir "$TARGET_DIR/VideoMAE_AutoGaze"
    success "VideoMAE → $TARGET_DIR/VideoMAE_AutoGaze"
    info "  videomae.pt 경로: $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt"
}

download_nvila() {
    info "[NVILA] nvidia/NVILA-8B-HD-Video (~16.2 GB)"
    info "  용도: AutoGaze 통합 8B MLLM (4K/1K-frame 비디오 이해 데모)"
    info "  주의: 사용 시 VILA 리포지터리 별도 설치 필요"
    info "        https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video"
    info "  라이선스: CC-BY-NC-4.0 (비상업적 용도)"
    echo ""
    huggingface-cli download nvidia/NVILA-8B-HD-Video \
        --local-dir "$TARGET_DIR/NVILA-8B-HD-Video"
    success "NVILA → $TARGET_DIR/NVILA-8B-HD-Video"
}

# ── 순서대로 실행 ───────────────────────────────────────────
STEP=0
TOTAL=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    STEP=$((STEP + 1))
    echo ""
    echo "━━━━ [$STEP/$TOTAL] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    case "$model" in
        autogaze) download_autogaze ;;
        videomae) download_videomae ;;
        nvila)    download_nvila    ;;
        *)
            warn "알 수 없는 모델: '$model' (autogaze / videomae / nvila / all)"
            ;;
    esac
done

# ── 완료 요약 ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
success "다운로드 완료"
echo "════════════════════════════════════════"
echo ""
echo "저장 경로: $TARGET_DIR"
echo ""

echo "다운로드된 모델:"
for model in "${MODELS[@]}"; do
    case "$model" in
        autogaze)
            [[ -d "$TARGET_DIR/AutoGaze" ]] \
                && echo "  ✓ AutoGaze           → $TARGET_DIR/AutoGaze/" \
                || echo "  ✗ AutoGaze           (다운로드 실패 또는 건너뜀)"
            ;;
        videomae)
            [[ -f "$TARGET_DIR/VideoMAE_AutoGaze/videomae.pt" ]] \
                && echo "  ✓ VideoMAE_AutoGaze  → $TARGET_DIR/VideoMAE_AutoGaze/" \
                || echo "  ✗ VideoMAE_AutoGaze  (다운로드 실패 또는 건너뜀)"
            ;;
        nvila)
            [[ -d "$TARGET_DIR/NVILA-8B-HD-Video" ]] \
                && echo "  ✓ NVILA-8B-HD-Video  → $TARGET_DIR/NVILA-8B-HD-Video/" \
                || echo "  ✗ NVILA-8B-HD-Video  (다운로드 실패 또는 건너뜀)"
            ;;
    esac
done

echo ""
echo "학습 시작 명령:"
echo "  NTP : bash scripts/train_ntp_single_gpu.sh <data_dir> $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt"
echo "  RL  : bash scripts/train_rl_single_gpu.sh  <data_dir> $TARGET_DIR/VideoMAE_AutoGaze/videomae.pt <ntp_ckpt>"

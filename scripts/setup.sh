#!/usr/bin/env bash
# AutoGaze 환경 일괄 설정 스크립트
#
# 사용법:
#   bash scripts/setup.sh [옵션]
#
# 옵션:
#   --skip-weights   모델 가중치 다운로드 건너뜀 (인퍼런스 때 HF에서 자동 다운로드)
#   --with-flash     CUDA 환경에서 flash-attn 설치 (GPU 학습 시 권장)
#   --no-hf-login    HuggingFace 로그인 건너뜀 (이미 로그인된 경우)
#   --python PATH    사용할 python 실행 파일 경로 (기본: 자동 감지)
#
# 예시:
#   bash scripts/setup.sh                          # 기본 (가중치 포함)
#   bash scripts/setup.sh --skip-weights           # 가중치 제외 (빠른 설정)
#   bash scripts/setup.sh --with-flash             # GPU 환경 (flash-attn 포함)
#   bash scripts/setup.sh --skip-weights --no-hf-login  # 오프라인 환경

set -euo pipefail

# ── 색상 출력 ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}── $* ──${NC}"; }

# ── 인수 파싱 ─────────────────────────────────────────────────────
SKIP_WEIGHTS=false
WITH_FLASH=false
NO_HF_LOGIN=false
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-weights)  SKIP_WEIGHTS=true  ;;
        --with-flash)    WITH_FLASH=true    ;;
        --no-hf-login)   NO_HF_LOGIN=true   ;;
        --python)        PYTHON_BIN="$2"; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) warn "알 수 없는 옵션: $1 (무시됨)" ;;
    esac
    shift
done

# ── 프로젝트 루트로 이동 ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"
info "프로젝트 루트: $ROOT_DIR"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "1. Python 버전 확인"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ -z "$PYTHON_BIN" ]]; then
    # Python 3.10+ 우선, 없으면 3.8+ 수용
    for candidate in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=${ver%%.*}; minor=${ver##*.}
            if [[ $major -eq 3 && $minor -ge 8 ]]; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    done
fi

[[ -z "$PYTHON_BIN" ]] && error "Python 3.8 이상을 찾을 수 없습니다."

PY_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
success "Python $PY_VER ($PYTHON_BIN)"

# 권장 버전 안내
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [[ $PY_MINOR -lt 10 ]]; then
    warn "Python 3.10 이상을 권장합니다. 현재: $PY_VER"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "2. 가상 환경(venv) 설정"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VENV_DIR="$ROOT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    VENV_PY="$VENV_DIR/bin/python"
    VENV_VER=$("$VENV_PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null || echo "unknown")
    success "기존 venv 사용: $VENV_DIR (Python $VENV_VER)"
else
    info "가상 환경 생성 중: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "가상 환경 생성 완료"
fi

# venv Python/pip 경로
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# pip 최신 버전으로 업그레이드
info "pip 업그레이드 중..."
"$VENV_PY" -m pip install --upgrade pip --quiet
success "pip 업그레이드 완료"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "3. 패키지 설치"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# OS / 디바이스 감지
OS=$(uname -s)
info "운영체제: $OS"

HAS_CUDA=false
if command -v nvcc &>/dev/null || [[ -d /usr/local/cuda ]]; then
    HAS_CUDA=true
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d',' -f1 || echo "unknown")
    info "CUDA 감지됨: $CUDA_VER"
fi

HAS_MPS=false
if [[ "$OS" == "Darwin" ]]; then
    if "$VENV_PY" -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        HAS_MPS=true
        info "Apple Silicon MPS 감지됨"
    fi
fi

# PyTorch 설치 여부 확인 (이미 있으면 건너뜀)
TORCH_INSTALLED=false
if "$VENV_PY" -c "import torch" 2>/dev/null; then
    TORCH_VER=$("$VENV_PY" -c "import torch; print(torch.__version__)")
    success "PyTorch 이미 설치됨: $TORCH_VER (재설치 건너뜀)"
    TORCH_INSTALLED=true
fi

if [[ "$TORCH_INSTALLED" == "false" ]]; then
    if [[ "$HAS_CUDA" == "true" ]]; then
        info "PyTorch (CUDA) 설치 중..."
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    elif [[ "$OS" == "Darwin" ]]; then
        info "PyTorch (macOS) 설치 중..."
        "$VENV_PIP" install torch torchvision --quiet
    else
        info "PyTorch (CPU) 설치 중..."
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    fi
    TORCH_VER=$("$VENV_PY" -c "import torch; print(torch.__version__)")
    success "PyTorch 설치 완료: $TORCH_VER"
fi

# autogaze 패키지 설치 (개발 모드)
info "autogaze 패키지 설치 중 (pip install -e \".[dev]\")..."
"$VENV_PIP" install -e ".[dev]" --quiet
success "autogaze 설치 완료"

# Jupyter/ipykernel (노트북 실행용)
if ! "$VENV_PY" -c "import jupyter" 2>/dev/null; then
    info "Jupyter 설치 중..."
    "$VENV_PIP" install jupyter ipykernel --quiet
    success "Jupyter 설치 완료"
else
    success "Jupyter 이미 설치됨 (건너뜀)"
fi

# scipy (검증 노트북용)
if ! "$VENV_PY" -c "import scipy" 2>/dev/null; then
    info "scipy 설치 중..."
    "$VENV_PIP" install scipy --quiet
    success "scipy 설치 완료"
fi

# Jupyter 커널 등록
info "Jupyter 커널 등록 중 (autogaze)..."
if "$VENV_PY" -m ipykernel install --user --name autogaze --display-name "AutoGaze (.venv)" --quiet 2>/dev/null; then
    success "Jupyter 커널 등록 완료"
else
    warn "커널 등록 실패 (수동 실행: python -m ipykernel install --user --name autogaze)"
fi

# flash-attn (CUDA + --with-flash 플래그가 있을 때만)
if [[ "$WITH_FLASH" == "true" ]]; then
    if [[ "$HAS_CUDA" == "false" ]]; then
        warn "--with-flash 지정됐지만 CUDA가 없습니다. flash-attn 건너뜁니다."
    else
        info "flash-attn 설치 중 (시간이 걸릴 수 있습니다)..."
        "$VENV_PIP" install flash-attn --no-build-isolation --quiet && \
            success "flash-attn 설치 완료" || \
            warn "flash-attn 설치 실패 (CUDA 버전 불일치일 수 있습니다). 학습은 가능하나 속도가 느릴 수 있습니다."
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "4. HuggingFace 로그인"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$NO_HF_LOGIN" == "true" ]]; then
    info "--no-hf-login 옵션 지정됨 — 로그인 건너뜁니다."
else
    # 이미 로그인 여부 확인
    HF_CACHE_TOKEN="${HF_HOME:-$HOME/.cache/huggingface}/token"
    if [[ -f "$HF_CACHE_TOKEN" ]]; then
        success "HuggingFace 이미 로그인됨 (건너뜀)"
    else
        warn "HuggingFace 미로그인 상태입니다."
        echo ""
        echo "  모델 가중치 다운로드에 HuggingFace 계정이 필요합니다."
        echo "  아래 명령으로 로그인하세요:"
        echo ""
        echo -e "    ${BOLD}source .venv/bin/activate && huggingface-cli login${NC}"
        echo ""
        echo "  또는 HF_TOKEN 환경 변수를 설정하세요:"
        echo "    export HF_TOKEN=hf_..."
        echo ""
        read -r -p "  지금 로그인하시겠습니까? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            "$VENV_DIR/bin/huggingface-cli" login
        else
            warn "HuggingFace 로그인 건너뜀. 가중치 다운로드 시 수동 로그인 필요."
        fi
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "5. 모델 가중치 다운로드"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$SKIP_WEIGHTS" == "true" ]]; then
    info "--skip-weights 지정됨 — 가중치 다운로드 건너뜁니다."
    info "나중에 다운로드: bash scripts/download_models.sh"
else
    WEIGHTS_DIR="$ROOT_DIR/weights"

    # AutoGaze 가중치
    if [[ -f "$WEIGHTS_DIR/AutoGaze/config.json" ]]; then
        success "AutoGaze 가중치 이미 존재: $WEIGHTS_DIR/AutoGaze"
    else
        info "AutoGaze 가중치 다운로드 중 (nvidia/AutoGaze) ..."
        "$VENV_DIR/bin/huggingface-cli" download nvidia/AutoGaze \
            --local-dir "$WEIGHTS_DIR/AutoGaze" && \
            success "AutoGaze 가중치 다운로드 완료" || \
            warn "AutoGaze 가중치 다운로드 실패. 수동 실행: bash scripts/download_models.sh"
    fi

    # VideoMAE 가중치 (학습용, 2GB)
    if [[ -f "$WEIGHTS_DIR/VideoMAE_AutoGaze/videomae.pt" ]]; then
        success "VideoMAE 가중치 이미 존재: $WEIGHTS_DIR/VideoMAE_AutoGaze"
    else
        echo ""
        warn "VideoMAE 가중치가 없습니다. (인퍼런스에는 불필요, 학습에 필요)"
        read -r -p "  VideoMAE 가중치를 다운로드하시겠습니까? (~2 GB) [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            info "VideoMAE 가중치 다운로드 중 (bfshi/VideoMAE_AutoGaze) ..."
            "$VENV_DIR/bin/huggingface-cli" download bfshi/VideoMAE_AutoGaze \
                --local-dir "$WEIGHTS_DIR/VideoMAE_AutoGaze" && \
                success "VideoMAE 가중치 다운로드 완료" || \
                warn "VideoMAE 다운로드 실패. 수동 실행: bash scripts/download_models.sh"
        else
            info "VideoMAE 건너뜀. 학습 전 실행: bash scripts/download_models.sh"
        fi
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "6. 설치 검증"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERIFY_FAIL=false

check_import() {
    local pkg="$1"
    if "$VENV_PY" -c "import $pkg" 2>/dev/null; then
        ver=$("$VENV_PY" -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null || echo "ok")
        success "$pkg ($ver)"
    else
        warn "$pkg 임포트 실패"
        VERIFY_FAIL=true
    fi
}

check_import torch
check_import torchvision
check_import transformers
check_import autogaze
check_import av
check_import matplotlib
check_import numpy
check_import scipy

# 디바이스 확인
DEVICE=$("$VENV_PY" -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA ({torch.cuda.get_device_name(0)})')
elif torch.backends.mps.is_available():
    print('MPS (Apple Silicon)')
else:
    print('CPU')
" 2>/dev/null || echo "확인 실패")
success "사용 가능 디바이스: $DEVICE"

# 예제 비디오 확인
if [[ -f "$ROOT_DIR/assets/example_input.mp4" ]]; then
    success "예제 비디오: assets/example_input.mp4"
else
    warn "예제 비디오 없음: assets/example_input.mp4"
    info "HuggingFace에서 자동 다운로드되거나 직접 준비하세요."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "7. 빠른 동작 확인 (smoke test)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHTS_PRESENT=false
if [[ -f "$ROOT_DIR/weights/AutoGaze/config.json" ]]; then
    WEIGHTS_PRESENT=true
fi

if [[ "$WEIGHTS_PRESENT" == "true" ]]; then
    info "AutoGaze 모델 로드 테스트 중..."
    "$VENV_PY" - <<'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
    transform = AutoGazeImageProcessor.from_pretrained("weights/AutoGaze")
    model     = AutoGaze.from_pretrained("weights/AutoGaze")
    model.eval()
    print(f"  모델 로드 성공 — max_num_frames={model.config.max_num_frames}, "
          f"tokens/frame={model.config.num_vision_tokens_each_frame}")
except Exception as e:
    print(f"  모델 로드 실패: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    success "smoke test 통과"
else
    warn "가중치 없음 — smoke test 건너뜀"
    info "weights/AutoGaze 에 가중치를 두거나 --skip-weights 없이 재실행하세요."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${GREEN}  설정 완료!${NC}"
echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  가상 환경 활성화:"
echo -e "    ${BOLD}source .venv/bin/activate${NC}"
echo ""
echo "  인퍼런스 실행:"
echo -e "    ${BOLD}bash scripts/run_inference.sh assets/example_input.mp4${NC}"
echo ""
echo "  Jupyter 노트북 실행:"
echo -e "    ${BOLD}jupyter notebook notebooks/${NC}"
echo ""
echo "  학습 데이터 다운로드 (필요 시):"
echo -e "    ${BOLD}bash scripts/download_data.sh [TARGET_DIR] [SUBSET]${NC}"
echo ""

if [[ "$VERIFY_FAIL" == "true" ]]; then
    warn "일부 패키지 임포트에 실패했습니다. 위 경고를 확인하세요."
fi

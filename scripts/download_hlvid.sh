#!/usr/bin/env bash
# Download HLVid benchmark (test set) from HuggingFace.
#
# Usage:
#   bash scripts/download_hlvid.sh [OPTIONS] [TARGET_DIR]
#
# Arguments:
#   TARGET_DIR            : Data 저장 경로 (기본: ./data/HLVid)
#
# Options:
#   --annotations-only    QA 어노테이션만 다운로드 (비디오 제외, ~수 MB)
#   --parts RANGE         특정 파트만 다운로드.
#                           e.g. "1-4"    → 파트 1, 2, 3, 4
#                                "1,3,5"  → 파트 1, 3, 5
#                                "1-3,7"  → 파트 1, 2, 3, 7
#   --to-json             어노테이션을 parquet에서 JSON으로 변환 (python + datasets 필요)
#   --skip-extract        tar 파일 추출 건너뜀 (파일 보관용)
#
# Dataset 정보:
#   비디오  : 4K 해상도 5분짜리 영상, 16개 tar 파트 (~152 GB)
#   어노테이션: 268개 다지선다 QA 쌍 (question, options, answer, video_path, category)
#
# 예시:
#   bash scripts/download_hlvid.sh                        # 전체 다운로드 (152 GB)
#   bash scripts/download_hlvid.sh --annotations-only     # QA만 (~수 MB)
#   bash scripts/download_hlvid.sh --parts 1-4            # 파트 1~4만 (~34 GB)
#   bash scripts/download_hlvid.sh --parts 1-4 --to-json  # 파트 1~4 + JSON 변환
#
# 예상 디렉터리 구조 (추출 후):
#   data/HLVid/
#   ├── annotations/
#   │   ├── test-00000-of-00001.parquet
#   │   └── hlvid_test.json          ← --to-json 옵션 사용 시
#   ├── videos/
#   │   ├── clip_av_video_*.mp4
#   │   └── clip_household_*.mp4
#   └── videos_parts/               ← --skip-extract 사용 시
#       ├── videos_part_0001.tar
#       └── ...

set -euo pipefail

# ── 색상 출력 ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 기본값 ─────────────────────────────────────────────────────
TARGET_DIR="./data/HLVid"
ANNOTATIONS_ONLY=false
TO_JSON=false
SKIP_EXTRACT=false
PARTS_RANGE=""   # 빈 문자열 = 전체

REPO="bfshi/HLVid"
TOTAL_PARTS=16

# ── 인수 파싱 ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --annotations-only) ANNOTATIONS_ONLY=true; shift ;;
        --to-json)          TO_JSON=true; shift ;;
        --skip-extract)     SKIP_EXTRACT=true; shift ;;
        --parts)
            [[ -z "${2:-}" ]] && error "--parts 옵션에 범위가 필요합니다. 예: --parts 1-4"
            PARTS_RANGE="$2"; shift 2 ;;
        --help|-h)
            head -40 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        -*)
            error "알 수 없는 옵션: $1  (--help로 사용법 확인)" ;;
        *)
            TARGET_DIR="$1"; shift ;;
    esac
done

# ── PARTS_RANGE 파싱 → 파트 번호 배열 생성 ─────────────────────
parse_parts() {
    local range="$1"
    local result=()
    IFS=',' read -ra segments <<< "$range"
    for seg in "${segments[@]}"; do
        if [[ "$seg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local s="${BASH_REMATCH[1]}" e="${BASH_REMATCH[2]}"
            for ((n=s; n<=e; n++)); do result+=("$n"); done
        elif [[ "$seg" =~ ^([0-9]+)$ ]]; then
            result+=("${BASH_REMATCH[1]}")
        else
            error "파트 범위 파싱 오류: '$seg'  (예: '1-4', '1,3,5', '1-3,7')"
        fi
    done
    printf '%s\n' "${result[@]}" | sort -un
}

if [[ -n "$PARTS_RANGE" ]]; then
    mapfile -t PARTS_TO_DOWNLOAD < <(parse_parts "$PARTS_RANGE")
else
    mapfile -t PARTS_TO_DOWNLOAD < <(seq 1 $TOTAL_PARTS)
fi

# ── huggingface-cli 확인 ───────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
    error "huggingface-cli를 찾을 수 없습니다.\n  pip install huggingface_hub 으로 설치하거나 scripts/setup.sh를 먼저 실행하세요."
fi

# ── 디렉터리 생성 ──────────────────────────────────────────────
mkdir -p "$TARGET_DIR/annotations"
if [[ "$ANNOTATIONS_ONLY" == false ]]; then
    mkdir -p "$TARGET_DIR/videos"
    if [[ "$SKIP_EXTRACT" == true ]]; then
        mkdir -p "$TARGET_DIR/videos_parts"
    fi
fi

echo ""
info "HLVid 벤치마크 다운로드"
info "저장 경로  : $TARGET_DIR"
info "옵션       : annotations_only=$ANNOTATIONS_ONLY  to_json=$TO_JSON  skip_extract=$SKIP_EXTRACT"
if [[ -n "$PARTS_RANGE" ]]; then
    info "다운로드 파트: ${PARTS_TO_DOWNLOAD[*]}"
else
    info "다운로드 파트: 전체 (1–$TOTAL_PARTS, ~152 GB)"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# 1. 어노테이션 다운로드
# ──────────────────────────────────────────────────────────────
download_annotations() {
    info "QA 어노테이션 다운로드 중..."
    huggingface-cli download "$REPO" \
        --repo-type dataset \
        --resume-download \
        --include "data/*.parquet" \
        --local-dir "$TARGET_DIR/annotations"
    success "어노테이션 다운로드 완료 → $TARGET_DIR/annotations/"
}

# ──────────────────────────────────────────────────────────────
# 2. parquet → JSON 변환 (선택)
# ──────────────────────────────────────────────────────────────
convert_to_json() {
    local out_json="$TARGET_DIR/annotations/hlvid_test.json"
    info "parquet → JSON 변환 중..."

    if python3 - <<'PYEOF' "$TARGET_DIR/annotations" "$out_json"
import sys, json, glob
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas가 설치되어 있지 않습니다. pip install pandas pyarrow")
    sys.exit(1)

annot_dir, out_path = sys.argv[1], sys.argv[2]
parquet_files = sorted(glob.glob(f"{annot_dir}/**/*.parquet", recursive=True))
if not parquet_files:
    print(f"ERROR: parquet 파일을 찾을 수 없습니다: {annot_dir}")
    sys.exit(1)

import pandas as pd
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
records = df.to_dict(orient="records")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"변환 완료: {len(records)}개 레코드 → {out_path}")
PYEOF
    then
        success "JSON 변환 완료 → $out_json"
    else
        warn "JSON 변환 실패 (parquet 파일은 $TARGET_DIR/annotations/ 에 유지됩니다)"
    fi
}

# ──────────────────────────────────────────────────────────────
# 3. 비디오 파트 다운로드 & 추출
# ──────────────────────────────────────────────────────────────
download_and_extract_part() {
    local part_num="$1"
    local filename
    filename=$(printf "videos_part_%04d.tar" "$part_num")

    info "파트 $part_num / $TOTAL_PARTS 다운로드: $filename"
    huggingface-cli download "$REPO" "$filename" \
        --repo-type dataset \
        --resume-download \
        --local-dir "$TARGET_DIR/videos_parts"

    local tar_path="$TARGET_DIR/videos_parts/$filename"

    if [[ "$SKIP_EXTRACT" == false ]]; then
        info "  추출 중: $filename → $TARGET_DIR/videos/"
        tar -xf "$tar_path" -C "$TARGET_DIR/videos/"
        rm -f "$tar_path"
        success "  파트 $part_num 완료 (tar 삭제됨)"
    else
        success "  파트 $part_num 다운로드 완료 (추출 건너뜀)"
    fi
}

# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────

download_annotations

if [[ "$TO_JSON" == true ]]; then
    convert_to_json
fi

if [[ "$ANNOTATIONS_ONLY" == false ]]; then
    n_parts=${#PARTS_TO_DOWNLOAD[@]}
    info "비디오 파트 다운로드 시작 (${n_parts}개 파트)"
    echo ""

    for part in "${PARTS_TO_DOWNLOAD[@]}"; do
        if [[ "$part" -lt 1 || "$part" -gt "$TOTAL_PARTS" ]]; then
            warn "파트 번호 $part 는 범위(1–$TOTAL_PARTS)를 벗어납니다. 건너뜁니다."
            continue
        fi
        download_and_extract_part "$part"
    done
fi

# ──────────────────────────────────────────────────────────────
# 완료 요약
# ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
success "HLVid 다운로드 완료"
echo "========================================"
echo ""
echo "저장 경로: $TARGET_DIR"
echo ""
echo "예상 디렉터리 구조:"
echo "  $TARGET_DIR/"
echo "  ├── annotations/"
echo "  │   ├── test-*.parquet       ← QA 어노테이션 (268개 샘플)"
if [[ "$TO_JSON" == true ]]; then
echo "  │   └── hlvid_test.json      ← JSON 변환본"
fi
if [[ "$ANNOTATIONS_ONLY" == false ]]; then
if [[ "$SKIP_EXTRACT" == false ]]; then
echo "  └── videos/"
echo "      ├── clip_av_video_*.mp4"
echo "      └── clip_household_*.mp4"
else
echo "  └── videos_parts/"
echo "      └── videos_part_NNNN.tar"
fi
fi
echo ""
echo "Python에서 어노테이션 로드:"
echo "  from datasets import load_dataset"
echo "  ds = load_dataset('bfshi/HLVid')  # HuggingFace에서 직접"
echo "  # 또는 로컬:"
echo "  ds = load_dataset('parquet', data_files={'test': '$TARGET_DIR/annotations/*.parquet'})"
if [[ "$TO_JSON" == true ]]; then
echo "  # JSON 사용:"
echo "  import json"
echo "  data = json.load(open('$TARGET_DIR/annotations/hlvid_test.json'))"
fi

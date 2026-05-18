#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
VILA_DIR="${EXTERNAL_DIR}/VILA"
PYTHON_BIN="${PYTHON:-python3}"

cd "${ROOT_DIR}"
mkdir -p "${EXTERNAL_DIR}" "${ROOT_DIR}/outputs/autogaze_repro"

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  if [[ -d "${target_dir}/.git" ]]; then
    git -C "${target_dir}" fetch --prune origin
    git -C "${target_dir}" switch main
    git -C "${target_dir}" pull --ff-only origin main
  else
    git clone "${repo_url}" "${target_dir}"
  fi
}

case "${1:-}" in
  "")
    ;;
  "--with-vila")
    clone_or_update "https://github.com/NVlabs/VILA.git" "${VILA_DIR}"
    ;;
  *)
    echo "Usage: $0 [--with-vila]" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path.cwd()
sources = {
    "AutoGaze": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "AutoGaze_layout": "repository-root",
}

vila = root / "external" / "VILA"
if (vila / ".git").exists():
    sources["VILA"] = subprocess.check_output(
        ["git", "-C", str(vila), "rev-parse", "HEAD"],
        text=True,
    ).strip()

out = root / "outputs" / "autogaze_repro" / "source_revisions.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sources, indent=2) + "\n")
print(json.dumps(sources, indent=2))
PY

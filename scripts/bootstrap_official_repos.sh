#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
AUTOGAZE_DIR="${EXTERNAL_DIR}/AutoGaze"
VILA_DIR="${EXTERNAL_DIR}/VILA"

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

clone_or_update "https://github.com/NVlabs/AutoGaze.git" "${AUTOGAZE_DIR}"

if [[ "${1:-}" == "--with-vila" ]]; then
  clone_or_update "https://github.com/NVlabs/VILA.git" "${VILA_DIR}"
fi

python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path.cwd()
sources = {}
for name in ["AutoGaze", "VILA"]:
    repo = root / "external" / name
    if (repo / ".git").exists():
        sources[name] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()

out = root / "outputs" / "autogaze_repro" / "source_revisions.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(sources, indent=2) + "\n")
print(json.dumps(sources, indent=2))
PY

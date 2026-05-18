from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "bfshi/AutoGaze"
DEFAULT_REVISION = "357a135349cb709217eeb7d083c1118df97d2cc7"
DEFAULT_FILES = [
    "example_inputs/doorbell.mp4",
    "example_inputs/tomjerry.mp4",
    "example_inputs/security.mp4",
]


def download_examples(repo: str, revision: str, output_dir: Path) -> list[dict[str, str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for filename in DEFAULT_FILES:
        source = Path(hf_hub_download(repo, filename, repo_type="space", revision=revision))
        target = output_dir / Path(filename).name
        shutil.copy2(source, target)
        rows.append(
            {
                "hub_path": filename,
                "local_path": str(target),
                "bytes": target.stat().st_size,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bfshi/AutoGaze Space example videos")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", default="inputs/hf_space_autogaze")
    args = parser.parse_args()

    rows = download_examples(args.repo, args.revision, Path(args.output_dir))
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

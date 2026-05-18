from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
DEFAULT_OUTPUT = "inputs/hlvid_example/clip_av_video_5_001.mp4"


def download_with_resume(url: str, output: Path, chunk_size: int) -> dict[str, int | str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.stat().st_size if output.exists() else 0
    request = urllib.request.Request(url)
    mode = "wb"
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        mode = "ab"

    with urllib.request.urlopen(request) as response, output.open(mode) as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    return {
        "url": url,
        "output": str(output),
        "existing_bytes_before": existing,
        "bytes_after": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the HLVid example video with resume support")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    result = download_with_resume(args.url, Path(args.output), args.chunk_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

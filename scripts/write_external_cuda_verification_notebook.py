from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "notebooks/autogaze_external_cuda_verification.ipynb"
DEFAULT_BRANCH = "codex/autogaze-vjepa"
DEFAULT_REPO = "https://github.com/manricheon/AutoGaze.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a Kaggle/Colab CUDA verification notebook for AutoGaze.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--platform", choices=["kaggle", "colab"], default="kaggle")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--output-root")
    parser.add_argument("--weights-root")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    platform_defaults = platform_paths(args.platform)
    notebook = build_notebook(
        branch=args.branch,
        platform=args.platform,
        repo_url=args.repo_url,
        output_root=args.output_root or platform_defaults["output_root"],
        weights_root=args.weights_root or platform_defaults["weights_root"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def platform_paths(platform: str) -> dict[str, str]:
    if platform == "colab":
        return {
            "work_root": "/content",
            "output_root": "/content/autogaze_vjepa_outputs",
            "weights_root": "/content/autogaze_weights",
        }
    return {
        "work_root": "/kaggle/working",
        "output_root": "/kaggle/working/autogaze_vjepa_outputs",
        "weights_root": "/kaggle/working/autogaze_weights",
    }


def build_notebook(
    *,
    branch: str,
    platform: str,
    output_root: str,
    weights_root: str,
    repo_url: str = DEFAULT_REPO,
) -> dict[str, Any]:
    paths = platform_paths(platform)
    work_root = paths["work_root"]
    cells = [
        markdown_cell(
            "# AutoGaze External CUDA Verification\n\n"
            "This notebook verifies the pushed AutoGaze reproduction branch on a CUDA runtime. "
            "It is intended for Kaggle or Colab after GPU access is enabled."
        ),
        code_cell(
            "import json, os, pathlib, subprocess, sys, textwrap, time\n"
            "import torch\n\n"
            "print('python:', sys.version)\n"
            "print('torch:', torch.__version__)\n"
            "print('cuda_available:', torch.cuda.is_available())\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('CUDA is required. Enable a Kaggle/Colab GPU runtime before continuing.')\n"
            "print('cuda_device:', torch.cuda.get_device_name(0))\n"
        ),
        code_cell(
            f"BRANCH = {branch!r}\n"
            f"REPO_URL = {repo_url!r}\n"
            f"WORK_ROOT = pathlib.Path({work_root!r})\n"
            f"OUTPUT_ROOT = pathlib.Path({output_root!r})\n"
            f"WEIGHTS_ROOT = pathlib.Path({weights_root!r})\n"
            f"REPORT_PATH = pathlib.Path({str(Path(output_root) / 'colab_verification.md')!r})\n"
            "REPO_DIR = WORK_ROOT / 'AutoGaze'\n"
            "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)\n\n"
            "def run(cmd, *, cwd=None):\n"
            "    cmd = [str(x) for x in cmd]\n"
            "    print('\\n$', ' '.join(cmd))\n"
            "    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)\n\n"
            "if not REPO_DIR.exists():\n"
            "    run(['git', 'clone', '--branch', BRANCH, REPO_URL, REPO_DIR])\n"
            "else:\n"
            "    run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_DIR)\n"
            "    run(['git', 'checkout', BRANCH], cwd=REPO_DIR)\n"
            "    run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO_DIR)\n"
            "os.chdir(REPO_DIR)\n"
            "run(['git', 'log', '--oneline', '-1'])\n"
        ),
        code_cell(
            "run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-repro.txt', "
            "'transformers>=4.57.0', 'qwen-vl-utils', 'av', 'pytest'])\n"
        ),
        code_cell(
            "run([\n"
            "    sys.executable, 'scripts/verify_autogaze_entrypoints.py',\n"
            "    '--output-json', OUTPUT_ROOT / 'entrypoint_verification.json',\n"
            "    '--output-md', OUTPUT_ROOT / 'entrypoint_verification.md',\n"
            "])\n"
        ),
        code_cell(
            "run([\n"
            "    sys.executable, 'scripts/run_colab_autogaze_cuda_smoke.py',\n"
            "    '--weights-root', WEIGHTS_ROOT,\n"
            "    '--output-root', OUTPUT_ROOT,\n"
            "    '--video', 'inputs/hlvid_example/clip_av_video_5_001.mp4',\n"
            "    '--num-video-frames', '16',\n"
            "    '--frames-per-clip', '16',\n"
            "    '--video-resize-longest-edge', '224',\n"
            "    '--max-new-tokens', '4',\n"
            "])\n"
        ),
        code_cell(
            "run([\n"
            "    sys.executable, '-m', 'repro.colab_verification_report',\n"
            "    '--output-md', REPORT_PATH,\n"
            "    '--title', 'AutoGaze External CUDA Verification',\n"
            "    '--video', 'inputs/hlvid_example/clip_av_video_5_001.mp4',\n"
            "    '--query', 'Describe the video in one short sentence.',\n"
            "    '--entrypoint-verification-json', OUTPUT_ROOT / 'entrypoint_verification.json',\n"
            "    '--case', f\"vjepa_qwen_dense_off={OUTPUT_ROOT / 'vjepa_qwen_dense_off_cuda_smoke.json'}\",\n"
            "    '--case', f\"autogaze_vjepa_qwen_on={OUTPUT_ROOT / 'autogaze_vjepa_qwen_on_cuda_smoke.json'}\",\n"
            "])\n"
        ),
        code_cell(
            "summary_path = OUTPUT_ROOT / 'colab_autogaze_cuda_smoke_summary.json'\n"
            "report_path = REPORT_PATH\n"
            "summary = json.loads(summary_path.read_text())\n"
            "print(json.dumps(summary['summary'], indent=2))\n"
            "print('report:', report_path)\n"
            "print('visualizations:', OUTPUT_ROOT / 'visualizations')\n"
            "assert summary['summary']['passed'] is True\n"
            "dense = summary['results']['vjepa_qwen_dense_off']\n"
            "ag = summary['results']['autogaze_vjepa_qwen_on']\n"
            "assert dense['status'] == 'passed'\n"
            "assert ag['status'] == 'passed'\n"
            "assert ag['tokens']['vjepa_selected_tokens'] < ag['tokens']['vjepa_raw_tokens']\n"
            "assert ag['tokens']['qwen_visual_tokens_inserted'] == ag['tokens']['vjepa_selected_tokens']\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": split_source(source)}


def code_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": split_source(source)}


def split_source(source: str) -> list[str]:
    return [line + "\n" for line in source.rstrip("\n").splitlines()]


if __name__ == "__main__":
    main()

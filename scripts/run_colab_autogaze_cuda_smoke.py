from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.colab_verification_report import render_colab_verification_markdown

DEFAULT_OUTPUT_ROOT = "/content/autogaze_vjepa_outputs"
DEFAULT_WEIGHTS_ROOT = "/content/autogaze_weights"
DEFAULT_VIDEO = "inputs/hlvid_example/clip_av_video_5_001.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Colab/H100 CUDA smoke used for AutoGaze + V-JEPA + Qwen verification. "
            "It runs the pre-CUDA entrypoint verifier and, by default, actual V-JEPA+Qwen "
            "dense/off plus AutoGaze/on generate paths."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-root", default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-entrypoint-verifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-vjepa-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dense-off", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-autogaze-on", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--video-resize-longest-edge", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--video-decode-strategy", default="seek", choices=["auto", "seek", "scan"])
    parser.add_argument("--autogaze-chunk-frames", type=int, default=16)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=4)
    parser.add_argument("--autogaze-tile-size", type=int, default=224)
    parser.add_argument("--autogaze-target-scales", default="32+64+112+224")
    parser.add_argument("--autogaze-target-patch-size", type=int, default=16)
    parser.add_argument("--prompt", default="Describe the video in one short sentence.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--verification-md", default=None)
    parser.add_argument("--write-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-output-dir")
    parser.add_argument("--visualization-max-frames", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_smoke(args)
    summary_json = Path(args.summary_json) if args.summary_json else Path(args.output_root) / "colab_autogaze_cuda_smoke_summary.json"
    if not args.dry_run:
        verification_md = Path(args.verification_md) if args.verification_md else Path(args.output_root) / "colab_verification.md"
        payload.setdefault("paths", {})["summary_json"] = str(summary_json)
        payload.setdefault("paths", {})["verification_md"] = str(verification_md)
        verification_md.parent.mkdir(parents=True, exist_ok=True)
        verification_md.write_text(render_colab_verification_markdown(payload, output_md=verification_md), encoding="utf-8")
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summarize_for_stdout(payload, summary_json), indent=2, sort_keys=True))
    if not payload["summary"]["passed"]:
        raise SystemExit(1)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    weights_root = Path(args.weights_root)
    python = str(args.python_executable)
    commands: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    if args.dry_run:
        return {
            "runner": "run_colab_autogaze_cuda_smoke",
            "dry_run": True,
            "summary": {"passed": True, "command_count": len(build_command_plan(args)), "elapsed_ms": 0.0},
            "command_plan": build_command_plan(args),
        }

    output_root.mkdir(parents=True, exist_ok=True)

    if args.download_missing:
        ensure_model_weights(
            python=python,
            repo_root=repo_root,
            weights_root=weights_root,
            commands=commands,
        )
        ensure_video(
            python=python,
            repo_root=repo_root,
            video_path=Path(args.video),
            commands=commands,
        )

    if args.run_entrypoint_verifier:
        verifier_json = output_root / "entrypoint_verification_colab_cuda_smoke.json"
        verifier_md = output_root / "entrypoint_verification_colab_cuda_smoke.md"
        commands.append(
            run_command(
                "entrypoint_verifier",
                [
                    python,
                    "scripts/verify_autogaze_entrypoints.py",
                    "--output-json",
                    str(verifier_json),
                    "--output-md",
                    str(verifier_md),
                ],
                cwd=repo_root,
            )
        )
        results["entrypoint_verifier"] = load_json_if_exists(verifier_json)

    if args.run_vjepa_qwen:
        if args.run_dense_off:
            dense_json = output_root / "vjepa_qwen_dense_off_cuda_smoke.json"
            dense_md = output_root / "vjepa_qwen_dense_off_cuda_smoke.md"
            commands.append(
                run_command(
                    "vjepa_qwen_dense_off",
                    vjepa_qwen_command(args, mode="off", output_json=dense_json, output_md=dense_md),
                    cwd=repo_root,
                )
            )
            results["vjepa_qwen_dense_off"] = load_json_if_exists(dense_json)
        if args.run_autogaze_on:
            autogaze_json = output_root / "autogaze_vjepa_qwen_on_cuda_smoke.json"
            autogaze_md = output_root / "autogaze_vjepa_qwen_on_cuda_smoke.md"
            commands.append(
                run_command(
                    "autogaze_vjepa_qwen_on",
                    vjepa_qwen_command(args, mode="on", output_json=autogaze_json, output_md=autogaze_md),
                    cwd=repo_root,
                )
            )
            results["autogaze_vjepa_qwen_on"] = load_json_if_exists(autogaze_json)

    failures = command_failures(commands) + result_failures(results)
    return {
        "runner": "run_colab_autogaze_cuda_smoke",
        "dry_run": False,
        "summary": {
            "passed": not failures,
            "command_count": len(commands),
            "failed_count": len(failures),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        },
        "paths": {
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "weights_root": str(weights_root),
            "video": str(args.video),
        },
        "prompt": str(args.prompt),
        "commands": commands,
        "results": compact_results(results),
        "failures": failures,
    }


def build_command_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if args.download_missing:
        plan.append({"name": "download_missing_weights", "command": download_weights_command(args)})
        plan.append({"name": "download_missing_video", "command": download_video_command(args)})
    if args.run_entrypoint_verifier:
        plan.append(
            {
                "name": "entrypoint_verifier",
                "command": [
                    str(args.python_executable),
                    "scripts/verify_autogaze_entrypoints.py",
                    "--output-json",
                    str(Path(args.output_root) / "entrypoint_verification_colab_cuda_smoke.json"),
                    "--output-md",
                    str(Path(args.output_root) / "entrypoint_verification_colab_cuda_smoke.md"),
                ],
            }
        )
    if args.run_vjepa_qwen and args.run_dense_off:
        plan.append(
            {
                "name": "vjepa_qwen_dense_off",
                "command": vjepa_qwen_command(
                    args,
                    mode="off",
                    output_json=Path(args.output_root) / "vjepa_qwen_dense_off_cuda_smoke.json",
                    output_md=Path(args.output_root) / "vjepa_qwen_dense_off_cuda_smoke.md",
                ),
            }
        )
    if args.run_vjepa_qwen and args.run_autogaze_on:
        plan.append(
            {
                "name": "autogaze_vjepa_qwen_on",
                "command": vjepa_qwen_command(
                    args,
                    mode="on",
                    output_json=Path(args.output_root) / "autogaze_vjepa_qwen_on_cuda_smoke.json",
                    output_md=Path(args.output_root) / "autogaze_vjepa_qwen_on_cuda_smoke.md",
                ),
            }
        )
    return plan


def vjepa_qwen_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output_json: Path,
    output_md: Path,
) -> list[str]:
    weights_root = Path(args.weights_root)
    command = [
        str(args.python_executable),
        "-m",
        "repro.vjepa_qwen_runner",
        "--video",
        str(args.video),
        "--prompt",
        str(args.prompt),
        "--vjepa-model",
        str(weights_root / "facebook__vjepa2-vitl-fpc64-256"),
        "--qwen-model",
        str(weights_root / "Qwen__Qwen2.5-VL-3B-Instruct"),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--num-video-frames",
        str(args.num_video_frames),
        "--frames-per-clip",
        str(args.frames_per_clip),
        "--video-decode-strategy",
        str(args.video_decode_strategy),
        "--video-resize-longest-edge",
        str(args.video_resize_longest_edge),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--autogaze-mode",
        mode,
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    if args.require_cuda:
        command.append("--require-cuda")
    if mode == "on":
        command.extend(
            [
                "--autogaze-model",
                str(weights_root / "nvidia__AutoGaze"),
                "--autogaze-device",
                str(args.device),
                "--autogaze-dtype",
                str(args.dtype),
                "--autogaze-chunk-frames",
                str(args.autogaze_chunk_frames),
                "--max-tiles-video",
                str(args.max_tiles_video),
                "--max-batch-size-autogaze",
                str(args.max_batch_size_autogaze),
                "--autogaze-tile-size",
                str(args.autogaze_tile_size),
                "--autogaze-target-scales",
                str(args.autogaze_target_scales),
                "--autogaze-target-patch-size",
                str(args.autogaze_target_patch_size),
            ]
        )
    if getattr(args, "write_visualizations", False):
        viz_root = Path(args.visualization_output_dir) if args.visualization_output_dir else Path(args.output_root) / "visualizations"
        command.extend(
            [
                "--visualization-output-dir",
                str(viz_root),
                "--visualization-max-frames",
                str(args.visualization_max_frames),
            ]
        )
    return command


def ensure_model_weights(*, python: str, repo_root: Path, weights_root: Path, commands: list[dict[str, Any]]) -> None:
    required = [
        weights_root / "nvidia__AutoGaze",
        weights_root / "facebook__vjepa2-vitl-fpc64-256",
        weights_root / "Qwen__Qwen2.5-VL-3B-Instruct",
    ]
    if all(path.exists() for path in required):
        return
    commands.append(run_command("download_vjepa_qwen_checkpoints", download_weights_command_from_paths(python, weights_root), cwd=repo_root))


def ensure_video(*, python: str, repo_root: Path, video_path: Path, commands: list[dict[str, Any]]) -> None:
    path = repo_root / video_path if not video_path.is_absolute() else video_path
    if path.exists():
        return
    commands.append(run_command("download_hlvid_example_video", download_video_command_from_paths(python, video_path), cwd=repo_root))


def download_weights_command(args: argparse.Namespace) -> list[str]:
    return download_weights_command_from_paths(str(args.python_executable), Path(args.weights_root))


def download_weights_command_from_paths(python: str, weights_root: Path) -> list[str]:
    return [python, "scripts/download_vjepa_qwen_checkpoints.py", "--output-root", str(weights_root), "--max-workers", "4"]


def download_video_command(args: argparse.Namespace) -> list[str]:
    return download_video_command_from_paths(str(args.python_executable), Path(args.video))


def download_video_command_from_paths(python: str, video_path: Path) -> list[str]:
    return [python, "scripts/download_hlvid_example_video.py", "--output", str(video_path)]


def run_command(name: str, command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": tail(proc.stdout),
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compact_results(results: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, payload in results.items():
        if payload is None:
            compact[name] = None
            continue
        if name == "entrypoint_verifier":
            compact[name] = compact_entrypoint_verifier_result(payload)
            continue
        compact[name] = {
            "status": payload.get("status") or payload.get("summary", {}).get("passed"),
            "summary": payload.get("summary"),
            "autogaze_mode": payload.get("autogaze_mode"),
            "tokens": payload.get("tokens"),
            "latency_ms": payload.get("latency_ms"),
            "memory_bytes": payload.get("memory_bytes"),
            "visualizations": payload.get("visualizations"),
            "failure": payload.get("failure"),
            "generated_text": payload.get("generated_text"),
        }
    return compact


def compact_entrypoint_verifier_result(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = payload.get("script_matrix") or []
    commands = payload.get("commands") or []
    return {
        "status": payload.get("summary", {}).get("passed"),
        "summary": payload.get("summary"),
        "verified_script_ids": [row.get("id") for row in matrix],
        "verified_entrypoints": [
            {
                "id": row.get("id"),
                "entrypoint": row.get("entrypoint"),
                "selector": row.get("selector"),
                "vit": row.get("vit"),
                "mllm": row.get("mllm"),
            }
            for row in matrix
        ],
        "dry_run_commands": [
            command.get("name")
            for command in commands
            if str(command.get("name", "")).endswith("_dry_run")
        ],
    }


def command_failures(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "command_failed",
            "name": item["name"],
            "returncode": item["returncode"],
            "stdout_tail": item.get("stdout_tail"),
        }
        for item in commands
        if int(item.get("returncode", 1)) != 0
    ]


def result_failures(results: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for name, payload in results.items():
        if payload is None:
            failures.append({"kind": "missing_result", "name": name})
            continue
        if name == "entrypoint_verifier":
            if not payload.get("summary", {}).get("passed", False):
                failures.append({"kind": "verification_failed", "name": name, "summary": payload.get("summary")})
        elif payload.get("status") != "passed":
            failures.append({"kind": "pipeline_failed", "name": name, "failure": payload.get("failure")})
    return failures


def summarize_for_stdout(payload: dict[str, Any], summary_json: Path) -> dict[str, Any]:
    return {
        "summary": payload.get("summary"),
        "summary_json": str(summary_json),
        "verification_md": (payload.get("paths") or {}).get("verification_md"),
        "results": payload.get("results"),
        "failures": payload.get("failures"),
    }


def tail(text: str, limit: int = 8000) -> str:
    return text[-limit:]


if __name__ == "__main__":
    main()

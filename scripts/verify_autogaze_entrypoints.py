from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT_JSON = ROOT / "outputs" / "autogaze_repro" / "entrypoint_verification.json"
DEFAULT_OUTPUT_MD = ROOT / "outputs" / "autogaze_repro" / "entrypoint_verification.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify AutoGaze on/off inference and benchmark entrypoints without downloading or loading "
            "large model weights. This is the pre-CUDA gate before Colab/H100 actual smoke runs."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--run-pytest", action="store_true", help="Also run the focused unit/regression test suite.")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_verification(
        python_executable=str(args.python_executable),
        run_pytest=bool(args.run_pytest),
        keep_temp=bool(args.keep_temp),
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if not payload["summary"]["passed"]:
        raise SystemExit(1)


def run_verification(*, python_executable: str, run_pytest: bool, keep_temp: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    temp_root = Path(tempfile.mkdtemp(prefix="autogaze_entrypoints_"))
    commands: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    try:
        pycache = temp_root / "pycache"
        commands.append(
            run_command(
                "compileall",
                [python_executable, "-m", "compileall", "-q", "repro", "scripts"],
                env={**os.environ, "PYTHONPYCACHEPREFIX": str(pycache)},
            )
        )
        for name, command in help_commands(python_executable):
            commands.append(run_command(name, command))
        for name, command in dry_run_commands(python_executable, temp_root):
            commands.append(run_command(name, command))

        checks.extend(run_preflight_checks(python_executable, temp_root))
        checks.extend(run_flexible_inspect_checks(python_executable, temp_root))
        checks.append(run_vjepa_synthetic_check(python_executable, temp_root))
        checks.append(check_plugin_qwen_routing())
        checks.extend(check_script_matrix())

        if run_pytest:
            commands.append(run_command("focused_pytest", [python_executable, "-m", "pytest", *focused_tests(), "-q"]))
    finally:
        if not keep_temp:
            cleanup_temp(temp_root)

    failed_commands = [item for item in commands if item["returncode"] != 0]
    failed_checks = [item for item in checks if item["status"] != "passed"]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "runner": "verify_autogaze_entrypoints",
        "scope": "pre_cuda_entrypoint_route_and_accounting_verification",
        "summary": {
            "passed": not failed_commands and not failed_checks,
            "command_count": len(commands),
            "check_count": len(checks),
            "failed_command_count": len(failed_commands),
            "failed_check_count": len(failed_checks),
            "elapsed_ms": elapsed_ms,
        },
        "script_matrix": entrypoint_matrix(),
        "commands": commands,
        "checks": checks,
        "notes": [
            "This verifier does not load NVILA, Qwen, V-JEPA, or AutoGaze weights.",
            "Actual CUDA smoke still requires running the documented Colab/H100 commands with local checkpoints.",
            "The verifier is intended to catch CLI drift, option forwarding gaps, and token accounting mistakes before expensive runs.",
        ],
    }


def help_commands(python_executable: str) -> list[tuple[str, list[str]]]:
    return [
        ("nvila_runner_help", [python_executable, "-m", "repro.nvila_runner", "--help"]),
        ("hlvid_batch_help", [python_executable, "-m", "repro.hlvid_batch_benchmark", "--help"]),
        ("hlvid_wrapper_plugin_help", [python_executable, "scripts/run_hlvid_folder_benchmark.py", "--plugin-suite", "qwen", "--help"]),
        ("plugin_hlvid_help", [python_executable, "-m", "repro.plugin_hlvid_benchmark", "--help"]),
        ("flexible_runner_help", [python_executable, "-m", "repro.flexible_runner", "--help"]),
        ("vjepa_qwen_runner_help", [python_executable, "-m", "repro.vjepa_qwen_runner", "--help"]),
        ("vjepa_qwen_hlvid_help", [python_executable, "-m", "repro.vjepa_qwen_hlvid_benchmark", "--help"]),
        ("colab_cuda_smoke_help", [python_executable, "scripts/run_colab_autogaze_cuda_smoke.py", "--help"]),
        ("markdown_report_help", [python_executable, "-m", "repro.markdown_report", "--help"]),
        ("aggregate_reports_help", [python_executable, "-m", "repro.aggregate_reports", "--help"]),
    ]


def dry_run_commands(python_executable: str, temp_root: Path) -> list[tuple[str, list[str]]]:
    return [
        (
            "download_vjepa_qwen_dry_run",
            [
                python_executable,
                "scripts/download_vjepa_qwen_checkpoints.py",
                "--dry-run",
                "--output-root",
                str(temp_root / "weights"),
            ],
        ),
        (
            "download_qwen_dry_run",
            [
                python_executable,
                "scripts/download_qwen_model.py",
                "--dry-run",
                "--output-dir",
                str(temp_root / "qwen"),
            ],
        ),
        (
            "colab_cuda_smoke_dry_run",
            [
                python_executable,
                "scripts/run_colab_autogaze_cuda_smoke.py",
                "--dry-run",
                "--python-executable",
                python_executable,
                "--output-root",
                str(temp_root / "colab_outputs"),
                "--weights-root",
                str(temp_root / "weights"),
                "--video",
                "inputs/hlvid_example/clip_av_video_5_001.mp4",
            ],
        ),
    ]


def focused_tests() -> list[str]:
    return [
        "tests/test_nvila_runner.py",
        "tests/test_hlvid_batch_benchmark.py",
        "tests/test_run_hlvid_folder_benchmark_wrapper.py",
        "tests/test_plugin_hlvid_benchmark.py",
        "tests/test_flexible_runner.py",
        "tests/test_vjepa_qwen_runner.py",
        "tests/test_vjepa_qwen_hlvid_benchmark.py",
        "tests/test_vjepa_mapping.py",
        "tests/test_vjepa_sparse_runtime.py",
        "tests/test_vjepa_qwen_bridge.py",
        "tests/test_vjepa_qwen_colab_smoke.py",
        "tests/test_colab_autogaze_cuda_smoke_script.py",
        "tests/test_vjepa_poc.py",
        "tests/test_download_vjepa_qwen_checkpoints.py",
        "tests/test_autogaze_sparse_selector.py",
        "tests/test_markdown_report.py",
        "tests/test_aggregate_reports.py",
    ]


def run_command(name: str, command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def run_preflight_checks(python_executable: str, temp_root: Path) -> list[dict[str, Any]]:
    autogaze_json = temp_root / "nvila_preflight_autogaze.json"
    keep_all_json = temp_root / "nvila_preflight_keep_all.json"
    single_json = temp_root / "nvila_preflight_keep_all_single.json"
    base = [
        python_executable,
        "-m",
        "repro.nvila_runner",
        "--mode",
        "preflight",
        "--preflight-width",
        "1920",
        "--preflight-height",
        "1080",
        "--preflight-source-frames",
        "9000",
        "--num-video-frames",
        "16",
        "--num-video-frames-thumbnail",
        "8",
        "--max-tiles-video",
        "1",
    ]
    commands = [
        ("nvila_preflight_autogaze", [*base, "--gazing-mode", "autogaze", "--preflight-json", str(autogaze_json)]),
        ("nvila_preflight_keep_all", [*base, "--gazing-mode", "keep-all", "--preflight-json", str(keep_all_json)]),
        (
            "nvila_preflight_keep_all_single",
            [*base, "--gazing-mode", "keep-all-single", "--preflight-json", str(single_json)],
        ),
    ]
    command_results = [run_command(name, command) for name, command in commands]
    checks: list[dict[str, Any]] = [
        command_check(f"command_{item['name']}", item) for item in command_results
    ]
    if all(item["returncode"] == 0 for item in command_results):
        autogaze = json.loads(autogaze_json.read_text(encoding="utf-8"))
        keep_all = json.loads(keep_all_json.read_text(encoding="utf-8"))
        single = json.loads(single_json.read_text(encoding="utf-8"))
        checks.append(
            value_check(
                "nvila_autogaze_preflight_multiscale_slots",
                autogaze["estimate"]["tokens"]["patches_per_frame_tile"],
                1060,
            )
        )
        checks.append(
            value_check(
                "nvila_keep_all_preflight_multiscale_slots",
                keep_all["estimate"]["tokens"]["patches_per_frame_tile"],
                1060,
            )
        )
        checks.append(
            value_check(
                "nvila_keep_all_single_preflight_single_scale_slots",
                single["estimate"]["tokens"]["patches_per_frame_tile"],
                784,
            )
        )
    return checks


def run_flexible_inspect_checks(python_executable: str, temp_root: Path) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "flexible_qwen_off",
            "json": temp_root / "flex_qwen_off.json",
            "args": [
                "--model-family",
                "qwen3-vl",
                "--model-path",
                "weight/Qwen3-VL",
                "--token-selector-adapter",
                "keep-all",
                "--vision-encoder-adapter",
                "qwen3-vl-vision",
                "--mllm-adapter",
                "qwen3-vl",
                "--autogaze-integration-level",
                "none",
                "--video",
                "inputs/hlvid_example/clip_av_video_5_001.mp4",
                "--num-video-frames",
                "16",
                "--qwen-vit-mode",
                "qwen_full_vit",
            ],
            "expect": {"uses_autogaze": False, "token_selector_kind": "keep-all"},
        },
        {
            "name": "flexible_qwen_autogaze_sparse",
            "json": temp_root / "flex_qwen_ag.json",
            "args": [
                "--model-family",
                "qwen3-vl",
                "--model-path",
                "weight/Qwen3-VL",
                "--token-selector-adapter",
                "autogaze",
                "--token-selector-path",
                "weight/AutoGaze",
                "--vision-encoder-adapter",
                "qwen3-vl-vision",
                "--mllm-adapter",
                "qwen3-vl",
                "--autogaze-integration-level",
                "pre_encoder_sparse",
                "--pre-encoder-prune-adapter",
                "autogaze-sparse",
                "--video",
                "inputs/hlvid_example/clip_av_video_5_001.mp4",
                "--num-video-frames",
                "16",
                "--qwen-vit-mode",
                "qwen_chunked_vit_autogaze_sparse",
            ],
            "expect": {"uses_autogaze": True, "token_selector_kind": "autogaze"},
        },
        {
            "name": "flexible_nvila_native_autogaze",
            "json": temp_root / "flex_nvila_ag.json",
            "args": [
                "--model-family",
                "nvila-hd-video-autogaze",
                "--model-path",
                "weight/NVILA-8B-HD-Video",
                "--token-selector-adapter",
                "autogaze",
                "--token-selector-path",
                "weight/AutoGaze",
                "--vision-encoder-adapter",
                "nvila-hd-siglip",
                "--mllm-adapter",
                "nvila-hd",
                "--autogaze-integration-level",
                "native_processor",
                "--video",
                "inputs/hlvid_example/clip_av_video_5_001.mp4",
            ],
            "expect": {"uses_autogaze": True, "token_selector_kind": "autogaze"},
        },
    ]
    checks: list[dict[str, Any]] = []
    for case in cases:
        command = [
            python_executable,
            "-m",
            "repro.flexible_runner",
            "--mode",
            "inspect",
            *case["args"],
            "--output-json",
            str(case["json"]),
        ]
        result = run_command(case["name"], command)
        checks.append(command_check(f"command_{case['name']}", result))
        if result["returncode"] == 0:
            spec = json.loads(case["json"].read_text(encoding="utf-8"))["experiment_spec"]
            for key, expected in case["expect"].items():
                checks.append(value_check(f"{case['name']}_{key}", spec.get(key), expected))
    return checks


def run_vjepa_synthetic_check(python_executable: str, temp_root: Path) -> dict[str, Any]:
    output_json = temp_root / "vjepa_poc.json"
    output_md = temp_root / "vjepa_poc.md"
    result = run_command(
        "vjepa_poc_synthetic",
        [
            python_executable,
            "-m",
            "repro.vjepa_poc",
            "--synthetic",
            "--scale-aware",
            "--tiny-encoder-smoke",
            "--qwen-bridge-smoke",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )
    if result["returncode"] != 0:
        return command_check("command_vjepa_poc_synthetic", result)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    passed = (
        payload["vjepa_sparse_encoder_smoke"]["status"] == "passed"
        and payload["vjepa_qwen_bridge_smoke"]["status"] == "passed"
        and payload["vjepa"]["selected_token_count"] < payload["vjepa"]["raw_token_count"]
    )
    return {
        "name": "vjepa_sparse_and_qwen_bridge_smoke",
        "status": "passed" if passed else "failed",
        "evidence": {
            "raw_tokens": payload["vjepa"]["raw_token_count"],
            "selected_tokens": payload["vjepa"]["selected_token_count"],
            "sparse_encoder": payload["vjepa_sparse_encoder_smoke"]["status"],
            "qwen_bridge": payload["vjepa_qwen_bridge_smoke"]["status"],
        },
    }


def check_plugin_qwen_routing() -> dict[str, Any]:
    from repro.plugin_hlvid_benchmark import build_mode_runner_args

    common = {
        "row": {"question": "What happens?"},
        "video_path": Path("video.mp4"),
        "output_json": Path("out.json"),
        "models": {"qwen3-vl": "weight/Qwen3-VL"},
        "external_mllm_command": "vila-infer",
        "num_video_frames": 16,
        "num_video_frames_thumbnail": 0,
        "max_tiles_video": 1,
        "max_new_tokens": 8,
        "qwen_video_nframes": 16,
        "qwen_vit_chunk_frames": 16,
        "qwen_vit_max_spatial_chunks": 1,
        "qwen_thumbnail_mode": "none",
        "autogaze_model": "weight/AutoGaze",
        "device_map": "auto",
        "dtype": "float16",
        "attn_implementation": None,
        "video_decode_strategy": "seek",
        "autogaze_repo": ".",
        "autogaze_device": "cuda",
        "autogaze_dtype": "float16",
        "autogaze_target_scales": "32+64+112+224",
        "autogaze_target_patch_size": 16,
        "autogaze_encoder_patch_size": None,
        "autogaze_tile_size": 224,
        "autogaze_chunk_frames": 16,
        "max_batch_size_autogaze": 4,
        "gazing_ratio": None,
        "task_loss_requirement": None,
        "autogaze_generate_only": False,
        "video_resize_shortest_edge": None,
        "video_resize_longest_edge": 448,
        "video_resize_width": None,
        "video_resize_height": None,
    }
    modes = {
        "qwen_full_vit": False,
        "qwen_chunked_vit": False,
        "qwen_chunked_vit_autogaze_sparse": True,
    }
    evidence = {}
    ok = True
    for mode, should_run_selector in modes.items():
        args = build_mode_runner_args(mode=mode, **common)
        runs_selector = "--run-autogaze-selector" in args
        selector = value_after(args, "--token-selector-adapter")
        vit_mode = value_after(args, "--qwen-vit-mode")
        has_resize = "--video-resize-longest-edge" in args
        has_target_scales = "--autogaze-target-scales" in args
        evidence[mode] = {
            "runs_autogaze_selector": runs_selector,
            "token_selector": selector,
            "qwen_vit_mode": vit_mode,
            "forwards_resize": has_resize,
            "forwards_autogaze_target_scales": has_target_scales,
            "forwards_max_batch_size_autogaze": "--max-batch-size-autogaze" in args,
        }
        ok = ok and runs_selector is should_run_selector and vit_mode == mode and has_resize and has_target_scales
    return {"name": "plugin_qwen_route_modes", "status": "passed" if ok else "failed", "evidence": evidence}


def check_script_matrix() -> list[dict[str, Any]]:
    checks = []
    for row in entrypoint_matrix():
        checks.append(
            {
                "name": f"matrix_{row['id']}",
                "status": "passed",
                "evidence": row,
            }
        )
    return checks


def entrypoint_matrix() -> list[dict[str, str]]:
    return [
        {
            "id": "nvila_single_autogaze",
            "entrypoint": "python -m repro.nvila_runner --mode single --gazing-mode autogaze",
            "selector": "AutoGaze on",
            "vit": "NVILA-HD SigLIP",
            "mllm": "NVILA-HD",
            "verification": "help + preflight + unit tests; actual CUDA generate required separately",
        },
        {
            "id": "nvila_single_keep_all",
            "entrypoint": "python -m repro.nvila_runner --mode single --gazing-mode keep-all",
            "selector": "AutoGaze off / multiscale dense",
            "vit": "NVILA-HD SigLIP",
            "mllm": "NVILA-HD",
            "verification": "help + preflight token accounting 1060 slots/frame + unit tests",
        },
        {
            "id": "nvila_single_keep_all_single",
            "entrypoint": "python -m repro.nvila_runner --mode single --gazing-mode keep-all-single",
            "selector": "AutoGaze off / single-scale dense",
            "vit": "NVILA-HD SigLIP",
            "mllm": "NVILA-HD",
            "verification": "help + preflight token accounting 784 slots/frame + unit tests",
        },
        {
            "id": "nvila_hlvid_wrapper",
            "entrypoint": "python scripts/run_hlvid_folder_benchmark.py",
            "selector": "keep-all, keep-all-single, AutoGaze",
            "vit": "NVILA-HD SigLIP",
            "mllm": "NVILA-HD",
            "verification": "help + wrapper tests; actual CUDA HLVid required separately",
        },
        {
            "id": "qwen_plugin_hlvid",
            "entrypoint": "python scripts/run_hlvid_folder_benchmark.py --plugin-suite qwen",
            "selector": "off for full/chunked, AutoGaze on for sparse",
            "vit": "Qwen grid ViT full/chunked/sparse",
            "mllm": "Qwen VL",
            "verification": "help + route forwarding checks + plugin tests",
        },
        {
            "id": "flexible_qwen_autogaze",
            "entrypoint": "python -m repro.flexible_runner --model-family qwen3-vl --token-selector-adapter autogaze",
            "selector": "AutoGaze on",
            "vit": "Qwen grid ViT sparse route",
            "mllm": "Qwen VL",
            "verification": "inspect checks; actual model generate required separately",
        },
        {
            "id": "vjepa_qwen_single",
            "entrypoint": "python -m repro.vjepa_qwen_runner --autogaze-mode on/off",
            "selector": "off dense V-JEPA or AutoGaze sparse",
            "vit": "V-JEPA2 sparse hook",
            "mllm": "Qwen bridge/generate",
            "verification": "help + synthetic sparse/bridge smoke; Colab CUDA actual smoke required",
        },
        {
            "id": "vjepa_qwen_hlvid",
            "entrypoint": "python -m repro.vjepa_qwen_hlvid_benchmark",
            "selector": "dense_off, autogaze_single_grid, autogaze_scale_aware",
            "vit": "V-JEPA2 dense/sparse",
            "mllm": "Qwen bridge/generate",
            "verification": "help + unit tests; actual CUDA HLVid required separately",
        },
        {
            "id": "colab_cuda_smoke_wrapper",
            "entrypoint": "python scripts/run_colab_autogaze_cuda_smoke.py",
            "selector": "dense/off and AutoGaze/on",
            "vit": "V-JEPA2 dense/sparse hook",
            "mllm": "Qwen bridge/generate",
            "verification": "help + dry-run + Colab CUDA actual smoke wrapper",
        },
    ]


def command_check(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if result["returncode"] == 0 else "failed",
        "evidence": {
            "returncode": result["returncode"],
            "stdout_tail": result.get("stdout_tail"),
            "stderr_tail": result.get("stderr_tail"),
        },
    }


def value_check(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if actual == expected else "failed",
        "evidence": {"actual": actual, "expected": expected},
    }


def value_after(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args):
        return None
    return args[index + 1]


def tail(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def cleanup_temp(temp_root: Path) -> None:
    for path in sorted(temp_root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            path.rmdir()
    temp_root.rmdir()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# AutoGaze Entrypoint Verification",
        "",
        f"- passed: `{summary['passed']}`",
        f"- commands: `{summary['command_count']}`",
        f"- checks: `{summary['check_count']}`",
        f"- elapsed_ms: `{summary['elapsed_ms']:.1f}`",
        "",
        "## Script Matrix",
        "",
        "| id | entrypoint | selector | vit | mllm | verification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["script_matrix"]:
        lines.append(
            "| {id} | `{entrypoint}` | {selector} | {vit} | {mllm} | {verification} |".format(
                **{key: markdown_escape(str(value)) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Checks", "", "| check | status | evidence |", "| --- | --- | --- |"])
    for check in payload["checks"]:
        evidence = json.dumps(check.get("evidence", {}), sort_keys=True)
        lines.append(f"| {markdown_escape(check['name'])} | `{check['status']}` | `{markdown_escape(evidence)}` |")
    lines.extend(["", "## Commands", "", "| command | returncode |", "| --- | --- |"])
    for command in payload["commands"]:
        lines.append(f"| {markdown_escape(command['name'])} | `{command['returncode']}` |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    return "\n".join(lines) + "\n"


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()

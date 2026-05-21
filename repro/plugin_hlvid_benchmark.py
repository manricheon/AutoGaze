from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_json, write_jsonl
from repro.flexible_runner import parse_args as parse_flexible_args
from repro.flexible_runner import run_single
from repro.hlvid import read_manifest_file, score_predictions


DEFAULT_MODELS = {
    "nvila-video": "weight/NVILA-8B-Video",
    "longvila": "weight/LongVILA",
    "internvl3": "weight/InternVL3",
    "qwen3-vl": "weight/Qwen3-VL-8B-Instruct",
}


def resolve_hlvid_video_path(video_root: str | Path, row_video_path: str) -> Path:
    root = Path(video_root)
    direct = root / row_video_path
    if direct.exists():
        return direct
    flat = root / Path(row_video_path).name
    if flat.exists():
        return flat
    return direct


def build_mode_runner_args(
    *,
    mode: str,
    row: dict[str, Any],
    video_path: Path,
    output_json: Path,
    models: dict[str, str],
    external_mllm_command: str,
    num_video_frames: int,
    max_tiles_video: int,
    max_new_tokens: int,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_chunk_frames: int = 16,
    qwen_vit_max_spatial_chunks: int | None = None,
) -> list[str]:
    if mode == "nvila-video-off":
        return _base_args(
            model_family="nvila-video-plugin",
            model_path=models.get("nvila-video", DEFAULT_MODELS["nvila-video"]),
            token_selector="keep-all",
            vision_adapter="nvila-video-vision",
            mllm_adapter="nvila-video",
            integration_level="none",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode == "nvila-video-autogaze-probe":
        return _base_args(
            model_family="nvila-video-plugin",
            model_path=models.get("nvila-video", DEFAULT_MODELS["nvila-video"]),
            token_selector="autogaze",
            vision_adapter="nvila-video-vision",
            mllm_adapter="nvila-video",
            integration_level="post_encoder_token_prune",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode == "longvila-off":
        return _base_args(
            model_family="longvila",
            model_path=models.get("longvila", DEFAULT_MODELS["longvila"]),
            token_selector="keep-all",
            vision_adapter="longvila-siglip",
            mllm_adapter="longvila",
            integration_level="none",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode == "longvila-autogaze-probe":
        return _base_args(
            model_family="longvila",
            model_path=models.get("longvila", DEFAULT_MODELS["longvila"]),
            token_selector="autogaze",
            vision_adapter="longvila-siglip",
            mllm_adapter="longvila",
            integration_level="post_encoder_token_prune",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode == "internvl3-off":
        return _base_args(
            model_family="internvl3",
            model_path=models.get("internvl3", DEFAULT_MODELS["internvl3"]),
            token_selector="keep-all",
            vision_adapter="internvl-dynamic-vision",
            mllm_adapter="internvl3",
            integration_level="none",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode == "qwen3-vl-off":
        return _base_args(
            model_family="qwen3-vl",
            model_path=models.get("qwen3-vl", DEFAULT_MODELS["qwen3-vl"]),
            token_selector="keep-all",
            vision_adapter="qwen3-vl-vision",
            mllm_adapter="qwen3-vl",
            integration_level="none",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
    if mode in {"qwen_full_vit", "qwen_chunked_vit", "qwen_chunked_vit_autogaze_sparse"}:
        sparse = mode == "qwen_chunked_vit_autogaze_sparse"
        args = _base_args(
            model_family="qwen3-vl",
            model_path=models.get("qwen3-vl", DEFAULT_MODELS["qwen3-vl"]),
            token_selector="autogaze" if sparse else "keep-all",
            vision_adapter="qwen3-vl-vision",
            mllm_adapter="qwen3-vl",
            integration_level="pre_encoder_sparse" if sparse else "none",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_vit_mode=mode,
            qwen_vit_chunk_frames=qwen_vit_chunk_frames,
            qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks or max_tiles_video,
        )
        if sparse:
            args.extend(
                [
                    "--run-autogaze-selector",
                    "--autogaze-generate-only",
                    "--enable-qwen-prune-generate",
                    "--pre-encoder-prune-adapter",
                    "autogaze-sparse",
                ]
            )
        return args
    if mode in {
        "qwen3-vl-autogaze-probe",
        "qwen3-vl-autogaze-poc",
        "qwen3-vl-autogaze-prune-generate",
        "qwen3-vl-autogaze-direct-prune-generate",
        "qwen3-vl-autogaze-direct-pre-vit-sparse",
    }:
        integration_level = "pre_encoder_sparse" if mode == "qwen3-vl-autogaze-direct-pre-vit-sparse" else "post_encoder_token_prune"
        args = _base_args(
            model_family="qwen3-vl",
            model_path=models.get("qwen3-vl", DEFAULT_MODELS["qwen3-vl"]),
            token_selector="autogaze",
            vision_adapter="qwen3-vl-vision",
            mllm_adapter="qwen3-vl",
            integration_level=integration_level,
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
        if mode in {"qwen3-vl-autogaze-prune-generate", "qwen3-vl-autogaze-direct-prune-generate"}:
            args.append("--enable-qwen-prune-generate")
        if mode in {"qwen3-vl-autogaze-direct-prune-generate", "qwen3-vl-autogaze-direct-pre-vit-sparse"}:
            args.extend(["--run-autogaze-selector", "--autogaze-generate-only"])
        if mode == "qwen3-vl-autogaze-direct-pre-vit-sparse":
            args.extend(["--enable-qwen-prune-generate", "--pre-encoder-prune-adapter", "autogaze-sparse"])
        return args
    if mode == "qwen3-vl-pixelprune-pre-vit":
        args = _base_args(
            model_family="qwen3-vl",
            model_path=models.get("qwen3-vl", DEFAULT_MODELS["qwen3-vl"]),
            token_selector="keep-all",
            vision_adapter="qwen3-vl-vision",
            mllm_adapter="qwen3-vl",
            integration_level="pre_encoder_sparse",
            row=row,
            video_path=video_path,
            output_json=output_json,
            external_mllm_command=external_mllm_command,
            num_video_frames=num_video_frames,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        )
        args.extend(["--pre-encoder-prune-adapter", "pixelprune"])
        return args
    raise ValueError(f"Unsupported plugin HLVid mode: {mode}")


def run_plugin_hlvid_benchmark(
    *,
    manifest: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    modes: list[str],
    models: dict[str, str] | None = None,
    external_mllm_command: str = "vila-infer",
    limit: int | None = None,
    num_video_frames: int = 256,
    max_tiles_video: int = 8,
    max_new_tokens: int = 8,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_chunk_frames: int = 16,
    qwen_vit_max_spatial_chunks: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest_file(manifest)
    if limit is not None:
        rows = rows[:limit]
    model_paths = {**DEFAULT_MODELS, **(models or {})}
    predictions: list[dict[str, Any]] = []

    for mode in modes:
        for row_index, row in enumerate(rows):
            video_path = resolve_hlvid_video_path(video_root, row["video_path"])
            run_json = output / "runs" / mode / f"{row_index:05d}.json"
            runner_args = build_mode_runner_args(
                mode=mode,
                row=row,
                video_path=video_path,
                output_json=run_json,
                models=model_paths,
                external_mllm_command=external_mllm_command,
                num_video_frames=num_video_frames,
                max_tiles_video=max_tiles_video,
                max_new_tokens=max_new_tokens,
                qwen_video_nframes=qwen_video_nframes,
                qwen_video_fps=qwen_video_fps,
                qwen_video_max_pixels=qwen_video_max_pixels,
                qwen_video_min_pixels=qwen_video_min_pixels,
                qwen_vit_chunk_frames=qwen_vit_chunk_frames,
                qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks,
            )
            payload = run_single(parse_flexible_args(runner_args))
            generation = payload.get("generation", {})
            metrics = generation.get("metrics", {})
            prediction = {
                "mode": mode,
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "video_path": row.get("video_path"),
                "resolved_video_path": str(video_path),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "raw_output": generation.get("text"),
                "status": _prediction_status(generation.get("status")),
                "runner_status": payload.get("implementation_status"),
                "metric_status": metrics.get("metric_status"),
                "metrics": metrics,
            }
            prediction.update(_flatten_key_metrics(metrics))
            predictions.append(prediction)

    summary = _summarize_by_mode(predictions)
    predictions_path = output / "plugin_hlvid_predictions.jsonl"
    summary_path = output / "plugin_hlvid_summary.json"
    report_path = output / "plugin_hlvid_report.md"
    write_jsonl(predictions_path, predictions)
    write_json(summary_path, summary)
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    return {
        "predictions": predictions,
        "summary": summary,
        "artifacts": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "markdown": str(report_path),
        },
    }


def build_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Plugin HLVid Limit Benchmark",
        "",
        "| mode | total | correct | failed | parse_failed | accuracy_total | accuracy_scored | status_counts | next_action |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for mode, mode_summary in summary["modes"].items():
        lines.append(
            "| {mode} | {total} | {correct} | {failed} | {parse_failed} | {accuracy_total:.4f} | {accuracy_scored:.4f} | {status_counts} | {next_action} |".format(
                mode=mode,
                total=mode_summary["total"],
                correct=mode_summary["correct"],
                failed=mode_summary["failed"],
                parse_failed=mode_summary["parse_failed"],
                accuracy_total=mode_summary["accuracy_total"],
                accuracy_scored=mode_summary["accuracy_scored"],
                status_counts=json.dumps(mode_summary.get("status_counts", {}), sort_keys=True),
                next_action=mode_summary.get("next_action"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `nvila-video-off` and `longvila-off` use the official VILA CLI adapter.",
            "- `nvila-video-autogaze-probe` and `longvila-autogaze-probe` record AutoGaze-on feature packing probes.",
            "- `qwen3-vl-pixelprune-pre-vit` applies PixelPrune before Qwen model load and then runs the native Qwen path.",
            "- `qwen3-vl-autogaze-poc` records a Qwen AutoGaze post-encoder attachment PoC until visual token packing is wired.",
            "- `qwen3-vl-autogaze-prune-generate` explicitly enables the experimental Qwen post-encoder prune + generate path.",
            "- `qwen3-vl-autogaze-direct-prune-generate` runs AutoGaze first, writes a sparse plan, then uses that plan for Qwen prune + generate.",
            "- `qwen3-vl-autogaze-direct-pre-vit-sparse` additionally installs the experimental Qwen sparse vision hook before MLLM packing.",
            "- `qwen_full_vit`, `qwen_chunked_vit`, and `qwen_chunked_vit_autogaze_sparse` compare native full Qwen ViT, temporal chunked Qwen ViT, and temporal chunked AutoGaze sparse Qwen ViT.",
            "- `accuracy_total` uses all rows in the denominator; failed and parse-failed rows are separated in the table.",
        ]
    )
    return "\n".join(lines) + "\n"


def _base_args(
    *,
    model_family: str,
    model_path: str,
    token_selector: str,
    vision_adapter: str,
    mllm_adapter: str,
    integration_level: str,
    row: dict[str, Any],
    video_path: Path,
    output_json: Path,
    external_mllm_command: str,
    num_video_frames: int,
    max_tiles_video: int,
    max_new_tokens: int,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_mode: str | None = None,
    qwen_vit_chunk_frames: int | None = None,
    qwen_vit_max_spatial_chunks: int | None = None,
) -> list[str]:
    args = [
        "--mode",
        "single",
        "--model-family",
        model_family,
        "--model-path",
        model_path,
        "--token-selector-adapter",
        token_selector,
        "--vision-encoder-adapter",
        vision_adapter,
        "--mllm-adapter",
        mllm_adapter,
        "--autogaze-integration-level",
        integration_level,
        "--prompt",
        str(row["question"]),
        "--video",
        str(video_path),
        "--num-video-frames",
        str(num_video_frames),
        "--max-tiles-video",
        str(max_tiles_video),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-json",
        str(output_json),
    ]
    if token_selector == "autogaze":
        args.extend(["--token-selector-path", "weight/AutoGaze"])
        args.extend(["--gazing-ratio", "0.1"])
    if model_family.startswith("qwen"):
        if qwen_video_nframes is not None:
            args.extend(["--qwen-video-nframes", str(qwen_video_nframes)])
        if qwen_video_fps is not None:
            args.extend(["--qwen-video-fps", str(qwen_video_fps)])
        if qwen_video_max_pixels is not None:
            args.extend(["--qwen-video-max-pixels", str(qwen_video_max_pixels)])
        if qwen_video_min_pixels is not None:
            args.extend(["--qwen-video-min-pixels", str(qwen_video_min_pixels)])
        if qwen_vit_mode is not None:
            args.extend(["--qwen-vit-mode", str(qwen_vit_mode)])
        if qwen_vit_chunk_frames is not None:
            args.extend(["--qwen-vit-chunk-frames", str(qwen_vit_chunk_frames)])
        if qwen_vit_max_spatial_chunks is not None:
            args.extend(["--qwen-vit-max-spatial-chunks", str(qwen_vit_max_spatial_chunks)])
    if external_mllm_command:
        args.extend(["--external-mllm-command", external_mllm_command])
    return args


def _prediction_status(generation_status: str | None) -> str:
    if generation_status in {"executed", "probe_required", "probe_collected", "poc_ready"}:
        return "ok"
    return "failed"


def _flatten_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    latency = metrics.get("latency_ms", {})
    memory = metrics.get("memory_bytes", {})
    tokens = metrics.get("tokens", {})
    qwen_vit = metrics.get("qwen_vit", {})
    spatial_chunking = qwen_vit.get("spatial_chunking") or {}
    tile_grid = spatial_chunking.get("tile_grid") or {}
    return {
        "total_ms": latency.get("total"),
        "generate_ms": latency.get("generate"),
        "qwen_vit_prepare_ms": latency.get("qwen_vit_prepare"),
        "llm_peak_memory_bytes": memory.get("peak_cuda_allocated"),
        "peak_memory_bytes": memory.get("peak_cuda_reserved"),
        "qwen_vit_mode": qwen_vit.get("mode"),
        "qwen_vit_raw_patch_tokens_before_vit": qwen_vit.get("raw_patch_tokens_before_vit"),
        "qwen_vit_chunk_count": qwen_vit.get("chunk_count"),
        "qwen_vit_executed_chunk_count": qwen_vit.get("executed_chunk_count"),
        "qwen_vit_spatial_tiles": tile_grid.get("tiles"),
        "visual_tokens_before_prune": tokens.get("visual_tokens_before_prune"),
        "visual_tokens_after_prune": tokens.get("visual_tokens_after_prune"),
        "visual_token_reduction_ratio": tokens.get("visual_token_reduction_ratio"),
        "llm_context_tokens": tokens.get("llm_context_tokens"),
    }


def _summarize_by_mode(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({row["mode"] for row in predictions})
    summaries: dict[str, Any] = {}
    for mode in modes:
        rows = [row for row in predictions if row["mode"] == mode]
        summary, _ = score_predictions(rows)
        summary["status_counts"] = _status_counts(rows)
        summary["next_action"] = _next_action_for_mode(mode, rows)
        summaries[mode] = summary
    return {"modes": summaries, "mode_order": modes, "total_predictions": len(predictions)}


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("runner_status") or row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _next_action_for_mode(mode: str, rows: list[dict[str, Any]]) -> str:
    statuses = _status_counts(rows)
    if mode in {"nvila-video-autogaze-probe", "longvila-autogaze-probe"}:
        if statuses.get("probe_collected"):
            return "instrument_vila_remote_code_feature_packing"
        return "run_vila_feature_packing_probe"
    if mode == "qwen3-vl-autogaze-prune-generate":
        if statuses.get("executed"):
            return "score_qwen_pruned_generation"
        return "inspect_qwen_prune_generate_failure"
    if mode in {"qwen3-vl-autogaze-poc", "qwen3-vl-autogaze-probe"}:
        return "implement_qwen_visual_feature_prune_generate"
    if mode == "qwen3-vl-pixelprune-pre-vit" and statuses.get("failed_missing_dependency"):
        return "install_pixelprune_and_rerun"
    if any(status.startswith("failed") for status in statuses):
        return "fix_failed_runtime_dependency"
    if statuses.get("executed"):
        return "score_and_compare_metrics"
    return "inspect_outputs"


def parse_model_overrides(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Model override must be name=path: {value}")
        name, path = value.split("=", 1)
        parsed[name] = path
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run flexible_runner plugin modes on an HLVid manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", default="outputs/autogaze_repro/plugin_hlvid_limit3")
    parser.add_argument("--modes", default="nvila-video-off,longvila-off,internvl3-off,qwen3-vl-off")
    parser.add_argument("--model", action="append", help="Override model path as adapter=path, e.g. nvila-video=weight/NVILA")
    parser.add_argument("--external-mllm-command", default="vila-infer")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--num-video-frames", type=int, default=256)
    parser.add_argument("--max-tiles-video", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--qwen-video-nframes", type=int)
    parser.add_argument("--qwen-video-fps", type=float)
    parser.add_argument("--qwen-video-max-pixels", type=int)
    parser.add_argument("--qwen-video-min-pixels", type=int)
    parser.add_argument("--qwen-vit-chunk-frames", type=int, default=16)
    parser.add_argument("--qwen-vit-max-spatial-chunks", type=int)
    args = parser.parse_args()
    payload = run_plugin_hlvid_benchmark(
        manifest=args.manifest,
        video_root=args.video_root,
        output_dir=args.output_dir,
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        models=parse_model_overrides(args.model),
        external_mllm_command=args.external_mllm_command,
        limit=args.limit,
        num_video_frames=args.num_video_frames,
        max_tiles_video=args.max_tiles_video,
        max_new_tokens=args.max_new_tokens,
        qwen_video_nframes=args.qwen_video_nframes,
        qwen_video_fps=args.qwen_video_fps,
        qwen_video_max_pixels=args.qwen_video_max_pixels,
        qwen_video_min_pixels=args.qwen_video_min_pixels,
        qwen_vit_chunk_frames=args.qwen_vit_chunk_frames,
        qwen_vit_max_spatial_chunks=args.qwen_vit_max_spatial_chunks,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

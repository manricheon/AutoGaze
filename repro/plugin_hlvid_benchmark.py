from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import compute_stats, write_json, write_jsonl
from repro.failure_logging import classify_exception, failure_generation_payload
from repro.flexible_runner import parse_args as parse_flexible_args
from repro.flexible_runner import run_single
from repro.hlvid import read_manifest_file, score_predictions
from repro.report_charts import ChartBar, ChartSegment, write_bar_chart


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
    num_video_frames_thumbnail: int = 0,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_chunk_frames: int = 16,
    qwen_vit_max_spatial_chunks: int | None = None,
    qwen_thumbnail_mode: str = "none",
    autogaze_model: str = "weight/AutoGaze",
    device_map: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    video_decode_strategy: str = "auto",
    autogaze_repo: str = ".",
    autogaze_device: str = "auto",
    autogaze_dtype: str = "auto",
    autogaze_target_scales: str | None = None,
    autogaze_target_patch_size: int | None = None,
    autogaze_encoder_patch_size: int | None = None,
    autogaze_tile_size: int | None = None,
    autogaze_chunk_frames: int | None = None,
    max_batch_size_autogaze: int | None = None,
    gazing_ratio: float | None = None,
    task_loss_requirement: float | None = None,
    autogaze_generate_only: bool = False,
    video_resize_shortest_edge: int | None = None,
    video_resize_longest_edge: int | None = None,
    video_resize_width: int | None = None,
    video_resize_height: int | None = None,
) -> list[str]:
    common_runner_options = {
        "device_map": device_map,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "video_decode_strategy": video_decode_strategy,
        "autogaze_repo": autogaze_repo,
        "autogaze_device": autogaze_device,
        "autogaze_dtype": autogaze_dtype,
        "autogaze_target_scales": autogaze_target_scales,
        "autogaze_target_patch_size": autogaze_target_patch_size,
        "autogaze_encoder_patch_size": autogaze_encoder_patch_size,
        "autogaze_tile_size": autogaze_tile_size,
        "autogaze_chunk_frames": autogaze_chunk_frames,
        "max_batch_size_autogaze": max_batch_size_autogaze,
        "gazing_ratio": gazing_ratio,
        "task_loss_requirement": task_loss_requirement,
        "autogaze_generate_only": autogaze_generate_only,
    }

    def with_common(args: list[str]) -> list[str]:
        return _append_common_runner_options(args, **common_runner_options)

    if mode == "nvila-video-off":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
    if mode == "nvila-video-autogaze-probe":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
    if mode == "longvila-off":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
    if mode == "longvila-autogaze-probe":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
    if mode == "internvl3-off":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
        ))
    if mode == "qwen3-vl-off":
        return with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
    if mode in {"qwen_full_vit", "qwen_chunked_vit", "qwen_chunked_vit_autogaze_sparse"}:
        sparse = mode == "qwen_chunked_vit_autogaze_sparse"
        args = with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_vit_mode=mode,
            qwen_vit_chunk_frames=qwen_vit_chunk_frames,
            qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks or max_tiles_video,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
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
        args = with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
        if mode in {"qwen3-vl-autogaze-prune-generate", "qwen3-vl-autogaze-direct-prune-generate"}:
            args.append("--enable-qwen-prune-generate")
        if mode in {"qwen3-vl-autogaze-direct-prune-generate", "qwen3-vl-autogaze-direct-pre-vit-sparse"}:
            args.extend(["--run-autogaze-selector", "--autogaze-generate-only"])
        if mode == "qwen3-vl-autogaze-direct-pre-vit-sparse":
            args.extend(["--enable-qwen-prune-generate", "--pre-encoder-prune-adapter", "autogaze-sparse"])
        return args
    if mode == "qwen3-vl-pixelprune-pre-vit":
        args = with_common(_base_args(
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
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            max_new_tokens=max_new_tokens,
            qwen_video_nframes=qwen_video_nframes,
            qwen_video_fps=qwen_video_fps,
            qwen_video_max_pixels=qwen_video_max_pixels,
            qwen_video_min_pixels=qwen_video_min_pixels,
            qwen_thumbnail_mode=qwen_thumbnail_mode,
            autogaze_model=autogaze_model,
            video_resize_shortest_edge=video_resize_shortest_edge,
            video_resize_longest_edge=video_resize_longest_edge,
            video_resize_width=video_resize_width,
            video_resize_height=video_resize_height,
        ))
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
    num_video_frames_thumbnail: int = 0,
    max_tiles_video: int = 8,
    max_new_tokens: int = 8,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_chunk_frames: int = 16,
    qwen_vit_max_spatial_chunks: int | None = None,
    qwen_thumbnail_mode: str = "none",
    autogaze_model: str = "weight/AutoGaze",
    device_map: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    video_decode_strategy: str = "auto",
    autogaze_repo: str = ".",
    autogaze_device: str = "auto",
    autogaze_dtype: str = "auto",
    autogaze_target_scales: str | None = None,
    autogaze_target_patch_size: int | None = None,
    autogaze_encoder_patch_size: int | None = None,
    autogaze_tile_size: int | None = None,
    autogaze_chunk_frames: int | None = None,
    max_batch_size_autogaze: int | None = None,
    gazing_ratio: float | None = None,
    task_loss_requirement: float | None = None,
    autogaze_generate_only: bool = False,
    video_resize_shortest_edge: int | None = None,
    video_resize_longest_edge: int | None = None,
    video_resize_width: int | None = None,
    video_resize_height: int | None = None,
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
                num_video_frames_thumbnail=num_video_frames_thumbnail,
                max_tiles_video=max_tiles_video,
                max_new_tokens=max_new_tokens,
                qwen_video_nframes=qwen_video_nframes,
                qwen_video_fps=qwen_video_fps,
                qwen_video_max_pixels=qwen_video_max_pixels,
                qwen_video_min_pixels=qwen_video_min_pixels,
                qwen_vit_chunk_frames=qwen_vit_chunk_frames,
                qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks,
                qwen_thumbnail_mode=qwen_thumbnail_mode,
                autogaze_model=autogaze_model,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_implementation,
                video_decode_strategy=video_decode_strategy,
                autogaze_repo=autogaze_repo,
                autogaze_device=autogaze_device,
                autogaze_dtype=autogaze_dtype,
                autogaze_target_scales=autogaze_target_scales,
                autogaze_target_patch_size=autogaze_target_patch_size,
                autogaze_encoder_patch_size=autogaze_encoder_patch_size,
                autogaze_tile_size=autogaze_tile_size,
                autogaze_chunk_frames=autogaze_chunk_frames,
                max_batch_size_autogaze=max_batch_size_autogaze,
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
                autogaze_generate_only=autogaze_generate_only,
                video_resize_shortest_edge=video_resize_shortest_edge,
                video_resize_longest_edge=video_resize_longest_edge,
                video_resize_width=video_resize_width,
                video_resize_height=video_resize_height,
            )
            parsed_args = parse_flexible_args(runner_args)
            try:
                payload = run_single(parsed_args)
            except Exception as exc:
                failure = classify_exception(exc, stage="mllm_generate")
                payload = {
                    "runner": "flexible_runner",
                    "mode": "single",
                    "implementation_status": failure["kind"],
                    "failure": failure,
                    "generation": failure_generation_payload(parsed_args, failure),
                }
                write_json(run_json, payload)
                if failure["kind"] != "oom":
                    raise
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
                "failure": payload.get("failure") or generation.get("failure"),
                "metric_status": metrics.get("metric_status"),
                "metrics": metrics,
                "processing_budget_summary": metrics.get("processing_budget_summary"),
            }
            prediction.update(_flatten_key_metrics(metrics))
            predictions.append(prediction)

    summary = _summarize_by_mode(predictions)
    predictions_path = output / "plugin_hlvid_predictions.jsonl"
    summary_path = output / "plugin_hlvid_summary.json"
    report_path = output / "plugin_hlvid_report.md"
    assets_dir = output / "plugin_hlvid_report_assets"
    write_jsonl(predictions_path, predictions)
    write_json(summary_path, summary)
    report_path.write_text(build_markdown_report(summary, assets_dir=assets_dir, report_dir=output), encoding="utf-8")
    return {
        "predictions": predictions,
        "summary": summary,
        "artifacts": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "markdown": str(report_path),
        },
    }


def build_markdown_report(
    summary: dict[str, Any],
    *,
    assets_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> str:
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
    budget_rows = []
    for mode, mode_summary in summary["modes"].items():
        budget = mode_summary.get("processing_budget_summary", {})
        mode_median = budget.get("mode_median", {}) if isinstance(budget, dict) else {}
        if not mode_median:
            continue
        budget_rows.append(
            [
                mode,
                mode_median.get("video.source_resolution"),
                mode_median.get("video.processor_input_resolution"),
                mode_median.get("video.requested_video_frames"),
                first_present(
                    mode_median.get("single_scale_dense_vision_budget.total_patch_tokens"),
                    mode_median.get("single_scale_dense_vision_budget.estimated_total_patch_tokens"),
                ),
                first_present(
                    mode_median.get("patch_budget_before_siglip.keep_all_total_patch_tokens"),
                    mode_median.get("patch_budget_before_vit.actual_raw_patch_tokens_before_vit"),
                    mode_median.get("patch_budget_before_vit.estimated_visual_tokens_before_prune"),
                ),
                first_present(
                    mode_median.get("patch_budget_before_siglip.autogaze_selected_total_patch_tokens"),
                    mode_median.get("patch_budget_before_vit.estimated_visual_tokens_after_prune"),
                ),
                first_present(
                    mode_median.get("patch_budget_before_siglip.total_patch_reduction_ratio"),
                    mode_median.get("patch_budget_before_vit.estimated_visual_token_reduction_ratio"),
                ),
                mode_median.get("llm_visual_budget.actual_visual_tokens"),
            ]
        )
    if budget_rows:
        lines.extend(
            [
                "",
                "## Processing Budget By Mode",
                "",
                "| mode | source_resolution | processor_input_resolution | frames | single-scale SigLIP patch ref | full/off patch tokens | selected/actual patch tokens | reduction_ratio | llm_visual_tokens |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in budget_rows:
            lines.append(
                "| "
                + " | ".join(_markdown_cell(value) for value in row)
                + " |"
            )
    if assets_dir is not None:
        chart_paths = _write_plugin_summary_charts(summary, Path(assets_dir))
        if chart_paths:
            base = Path(report_dir) if report_dir is not None else Path(assets_dir).parent
            lines.extend(["", "## Charts", ""])
            for title, path in chart_paths:
                display = path.relative_to(base) if path.is_relative_to(base) else path
                lines.extend([f"### {title}", "", f"![{title}]({display})", ""])
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


def _write_plugin_summary_charts(summary: dict[str, Any], assets_dir: Path) -> list[tuple[str, Path]]:
    modes = summary.get("modes", {})
    if not isinstance(modes, dict):
        return []
    charts: list[tuple[str, Path]] = []
    latency_bars = []
    memory_bars = []
    token_bars = []
    status_bars = []
    for mode, mode_summary in modes.items():
        budget = mode_summary.get("processing_budget_summary", {})
        mode_median = budget.get("mode_median", {}) if isinstance(budget, dict) else {}
        selected = first_present(
            mode_median.get("patch_budget_before_siglip.autogaze_selected_total_patch_tokens"),
            mode_median.get("patch_budget_before_vit.estimated_visual_tokens_after_prune"),
        )
        reduction = first_present(
            mode_median.get("patch_budget_before_siglip.total_patch_reduction_ratio"),
            mode_median.get("patch_budget_before_vit.estimated_visual_token_reduction_ratio"),
        )
        if selected is not None:
            token_bars.append(ChartBar(str(mode), [ChartSegment("selected_patch_tokens", float(selected))]))
        if reduction is not None:
            status_bars.append(ChartBar(f"{mode}:token_reduction", [ChartSegment("token_reduction_ratio", float(reduction))]))
        status_counts = mode_summary.get("status_counts", {})
        if isinstance(status_counts, dict):
            for status, count in status_counts.items():
                status_bars.append(ChartBar(f"{mode}:{status}", [ChartSegment(str(status), float(count))]))
        latency = mode_summary.get("latency_ms", {})
        if isinstance(latency, dict) and latency.get("total", {}).get("median") is not None:
            latency_bars.append(ChartBar(str(mode), [ChartSegment("total_ms", float(latency["total"]["median"]))]))
        memory = mode_summary.get("memory_bytes", {})
        if isinstance(memory, dict) and memory.get("peak_memory_bytes", {}).get("median") is not None:
            memory_bars.append(
                ChartBar(str(mode), [ChartSegment("peak_memory_bytes", float(memory["peak_memory_bytes"]["median"]))])
            )
    if latency_bars:
        artifact = write_bar_chart(assets_dir / "latency_by_mode.svg", title="Latency By Mode", bars=latency_bars, unit="ms")
        charts.append((artifact.title, artifact.path))
    if token_bars:
        artifact = write_bar_chart(
            assets_dir / "selected_tokens_by_mode.svg",
            title="Selected Tokens By Mode",
            bars=token_bars,
            unit="tokens",
        )
        charts.append((artifact.title, artifact.path))
    if memory_bars:
        artifact = write_bar_chart(assets_dir / "memory_by_mode.svg", title="Memory By Mode", bars=memory_bars, unit="bytes")
        charts.append((artifact.title, artifact.path))
    if status_bars:
        artifact = write_bar_chart(assets_dir / "status_by_mode.svg", title="Status By Mode", bars=status_bars, unit="count")
        charts.append((artifact.title, artifact.path))
    return charts


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True).replace("|", "\\|")
    return str(value).replace("|", "\\|")


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
    num_video_frames_thumbnail: int = 0,
    qwen_thumbnail_mode: str = "none",
    autogaze_model: str = "weight/AutoGaze",
    video_resize_shortest_edge: int | None = None,
    video_resize_longest_edge: int | None = None,
    video_resize_width: int | None = None,
    video_resize_height: int | None = None,
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
        "--num-video-frames-thumbnail",
        str(num_video_frames_thumbnail),
        "--max-tiles-video",
        str(max_tiles_video),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-json",
        str(output_json),
    ]
    if token_selector == "autogaze":
        args.extend(["--token-selector-path", autogaze_model])
        args.extend(["--autogaze-model", autogaze_model])
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
        if qwen_thumbnail_mode != "none":
            args.extend(["--qwen-thumbnail-mode", str(qwen_thumbnail_mode)])
    if video_resize_shortest_edge is not None:
        args.extend(["--video-resize-shortest-edge", str(video_resize_shortest_edge)])
    if video_resize_longest_edge is not None:
        args.extend(["--video-resize-longest-edge", str(video_resize_longest_edge)])
    if video_resize_width is not None:
        args.extend(["--video-resize-width", str(video_resize_width)])
    if video_resize_height is not None:
        args.extend(["--video-resize-height", str(video_resize_height)])
    if external_mllm_command:
        args.extend(["--external-mllm-command", external_mllm_command])
    return args


def _append_common_runner_options(
    args: list[str],
    *,
    device_map: str = "auto",
    dtype: str = "auto",
    attn_implementation: str | None = None,
    video_decode_strategy: str = "auto",
    autogaze_repo: str = ".",
    autogaze_device: str = "auto",
    autogaze_dtype: str = "auto",
    autogaze_target_scales: str | None = None,
    autogaze_target_patch_size: int | None = None,
    autogaze_encoder_patch_size: int | None = None,
    autogaze_tile_size: int | None = None,
    autogaze_chunk_frames: int | None = None,
    max_batch_size_autogaze: int | None = None,
    gazing_ratio: float | None = None,
    task_loss_requirement: float | None = None,
    autogaze_generate_only: bool = False,
) -> list[str]:
    output = list(args)

    def remove_value(flag: str) -> None:
        while flag in output:
            index = output.index(flag)
            del output[index : index + 2]

    def add_value(flag: str, value: Any, *, default: Any = None) -> None:
        if value is not None and value != default:
            remove_value(flag)
            output.extend([flag, str(value)])

    def add_flag(flag: str, enabled: bool) -> None:
        if enabled and flag not in output:
            output.append(flag)

    add_value("--device-map", device_map, default="auto")
    add_value("--dtype", dtype, default="auto")
    add_value("--attn-implementation", attn_implementation)
    add_value("--video-decode-strategy", video_decode_strategy, default="auto")
    add_value("--autogaze-repo", autogaze_repo, default=".")
    add_value("--autogaze-device", autogaze_device, default="auto")
    add_value("--autogaze-dtype", autogaze_dtype, default="auto")
    add_value("--autogaze-target-scales", autogaze_target_scales)
    add_value("--autogaze-target-patch-size", autogaze_target_patch_size)
    add_value("--autogaze-encoder-patch-size", autogaze_encoder_patch_size)
    add_value("--autogaze-tile-size", autogaze_tile_size)
    add_value("--autogaze-chunk-frames", autogaze_chunk_frames)
    add_value("--max-batch-size-autogaze", max_batch_size_autogaze)
    add_value("--gazing-ratio", gazing_ratio)
    add_value("--task-loss-requirement", task_loss_requirement)
    add_flag("--autogaze-generate-only", autogaze_generate_only)
    return output


def _prediction_status(generation_status: str | None) -> str:
    if generation_status == "oom":
        return "oom"
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
        summary["processing_budget_summary"] = _summarize_processing_budget_by_mode(rows)
        summaries[mode] = summary
    return {"modes": summaries, "mode_order": modes, "total_predictions": len(predictions)}


def _summarize_processing_budget_by_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    budgets = [
        row.get("processing_budget_summary") or row.get("metrics", {}).get("processing_budget_summary")
        for row in rows
    ]
    budgets = [budget for budget in budgets if isinstance(budget, dict)]
    if not budgets:
        return {
            "mode_median": {},
            "fields": [],
            "note": "No per-row metrics.processing_budget_summary was available for this mode.",
        }
    fields = sorted({field for budget in budgets for field, _ in _iter_leaf_values(budget)})
    mode_median = {
        field: _summarize_leaf_values([_get_nested_value(budget, field) for budget in budgets])
        for field in fields
    }
    return {
        "mode_median": mode_median,
        "fields": fields,
        "note": (
            "Numeric values are medians across rows. String, boolean, list, and dict values use the first "
            "non-null row. Full per-sample details remain in plugin_hlvid_predictions.jsonl."
        ),
    }


def _iter_leaf_values(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_values(child, path)
        return
    yield prefix, value


def _get_nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _summarize_leaf_values(values: list[Any]) -> Any:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    numeric: list[float] = []
    for value in non_null:
        if isinstance(value, bool):
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if numeric and len(numeric) == len(non_null):
        median = compute_stats(numeric)["median"]
        return int(median) if float(median).is_integer() else median
    return non_null[0]


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
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=0)
    parser.add_argument("--max-tiles-video", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--qwen-video-nframes", type=int)
    parser.add_argument("--qwen-video-fps", type=float)
    parser.add_argument("--qwen-video-max-pixels", type=int)
    parser.add_argument("--qwen-video-min-pixels", type=int)
    parser.add_argument("--qwen-vit-chunk-frames", type=int, default=16)
    parser.add_argument("--qwen-vit-max-spatial-chunks", type=int)
    parser.add_argument("--qwen-thumbnail-mode", choices=["none", "append-video"], default="none")
    parser.add_argument("--autogaze-model", default="weight/AutoGaze")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--video-decode-strategy", choices=["auto", "seek", "scan"], default="auto")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--autogaze-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--autogaze-target-scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--autogaze-encoder-patch-size", type=int)
    parser.add_argument("--autogaze-tile-size", type=int)
    parser.add_argument("--autogaze-chunk-frames", type=int)
    parser.add_argument("--max-batch-size-autogaze", type=int)
    parser.add_argument("--gazing-ratio", "--gazing-ratio-tile", dest="gazing_ratio", type=float)
    parser.add_argument("--task-loss-requirement", "--task-loss-requirement-tile", dest="task_loss_requirement", type=float)
    parser.add_argument("--autogaze-generate-only", action="store_true")
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
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
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        max_new_tokens=args.max_new_tokens,
        qwen_video_nframes=args.qwen_video_nframes,
        qwen_video_fps=args.qwen_video_fps,
        qwen_video_max_pixels=args.qwen_video_max_pixels,
        qwen_video_min_pixels=args.qwen_video_min_pixels,
        qwen_vit_chunk_frames=args.qwen_vit_chunk_frames,
        qwen_vit_max_spatial_chunks=args.qwen_vit_max_spatial_chunks,
        qwen_thumbnail_mode=args.qwen_thumbnail_mode,
        autogaze_model=args.autogaze_model,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        video_decode_strategy=args.video_decode_strategy,
        autogaze_repo=args.autogaze_repo,
        autogaze_device=args.autogaze_device or args.device or "auto",
        autogaze_dtype=args.autogaze_dtype,
        autogaze_target_scales=args.autogaze_target_scales,
        autogaze_target_patch_size=args.autogaze_target_patch_size,
        autogaze_encoder_patch_size=args.autogaze_encoder_patch_size,
        autogaze_tile_size=args.autogaze_tile_size,
        autogaze_chunk_frames=args.autogaze_chunk_frames,
        max_batch_size_autogaze=args.max_batch_size_autogaze,
        gazing_ratio=args.gazing_ratio,
        task_loss_requirement=args.task_loss_requirement,
        autogaze_generate_only=args.autogaze_generate_only,
        video_resize_shortest_edge=args.video_resize_shortest_edge,
        video_resize_longest_edge=args.video_resize_longest_edge,
        video_resize_width=args.video_resize_width,
        video_resize_height=args.video_resize_height,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from repro.common import write_csv, write_json
from repro.nvila_runner import (
    apply_resize_to_dimensions,
    estimate_stream_profile_plan,
    parse_int_sequence,
    read_video_metadata,
    spatial_tile_grid,
)


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    num_video_frames: int
    num_video_frames_thumbnail: int
    max_tiles_video: int
    stream_chunk_frames: int
    max_batch_size_autogaze: int
    max_batch_size_siglip: int
    video_resize_shortest_edge: int | None = None


def default_candidate_matrix() -> list[SweepCandidate]:
    return [
        SweepCandidate(
            name="fast_448p_1tile_64f",
            num_video_frames=64,
            num_video_frames_thumbnail=32,
            max_tiles_video=1,
            stream_chunk_frames=16,
            max_batch_size_autogaze=4,
            max_batch_size_siglip=16,
            video_resize_shortest_edge=448,
        ),
        SweepCandidate(
            name="fast_720p_2tile_64f",
            num_video_frames=64,
            num_video_frames_thumbnail=32,
            max_tiles_video=4,
            stream_chunk_frames=16,
            max_batch_size_autogaze=4,
            max_batch_size_siglip=16,
            video_resize_shortest_edge=720,
        ),
        SweepCandidate(
            name="balanced_720p_8tile_128f",
            num_video_frames=128,
            num_video_frames_thumbnail=64,
            max_tiles_video=8,
            stream_chunk_frames=16,
            max_batch_size_autogaze=8,
            max_batch_size_siglip=16,
            video_resize_shortest_edge=720,
        ),
        SweepCandidate(
            name="balanced_1080p_15tile_128f",
            num_video_frames=128,
            num_video_frames_thumbnail=64,
            max_tiles_video=16,
            stream_chunk_frames=16,
            max_batch_size_autogaze=8,
            max_batch_size_siglip=16,
            video_resize_shortest_edge=1080,
        ),
        SweepCandidate(
            name="quality_1080p_28tile_256f",
            num_video_frames=256,
            num_video_frames_thumbnail=64,
            max_tiles_video=32,
            stream_chunk_frames=16,
            max_batch_size_autogaze=8,
            max_batch_size_siglip=16,
            video_resize_shortest_edge=1080,
        ),
        SweepCandidate(
            name="quality_native_48tile_256f",
            num_video_frames=256,
            num_video_frames_thumbnail=64,
            max_tiles_video=48,
            stream_chunk_frames=16,
            max_batch_size_autogaze=16,
            max_batch_size_siglip=32,
            video_resize_shortest_edge=None,
        ),
        SweepCandidate(
            name="paper_native_48tile_1024f",
            num_video_frames=1024,
            num_video_frames_thumbnail=64,
            max_tiles_video=48,
            stream_chunk_frames=16,
            max_batch_size_autogaze=16,
            max_batch_size_siglip=32,
            video_resize_shortest_edge=None,
        ),
    ]


def _candidate_output_path(out_dir: str | Path, candidate: SweepCandidate, gazing_mode: str) -> Path:
    return Path(out_dir) / f"{candidate.name}_{gazing_mode}.json"


def _latency_proxy(plan: dict[str, Any]) -> float:
    chunking = plan["chunking"]
    memory = plan["memory"]
    tokens = plan["tokens"]
    return (
        float(chunking["tile_sequences"])
        + float(tokens["encoder_raw_tile_patch_tokens"]) / 100000.0
        + float(memory["streaming_raw_frame_buffer_bytes"]) / (256 * 1024 * 1024)
        + float(memory["streaming_autogaze_tile_tensor_bytes_per_batch"]) / (256 * 1024 * 1024)
    )


def build_command(
    *,
    video: str,
    candidate: SweepCandidate,
    device: str,
    stream_dtype: str,
    gazing_mode: str,
    output_json: str | Path,
    stream_run_siglip: bool = False,
    stream_siglip_mode: str = "gazed",
    stream_siglip_model: str | None = None,
    stream_siglip_max_embed_batch_size: int | None = None,
    stream_decode_strategy: str = "scan",
    autogaze_target_scales: str | None = None,
    autogaze_target_patch_size: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "repro.nvila_runner",
        "--mode",
        "stream-profile",
        "--device",
        device,
        "--stream-dtype",
        stream_dtype,
        "--video",
        video,
        "--gazing-mode",
        gazing_mode,
        "--num-video-frames",
        str(candidate.num_video_frames),
        "--num-video-frames-thumbnail",
        str(candidate.num_video_frames_thumbnail),
        "--max-tiles-video",
        str(candidate.max_tiles_video),
        "--stream-chunk-frames",
        str(candidate.stream_chunk_frames),
        "--max-batch-size-autogaze",
        str(candidate.max_batch_size_autogaze),
        "--max-batch-size-siglip",
        str(candidate.max_batch_size_siglip),
        "--stream-profile-json",
        str(output_json),
    ]
    if stream_decode_strategy != "scan":
        command.extend(["--stream-decode-strategy", stream_decode_strategy])
    if autogaze_target_scales:
        command.extend(["--autogaze-target-scales", autogaze_target_scales])
    if autogaze_target_patch_size is not None:
        command.extend(["--autogaze-target-patch-size", str(autogaze_target_patch_size)])
    if stream_run_siglip:
        command.extend(["--stream-run-siglip", "--stream-siglip-mode", stream_siglip_mode])
        if stream_siglip_model:
            command.extend(["--stream-siglip-model", stream_siglip_model])
        if stream_siglip_max_embed_batch_size is not None:
            command.extend(["--stream-siglip-max-embed-batch-size", str(stream_siglip_max_embed_batch_size)])
    if candidate.video_resize_shortest_edge is not None:
        command.extend(["--video-resize-shortest-edge", str(candidate.video_resize_shortest_edge)])
    return command


def build_recommendation_rows(
    *,
    video: str,
    width: int,
    height: int,
    source_frames: int | None,
    candidates: list[SweepCandidate],
    device: str,
    stream_dtype: str,
    out_dir: str | Path,
    gazing_mode: str = "autogaze",
    stream_run_siglip: bool = False,
    stream_siglip_mode: str = "gazed",
    stream_siglip_model: str | None = None,
    stream_siglip_max_embed_batch_size: int | None = None,
    stream_decode_strategy: str = "scan",
    autogaze_target_scales: str | None = None,
    autogaze_target_patch_size: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    parsed_scales = parse_int_sequence(autogaze_target_scales)
    plan_overrides: dict[str, Any] = {}
    if parsed_scales:
        plan_overrides["scales"] = parsed_scales
    if autogaze_target_patch_size is not None:
        plan_overrides["patch_size"] = autogaze_target_patch_size
    for candidate in candidates:
        effective = apply_resize_to_dimensions(
            width=width,
            height=height,
            shortest_edge=candidate.video_resize_shortest_edge,
            longest_edge=None,
            exact_width=None,
            exact_height=None,
        )
        plan = estimate_stream_profile_plan(
            width=int(effective["width"]),
            height=int(effective["height"]),
            source_frames=source_frames,
            num_video_frames=candidate.num_video_frames,
            num_video_frames_thumbnail=candidate.num_video_frames_thumbnail,
            max_tiles_video=candidate.max_tiles_video,
            chunk_frames=candidate.stream_chunk_frames,
            max_batch_size_autogaze=candidate.max_batch_size_autogaze,
            **plan_overrides,
        )
        grid = spatial_tile_grid(int(effective["width"]), int(effective["height"]), candidate.max_tiles_video)
        output_json = _candidate_output_path(out_dir, candidate, gazing_mode)
        command = build_command(
            video=video,
            candidate=candidate,
            device=device,
            stream_dtype=stream_dtype,
            gazing_mode=gazing_mode,
            output_json=output_json,
            stream_run_siglip=stream_run_siglip,
            stream_siglip_mode=stream_siglip_mode,
            stream_siglip_model=stream_siglip_model,
            stream_siglip_max_embed_batch_size=stream_siglip_max_embed_batch_size,
            stream_decode_strategy=stream_decode_strategy,
            autogaze_target_scales=autogaze_target_scales,
            autogaze_target_patch_size=autogaze_target_patch_size,
        )
        rows.append(
            {
                "candidate": candidate.name,
                "video": video,
                "gazing_mode": gazing_mode,
                "effective_width": int(effective["width"]),
                "effective_height": int(effective["height"]),
                "resize_shortest_edge": candidate.video_resize_shortest_edge,
                "spatial_tiles": int(grid["tiles"]),
                "num_video_frames": candidate.num_video_frames,
                "num_video_frames_thumbnail": candidate.num_video_frames_thumbnail,
                "stream_chunk_frames": candidate.stream_chunk_frames,
                "max_batch_size_autogaze": candidate.max_batch_size_autogaze,
                "max_batch_size_siglip": candidate.max_batch_size_siglip,
                "stream_decode_strategy": stream_decode_strategy,
                "stream_run_siglip": stream_run_siglip,
                "stream_siglip_mode": stream_siglip_mode if stream_run_siglip else None,
                "stream_siglip_model": stream_siglip_model if stream_run_siglip else None,
                "stream_siglip_max_embed_batch_size": stream_siglip_max_embed_batch_size if stream_run_siglip else None,
                "autogaze_target_scales": autogaze_target_scales,
                "autogaze_target_patch_size": autogaze_target_patch_size,
                "tile_sequences": int(plan["chunking"]["tile_sequences"]),
                "encoder_raw_patch_tokens": int(plan["tokens"]["encoder_raw_patch_tokens"]),
                "encoder_raw_tile_patch_tokens": int(plan["tokens"]["encoder_raw_tile_patch_tokens"]),
                "encoder_raw_thumbnail_patch_tokens": int(plan["tokens"]["encoder_raw_thumbnail_patch_tokens"]),
                "llm_keep_all_visual_tokens_estimated": int(plan["tokens"]["llm_keep_all_visual_tokens_estimated"]),
                "streaming_raw_frame_buffer_bytes": int(plan["memory"]["streaming_raw_frame_buffer_bytes"]),
                "streaming_autogaze_tile_tensor_bytes_per_batch": int(
                    plan["memory"]["streaming_autogaze_tile_tensor_bytes_per_batch"]
                ),
                "streaming_autogaze_tile_tensor_bytes_full_chunk": int(
                    plan["memory"]["streaming_autogaze_tile_tensor_bytes_full_chunk"]
                ),
                "latency_proxy": _latency_proxy(plan),
                "output_json": str(output_json),
                "command": shlex.join(command),
            }
        )
    return rows


def _row_with_result(row: dict[str, Any], result_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(result_path).read_text())
    timing = payload.get("timing_ms", {})
    memory = payload.get("memory_bytes", {})
    tokens = payload.get("token_metrics", {})
    gaze = payload.get("gaze", {})
    sampling = payload.get("sampling", {})
    return {
        **row,
        "status": "ok",
        "measured_pre_llm_stream_total_ms": timing.get("pre_llm_stream_total_measured"),
        "measured_keyframe_index_ms": timing.get("video_keyframe_index_scan"),
        "measured_seek_ms": timing.get("video_seek"),
        "measured_decode_seek_ms": timing.get("video_decode_seek"),
        "measured_decode_ms": timing.get("video_decode_scan"),
        "measured_tile_build_ms": timing.get("spatial_tile_build"),
        "measured_autogaze_tensorize_ms": timing.get("tile_autogaze_tensorize"),
        "measured_autogaze_forward_ms": timing.get("tile_autogaze_forward"),
        "measured_siglip_gazed_forward_ms": timing.get("siglip_gazed_forward"),
        "measured_siglip_keep_all_forward_ms": timing.get("siglip_keep_all_forward"),
        "measured_raw_frame_buffer_peak": memory.get("raw_frame_buffer_peak"),
        "measured_tile_pil_buffer_peak": memory.get("tile_pil_buffer_peak"),
        "measured_autogaze_tensor_peak": memory.get("autogaze_tile_tensor_peak_per_temporal_chunk"),
        "measured_siglip_gazed_hidden_peak": memory.get("siglip_gazed_hidden_peak"),
        "measured_siglip_keep_all_hidden_peak": memory.get("siglip_keep_all_hidden_peak"),
        "encoder_autogaze_selected_patch_tokens": tokens.get("encoder_autogaze_selected_patch_tokens"),
        "encoder_token_reduction_ratio": tokens.get("encoder_token_reduction_ratio"),
        "encoder_tile_token_reduction_ratio": tokens.get("encoder_tile_token_reduction_ratio"),
        "llm_autogaze_visual_tokens_lower_bound_estimated": tokens.get(
            "llm_autogaze_visual_tokens_lower_bound_estimated"
        ),
        "gaze_token_reduction_ratio": gaze.get("token_reduction_ratio"),
        "measured_decode_strategy": sampling.get("decode_strategy"),
        "measured_decode_frames_read": sampling.get("decode_frames_read"),
        "measured_decode_seek_groups": sampling.get("decode_seek_groups"),
    }


def run_candidates(rows: list[dict[str, Any]], *, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        command = shlex.split(row["command"])
        output_path = Path(row["output_json"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    **row,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "stderr_tail": str(exc)[-2000:],
                }
            )
            continue
        if completed.returncode == 0 and output_path.exists():
            results.append(_row_with_result(row, output_path))
        else:
            results.append(
                {
                    **row,
                    "status": "failed",
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
    return results


def rank_recommendations(
    rows: list[dict[str, Any]],
    *,
    max_visual_tokens: int,
    max_stream_memory_bytes: int,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if int(row["llm_keep_all_visual_tokens_estimated"]) <= max_visual_tokens
        and max(int(row["streaming_raw_frame_buffer_bytes"]), int(row["streaming_autogaze_tile_tensor_bytes_per_batch"]))
        <= max_stream_memory_bytes
        and row.get("status", "ok") == "ok"
    ]
    return sorted(filtered, key=lambda row: float(row.get("measured_pre_llm_stream_total_ms") or row["latency_proxy"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally run NVILA stream-profile sweep candidates.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--stream-dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--stream-decode-strategy", default="scan", choices=["scan", "seek"])
    parser.add_argument("--gazing-mode", default="autogaze", choices=["autogaze", "keep-all"])
    parser.add_argument("--out-dir", default="outputs/autogaze_repro/stream_sweep")
    parser.add_argument("--summary-json", default="outputs/autogaze_repro/stream_sweep_summary.json")
    parser.add_argument("--summary-csv", default="outputs/autogaze_repro/stream_sweep_summary.csv")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--stream-run-siglip", action="store_true")
    parser.add_argument("--stream-siglip-mode", default="gazed", choices=["gazed", "keep-all", "both"])
    parser.add_argument("--stream-siglip-model")
    parser.add_argument("--stream-siglip-max-embed-batch-size", type=int, default=1)
    parser.add_argument("--autogaze-target-scales", "--autogaze-resize-scales", dest="autogaze_target_scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--max-visual-tokens", type=int, default=750000)
    parser.add_argument("--max-stream-memory-gib", type=float, default=8.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    metadata = read_video_metadata(args.video)
    candidates = default_candidate_matrix()
    if args.include:
        candidates = [candidate for candidate in candidates if any(item in candidate.name for item in args.include)]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    rows = build_recommendation_rows(
        video=args.video,
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        source_frames=metadata["frames"],
        candidates=candidates,
        device=args.device,
        stream_dtype=args.stream_dtype,
        out_dir=args.out_dir,
        gazing_mode=args.gazing_mode,
        stream_decode_strategy=args.stream_decode_strategy,
        stream_run_siglip=args.stream_run_siglip,
        stream_siglip_mode=args.stream_siglip_mode,
        stream_siglip_model=args.stream_siglip_model,
        stream_siglip_max_embed_batch_size=args.stream_siglip_max_embed_batch_size,
        autogaze_target_scales=args.autogaze_target_scales,
        autogaze_target_patch_size=args.autogaze_target_patch_size,
    )
    if args.run:
        rows = run_candidates(rows, timeout_seconds=args.timeout_seconds)

    ranked = rank_recommendations(
        rows,
        max_visual_tokens=args.max_visual_tokens,
        max_stream_memory_bytes=int(args.max_stream_memory_gib * 1024**3),
    )
    payload = {
        "video": args.video,
        "source_metadata": metadata,
        "device": args.device,
        "stream_dtype": args.stream_dtype,
        "stream_decode_strategy": args.stream_decode_strategy,
        "gazing_mode": args.gazing_mode,
        "stream_run_siglip": args.stream_run_siglip,
        "stream_siglip_mode": args.stream_siglip_mode if args.stream_run_siglip else None,
        "stream_siglip_model": args.stream_siglip_model if args.stream_run_siglip else None,
        "autogaze_target_scales": args.autogaze_target_scales,
        "autogaze_target_patch_size": args.autogaze_target_patch_size,
        "rows": rows,
        "ranked_candidates": ranked,
    }
    write_json(args.summary_json, payload)
    write_csv(args.summary_csv, rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

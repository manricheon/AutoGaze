from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.hlvid_batch_benchmark import discover_dataset_layout
from repro.hlvid_batch_benchmark import main as run_nvila_hlvid_main
from repro.plugin_hlvid_benchmark import parse_model_overrides, run_plugin_hlvid_benchmark


PLUGIN_SUITE_MODES = {
    "qwen": [
        "qwen_full_vit",
        "qwen_chunked_vit",
        "qwen_chunked_vit_autogaze_sparse",
    ],
}


def build_plugin_router_parser(*, add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route HLVid wrapper calls to plugin benchmarks such as Qwen.",
        add_help=add_help,
    )
    parser.add_argument("--dataset-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--video-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--plugin-suite", choices=["qwen", "custom"])
    parser.add_argument("--plugin-modes")
    parser.add_argument(
        "--plugin-model",
        action="append",
        help="Plugin model override as adapter=path, e.g. qwen3-vl=weight/Qwen3-VL",
    )
    parser.add_argument("--plugin-external-mllm-command", default="vila-infer")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Compatibility flag. Plugin HLVid always records per-row failures and continues where possible.",
    )
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
    return parser


def _split_modes(value: str | None) -> list[str]:
    return [mode.strip() for mode in (value or "").split(",") if mode.strip()]


def _plugin_modes(args: argparse.Namespace) -> list[str]:
    explicit = _split_modes(args.plugin_modes)
    if explicit:
        return explicit
    if args.plugin_suite in PLUGIN_SUITE_MODES:
        return list(PLUGIN_SUITE_MODES[args.plugin_suite])
    raise SystemExit("--plugin-modes is required when --plugin-suite custom is used.")


def _resolve_plugin_layout(args: argparse.Namespace) -> tuple[str, str]:
    if args.manifest and args.video_root:
        return args.manifest, args.video_root
    if not args.dataset_dir:
        raise SystemExit("--dataset-dir is required unless both --manifest and --video-root are provided.")
    layout = discover_dataset_layout(Path(args.dataset_dir))
    return str(layout["manifest"]), str(layout["video_root"])


def _is_plugin_route(argv: list[str]) -> bool:
    return "--plugin-suite" in argv or "--plugin-modes" in argv or "--plugin-model" in argv


def run_plugin_route(argv: list[str]) -> dict:
    parser = build_plugin_router_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            "unsupported option(s) for plugin HLVid route: "
            + " ".join(unknown)
            + ". Use the NVILA route without --plugin-suite, or add the option to the plugin wrapper."
        )
    manifest, video_root = _resolve_plugin_layout(args)
    output_dir = args.output_dir or f"outputs/autogaze_repro/plugin_hlvid_{args.plugin_suite or 'custom'}"
    qwen_video_nframes = args.qwen_video_nframes
    if args.plugin_suite == "qwen" and qwen_video_nframes is None:
        qwen_video_nframes = args.num_video_frames
    qwen_vit_max_spatial_chunks = args.qwen_vit_max_spatial_chunks
    if args.plugin_suite == "qwen" and qwen_vit_max_spatial_chunks is None:
        qwen_vit_max_spatial_chunks = args.max_tiles_video
    return run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=_plugin_modes(args),
        models=parse_model_overrides(args.plugin_model),
        external_mllm_command=args.plugin_external_mllm_command,
        limit=args.limit,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        max_new_tokens=args.max_new_tokens,
        qwen_video_nframes=qwen_video_nframes,
        qwen_video_fps=args.qwen_video_fps,
        qwen_video_max_pixels=args.qwen_video_max_pixels,
        qwen_video_min_pixels=args.qwen_video_min_pixels,
        qwen_vit_chunk_frames=args.qwen_vit_chunk_frames,
        qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks,
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


def main(argv: list[str] | None = None) -> None:
    original_argv = sys.argv
    routed_argv = list(sys.argv[1:] if argv is None else argv)
    if _is_plugin_route(routed_argv):
        if "--help" in routed_argv or "-h" in routed_argv:
            build_plugin_router_parser(add_help=True).parse_args(routed_argv)
            return
        payload = run_plugin_route(routed_argv)
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        return
    if argv is None:
        run_nvila_hlvid_main()
        return
    try:
        sys.argv = [original_argv[0], *routed_argv]
        run_nvila_hlvid_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

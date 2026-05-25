import json
import os
from pathlib import Path

from repro.plugin_hlvid_benchmark import (
    _flatten_key_metrics,
    _prediction_status,
    _summarize_by_mode,
    build_markdown_report,
    build_mode_runner_args,
    resolve_hlvid_video_path,
    run_plugin_hlvid_benchmark,
)


def test_resolve_hlvid_video_path_falls_back_to_basename_when_video_root_is_flat(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    flat_video = video_root / "clip_001.mp4"
    flat_video.write_text("fake")

    resolved = resolve_hlvid_video_path(video_root, "nested/path/clip_001.mp4")

    assert resolved == flat_video


def test_build_mode_runner_args_for_nvila_video_off():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="nvila-video-off",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={
            "nvila-video": "weight/NVILA-8B-Video",
            "longvila": "weight/LongVILA",
            "internvl3": "weight/InternVL3",
            "qwen3-vl": "weight/Qwen3-VL",
        },
        external_mllm_command="/opt/vila/bin/vila-infer",
        num_video_frames=256,
        max_tiles_video=8,
        max_new_tokens=16,
    )

    assert args[:8] == [
        "--mode",
        "single",
        "--model-family",
        "nvila-video-plugin",
        "--model-path",
        "weight/NVILA-8B-Video",
        "--token-selector-adapter",
        "keep-all",
    ]
    assert "--external-mllm-command" in args
    assert "/opt/vila/bin/vila-infer" in args
    assert "Question? A. one B. two C. three D. four" in args


def test_build_mode_runner_args_for_qwen3_pixelprune_pre_vit():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-pixelprune-pre-vit",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert "--model-family" in args
    assert args[args.index("--model-family") + 1] == "qwen3-vl"
    assert args[args.index("--autogaze-integration-level") + 1] == "pre_encoder_sparse"
    assert args[args.index("--pre-encoder-prune-adapter") + 1] == "pixelprune"
    assert args[args.index("--token-selector-adapter") + 1] == "keep-all"


def test_build_mode_runner_args_for_qwen3_passes_runner_video_resize():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-off",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
        video_resize_shortest_edge=448,
    )

    assert args[args.index("--video-resize-shortest-edge") + 1] == "448"


def test_build_mode_runner_args_for_qwen3_passes_thumbnail_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-off",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        num_video_frames_thumbnail=16,
        max_tiles_video=1,
        max_new_tokens=8,
        qwen_thumbnail_mode="append-video",
    )

    assert args[args.index("--num-video-frames-thumbnail") + 1] == "16"
    assert args[args.index("--qwen-thumbnail-mode") + 1] == "append-video"


def test_build_mode_runner_args_for_priority_autogaze_poc_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {
            "nvila-video": "weight/NVILA-8B-Video",
            "longvila": "weight/LongVILA",
            "qwen3-vl": "weight/Qwen3-VL",
        },
        "external_mllm_command": "vila-infer",
        "num_video_frames": 128,
        "max_tiles_video": 4,
        "max_new_tokens": 8,
    }

    nvila_args = build_mode_runner_args(mode="nvila-video-autogaze-probe", **common)
    longvila_args = build_mode_runner_args(mode="longvila-autogaze-probe", **common)
    qwen_args = build_mode_runner_args(mode="qwen3-vl-autogaze-poc", **common)

    assert nvila_args[nvila_args.index("--model-family") + 1] == "nvila-video-plugin"
    assert nvila_args[nvila_args.index("--token-selector-adapter") + 1] == "autogaze"
    assert nvila_args[nvila_args.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert longvila_args[longvila_args.index("--model-family") + 1] == "longvila"
    assert longvila_args[longvila_args.index("--token-selector-adapter") + 1] == "autogaze"
    assert qwen_args[qwen_args.index("--model-family") + 1] == "qwen3-vl"
    assert qwen_args[qwen_args.index("--gazing-ratio") + 1] == "0.1"


def test_build_mode_runner_args_for_qwen3_autogaze_prune_generate_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-prune-generate",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert args[args.index("--model-family") + 1] == "qwen3-vl"
    assert args[args.index("--token-selector-adapter") + 1] == "autogaze"
    assert args[args.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert "--enable-qwen-prune-generate" in args


def test_build_mode_runner_args_for_qwen3_direct_autogaze_prune_generate_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-direct-prune-generate",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
    )

    assert "--enable-qwen-prune-generate" in args
    assert "--run-autogaze-selector" in args
    assert "--autogaze-generate-only" in args


def test_build_mode_runner_args_for_qwen3_direct_autogaze_pre_vit_sparse_mode():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }

    args = build_mode_runner_args(
        mode="qwen3-vl-autogaze-direct-pre-vit-sparse",
        row=row,
        video_path=Path("/data/clip.mp4"),
        output_json=Path("/tmp/run.json"),
        models={"qwen3-vl": "weight/Qwen3-VL"},
        external_mllm_command="vila-infer",
        num_video_frames=128,
        max_tiles_video=1,
        max_new_tokens=8,
        qwen_video_max_pixels=200704,
    )

    assert args[args.index("--autogaze-integration-level") + 1] == "pre_encoder_sparse"
    assert args[args.index("--pre-encoder-prune-adapter") + 1] == "autogaze-sparse"
    assert "--enable-qwen-prune-generate" in args
    assert "--run-autogaze-selector" in args
    assert args[args.index("--qwen-video-max-pixels") + 1] == "200704"


def test_build_mode_runner_args_for_qwen_vit_comparison_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {"qwen3-vl": "weight/Qwen3-VL"},
        "external_mllm_command": "vila-infer",
        "num_video_frames": 128,
        "max_tiles_video": 1,
        "max_new_tokens": 8,
        "qwen_vit_chunk_frames": 16,
        "qwen_vit_max_spatial_chunks": 4,
    }

    full_args = build_mode_runner_args(mode="qwen_full_vit", **common)
    chunked_args = build_mode_runner_args(mode="qwen_chunked_vit", **common)
    sparse_args = build_mode_runner_args(mode="qwen_chunked_vit_autogaze_sparse", **common)
    tile_dense_args = build_mode_runner_args(mode="qwen_tile_packed_vit", **common)
    tile_sparse_args = build_mode_runner_args(mode="qwen_tile_packed_vit_autogaze_sparse", **common)

    assert full_args[full_args.index("--qwen-vit-mode") + 1] == "qwen_full_vit"
    assert full_args[full_args.index("--token-selector-adapter") + 1] == "keep-all"
    assert chunked_args[chunked_args.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit"
    assert chunked_args[chunked_args.index("--qwen-vit-chunk-frames") + 1] == "16"
    assert chunked_args[chunked_args.index("--qwen-vit-max-spatial-chunks") + 1] == "4"
    assert sparse_args[sparse_args.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit_autogaze_sparse"
    assert sparse_args[sparse_args.index("--pre-encoder-prune-adapter") + 1] == "autogaze-sparse"
    assert "--run-autogaze-selector" in sparse_args
    assert "--enable-qwen-prune-generate" in sparse_args
    assert tile_dense_args[tile_dense_args.index("--qwen-vit-mode") + 1] == "qwen_tile_packed_vit"
    assert tile_dense_args[tile_dense_args.index("--token-selector-adapter") + 1] == "keep-all"
    assert tile_sparse_args[tile_sparse_args.index("--qwen-vit-mode") + 1] == "qwen_tile_packed_vit_autogaze_sparse"
    assert tile_sparse_args[tile_sparse_args.index("--pre-encoder-prune-adapter") + 1] == "autogaze-sparse"
    assert "--run-autogaze-selector" in tile_sparse_args


def test_build_mode_runner_args_for_qwen25_and_qwen3_named_vit_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {
            "qwen2.5-vl": "weight/Qwen2.5-VL",
            "qwen3-vl": "weight/Qwen3-VL",
        },
        "external_mllm_command": "vila-infer",
        "num_video_frames": 16,
        "max_tiles_video": 4,
        "max_new_tokens": 8,
        "qwen_vit_chunk_frames": 16,
        "qwen_vit_max_spatial_chunks": 4,
    }

    qwen25_sparse = build_mode_runner_args(mode="qwen2.5_chunked_vit_autogaze_sparse", **common)
    qwen3_sparse = build_mode_runner_args(mode="qwen3_chunked_vit_autogaze_sparse", **common)
    qwen25_full = build_mode_runner_args(mode="qwen2.5_full_vit", **common)

    assert qwen25_sparse[qwen25_sparse.index("--model-family") + 1] == "qwen2.5-vl"
    assert qwen25_sparse[qwen25_sparse.index("--model-path") + 1] == "weight/Qwen2.5-VL"
    assert qwen25_sparse[qwen25_sparse.index("--vision-encoder-adapter") + 1] == "qwen2.5-vl-vision"
    assert qwen25_sparse[qwen25_sparse.index("--mllm-adapter") + 1] == "qwen2.5-vl"
    assert qwen25_sparse[qwen25_sparse.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit_autogaze_sparse"
    assert "--run-autogaze-selector" in qwen25_sparse
    assert "--enable-qwen-prune-generate" in qwen25_sparse
    assert qwen3_sparse[qwen3_sparse.index("--model-family") + 1] == "qwen3-vl"
    assert qwen3_sparse[qwen3_sparse.index("--qwen-vit-mode") + 1] == "qwen_chunked_vit_autogaze_sparse"
    assert qwen25_full[qwen25_full.index("--qwen-vit-mode") + 1] == "qwen_full_vit"
    assert qwen25_full[qwen25_full.index("--token-selector-adapter") + 1] == "keep-all"


def test_build_mode_runner_args_for_llava_and_internvl_autogaze_probe_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {
            "llava-onevision": "weight/LLaVA-OneVision",
            "internvl3": "weight/InternVL3",
        },
        "external_mllm_command": "vila-infer",
        "num_video_frames": 64,
        "num_video_frames_thumbnail": 8,
        "max_tiles_video": 4,
        "max_new_tokens": 8,
        "qwen_thumbnail_mode": "append-video",
        "video_resize_longest_edge": 448,
    }

    llava_off = build_mode_runner_args(mode="llava-onevision-off", **common)
    llava_probe = build_mode_runner_args(mode="llava-onevision-autogaze-probe", **common)
    llava_materialized = build_mode_runner_args(mode="llava-onevision-autogaze-materialized", **common)
    internvl_off = build_mode_runner_args(mode="internvl3-off", **common)
    internvl_probe = build_mode_runner_args(mode="internvl3-autogaze-probe", **common)

    assert llava_off[llava_off.index("--model-family") + 1] == "llava-onevision"
    assert llava_off[llava_off.index("--token-selector-adapter") + 1] == "keep-all"
    assert llava_off[llava_off.index("--autogaze-integration-level") + 1] == "none"
    assert llava_probe[llava_probe.index("--model-family") + 1] == "llava-onevision"
    assert llava_probe[llava_probe.index("--token-selector-adapter") + 1] == "autogaze"
    assert llava_probe[llava_probe.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert internvl_probe[internvl_probe.index("--model-family") + 1] == "internvl3"
    assert internvl_probe[internvl_probe.index("--token-selector-adapter") + 1] == "autogaze"
    assert internvl_probe[internvl_probe.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"
    assert internvl_off[internvl_off.index("--num-video-frames-thumbnail") + 1] == "8"
    assert internvl_off[internvl_off.index("--video-resize-longest-edge") + 1] == "448"
    assert llava_probe[llava_probe.index("--num-video-frames-thumbnail") + 1] == "8"
    assert llava_probe[llava_probe.index("--video-resize-longest-edge") + 1] == "448"
    assert llava_materialized[llava_materialized.index("--autogaze-integration-level") + 1] == "input_materialization_diagnostic"
    assert "--run-autogaze-selector" in llava_materialized
    assert "--enable-visual-prune-generate" in llava_materialized


def test_build_mode_runner_args_for_executable_autogaze_expansion_modes():
    row = {
        "video_path": "clip.mp4",
        "question": "Question? A. one B. two C. three D. four",
        "answer": "A",
    }
    common = {
        "row": row,
        "video_path": Path("/data/clip.mp4"),
        "output_json": Path("/tmp/run.json"),
        "models": {
            "nvila-video": "weight/NVILA-8B-Video",
            "longvila": "weight/LongVILA",
            "llava-onevision": "weight/LLaVA-OneVision",
            "internvl3": "weight/InternVL3",
        },
        "external_mllm_command": "vila-infer",
        "num_video_frames": 64,
        "max_tiles_video": 4,
        "max_new_tokens": 8,
    }

    for mode in [
        "nvila-video-autogaze-actual",
        "longvila-autogaze-actual",
        "llava-onevision-autogaze-actual",
        "internvl3-autogaze-sidecar-generate",
    ]:
        args = build_mode_runner_args(mode=mode, **common)
        assert "--run-autogaze-selector" in args
        assert "--autogaze-generate-only" in args
        assert "--enable-visual-prune-generate" in args
        assert args[args.index("--token-selector-adapter") + 1] == "autogaze"
        assert args[args.index("--autogaze-integration-level") + 1] == "post_encoder_token_prune"


def test_flatten_key_metrics_includes_qwen_vit_comparison_fields():
    flattened = _flatten_key_metrics(
        {
            "latency_ms": {"total": 100.0, "generate": 40.0, "qwen_vit_prepare": 30.0},
            "memory_bytes": {"peak_cuda_allocated": 10, "peak_cuda_reserved": 20},
            "tokens": {
                "visual_tokens_before_prune": 1000,
                "visual_tokens_after_prune": 100,
                "visual_token_reduction_ratio": 10.0,
                "llm_context_tokens": 120,
            },
            "qwen_vit": {
                "mode": "qwen_chunked_vit_autogaze_sparse",
                "raw_patch_tokens_before_vit": 4000,
                "chunk_count": 4,
                "executed_chunk_count": 3,
                "spatial_chunking": {"tile_grid": {"tiles": 4}},
            },
        }
    )

    assert flattened["qwen_vit_prepare_ms"] == 30.0
    assert flattened["qwen_vit_mode"] == "qwen_chunked_vit_autogaze_sparse"
    assert flattened["visual_tokens_before_prune"] == 1000
    assert flattened["visual_tokens_after_prune"] == 100
    assert flattened["visual_token_reduction_ratio"] == 10.0
    assert flattened["qwen_vit_raw_patch_tokens_before_vit"] == 4000
    assert flattened["qwen_vit_executed_chunk_count"] == 3
    assert flattened["qwen_vit_spatial_tiles"] == 4
    assert flattened["autogaze_attachment_mode"] == "actual_pre_encoder_sparse"
    assert flattened["visual_pruning_applied"] is True
    assert flattened["vision_encoder_latency_reduced"] is True
    assert flattened["mllm_context_reduced"] is True


def test_flatten_key_metrics_includes_common_selector_vit_memory_and_failure_fields():
    flattened = _flatten_key_metrics(
        {
            "latency_ms": {
                "total": 120.0,
                "generate": 45.0,
                "qwen_vit_prepare": 30.0,
                "selector": 12.0,
                "vision_encoder": 28.0,
            },
            "memory_bytes": {"peak_cuda_allocated": 1024, "peak_cuda_reserved": 2048},
            "tokens": {
                "visual_tokens_before_prune": 1000,
                "visual_tokens_after_prune": 125,
                "visual_token_reduction_ratio": 8.0,
                "llm_context_tokens": 256,
            },
            "sparse_selection_plan": {
                "token_accounting": {
                    "raw_patch_tokens": 4000,
                    "selected_patch_tokens": 400,
                    "reduction_ratio": 10.0,
                },
                "encoder_mapping": {"encoder_patch_indices": [1, 2, 3]},
                "mllm_mapping": {"visual_feature_indices": [1, 3]},
            },
            "autogaze_attachment": {
                "mode": "dense_generation_with_autogaze_sidecar",
                "visual_pruning_applied": False,
                "vision_encoder_latency_reduced": False,
                "mllm_context_reduced": False,
            },
            "failure": {"kind": "oom", "stage": "vision_encoder"},
        }
    )

    assert flattened["selector_ms"] == 12.0
    assert flattened["vision_encoder_ms"] == 28.0
    assert flattened["raw_patch_tokens"] == 4000
    assert flattened["selected_patch_tokens"] == 400
    assert flattened["patch_token_reduction_ratio"] == 10.0
    assert flattened["encoder_input_tokens"] == 3
    assert flattened["llm_visual_tokens"] == 125
    assert flattened["failure_stage"] == "vision_encoder"
    assert flattened["autogaze_attachment_mode"] == "dense_generation_with_autogaze_sidecar"
    assert flattened["visual_pruning_applied"] is False
    assert flattened["vision_encoder_latency_reduced"] is False
    assert flattened["mllm_context_reduced"] is False


def test_plugin_hlvid_summary_surfaces_processing_budget_by_mode():
    summary = _summarize_by_mode(
        [
            {
                "mode": "qwen_chunked_vit_autogaze_sparse",
                "question_id": "q1",
                "video_path": "clip.mp4",
                "question": "Question? A. one B. two C. three D. four",
                "answer": "B",
                "expected_answer": "B",
                "raw_output": "B",
                "parsed_answer": "B",
                "correct": True,
                "status": "ok",
                "runner_status": "executed",
                "metrics": {
                    "processing_budget_summary": {
                        "video": {
                            "source_resolution": "3840x2160",
                            "processor_input_resolution": "1280x720",
                            "requested_video_frames": 128,
                        },
                        "thumbnail": {"enabled": True, "effective_frames": 16},
                        "patch_budget_before_vit": {
                            "actual_raw_patch_tokens_before_vit": 1000,
                            "estimated_visual_tokens_after_prune": 100,
                            "estimated_visual_token_reduction_ratio": 10.0,
                        },
                    }
                },
            }
        ]
    )

    mode_summary = summary["modes"]["qwen_chunked_vit_autogaze_sparse"]
    budget = mode_summary["processing_budget_summary"]
    assert budget["mode_median"]["video.source_resolution"] == "3840x2160"
    assert budget["mode_median"]["patch_budget_before_vit.actual_raw_patch_tokens_before_vit"] == 1000
    assert budget["mode_median"]["patch_budget_before_vit.estimated_visual_tokens_after_prune"] == 100

    markdown = build_markdown_report(summary)
    assert "## Processing Budget By Mode" in markdown
    assert "qwen_chunked_vit_autogaze_sparse" in markdown
    assert "1,000" in markdown
    assert "100" in markdown


def test_plugin_hlvid_summary_separates_sidecar_and_actual_pruning_claims():
    summary = _summarize_by_mode(
        [
            {
                "mode": "longvila-autogaze-actual",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "runner_status": "executed_dense_with_autogaze_sidecar",
                "visual_pruning_applied": False,
                "vision_encoder_latency_reduced": False,
                "mllm_context_reduced": False,
            },
            {
                "mode": "qwen3_chunked_vit_autogaze_sparse",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "runner_status": "executed",
                "visual_pruning_applied": True,
                "vision_encoder_latency_reduced": True,
                "mllm_context_reduced": True,
            },
            {
                "mode": "llava-onevision-autogaze-actual",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "runner_status": "executed",
                "visual_pruning_applied": True,
                "vision_encoder_latency_reduced": False,
                "mllm_context_reduced": True,
            },
        ]
    )

    longvila = summary["modes"]["longvila-autogaze-actual"]["integration_summary"]
    qwen = summary["modes"]["qwen3_chunked_vit_autogaze_sparse"]["integration_summary"]
    llava = summary["modes"]["llava-onevision-autogaze-actual"]["integration_summary"]

    assert longvila["execution_claim"] == "dense_generation_with_autogaze_sidecar"
    assert longvila["vision_encoder_latency_reduction_claim"] == "no"
    assert longvila["mllm_context_reduction_claim"] == "no"
    assert qwen["execution_claim"] == "actual_pre_encoder_sparse"
    assert qwen["vision_encoder_latency_reduction_claim"] == "yes"
    assert qwen["mllm_context_reduction_claim"] == "yes"
    assert llava["execution_claim"] == "actual_post_encoder_token_prune"
    assert llava["vision_encoder_latency_reduction_claim"] == "no"
    assert llava["mllm_context_reduction_claim"] == "yes"

    markdown = build_markdown_report(summary)
    assert "## AutoGaze Integration Claims" in markdown
    assert "| longvila-autogaze-actual | post_encoder_token_prune | dense_generation_with_autogaze_sidecar | no | no | no |" in markdown
    assert "| qwen3_chunked_vit_autogaze_sparse | pre_encoder_sparse | actual_pre_encoder_sparse | yes | yes | yes |" in markdown
    assert "| llava-onevision-autogaze-actual | post_encoder_token_prune | actual_post_encoder_token_prune | yes | no | yes |" in markdown


def test_plugin_hlvid_summary_reports_materialized_sparse_video_claim_from_rows():
    summary = _summarize_by_mode(
        [
            {
                "mode": "longvila-autogaze-actual",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "runner_status": "executed_materialized_sparse_video",
                "autogaze_attachment_mode": "materialized_sparse_video",
                "visual_pruning_applied": False,
                "vision_encoder_latency_reduced": False,
                "mllm_context_reduced": False,
            }
        ]
    )

    integration = summary["modes"]["longvila-autogaze-actual"]["integration_summary"]
    assert integration["execution_claim"] == "materialized_sparse_video"
    assert integration["actual_pruning_applied_claim"] == "no"
    assert integration["vision_encoder_latency_reduction_claim"] == "no"
    assert integration["mllm_context_reduction_claim"] == "no"

    markdown = build_markdown_report(summary)
    assert "| longvila-autogaze-actual | post_encoder_token_prune | materialized_sparse_video | no | no | no |" in markdown


def test_prediction_status_treats_materialized_sparse_video_as_ok():
    assert _prediction_status("executed_materialized_sparse_video") == "ok"


def test_plugin_hlvid_summary_adds_pairwise_autogaze_comparisons():
    summary = _summarize_by_mode(
        [
            {
                "mode": "qwen3_full_vit",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "status": "ok",
                "runner_status": "executed",
                "total_ms": 1000.0,
                "peak_memory_bytes": 2000,
                "visual_tokens_before_prune": 1000,
                "visual_tokens_after_prune": 1000,
                "llm_visual_tokens": 1000,
            },
            {
                "mode": "qwen3_chunked_vit_autogaze_sparse",
                "question_id": "q1",
                "answer": "A",
                "raw_output": "A",
                "status": "ok",
                "runner_status": "executed",
                "total_ms": 400.0,
                "peak_memory_bytes": 800,
                "raw_patch_tokens": 1000,
                "selected_patch_tokens": 100,
                "encoder_input_tokens": 100,
                "llm_visual_tokens": 100,
                "visual_tokens_before_prune": 1000,
                "visual_tokens_after_prune": 100,
                "visual_token_reduction_ratio": 10.0,
                "visual_pruning_applied": True,
                "vision_encoder_latency_reduced": True,
                "mllm_context_reduced": True,
            },
        ]
    )

    comparisons = summary["pairwise_comparisons"]
    pair = next(item for item in comparisons if item["candidate_mode"] == "qwen3_chunked_vit_autogaze_sparse")

    assert pair["baseline_mode"] == "qwen3_full_vit"
    assert pair["integration_level"] == "pre_encoder_sparse"
    assert pair["latency_speedup"] == 2.5
    assert pair["patch_or_visual_token_reduction_ratio"] == 10.0
    assert pair["llm_visual_token_reduction_ratio"] == 10.0
    assert pair["memory_reduction_ratio"] == 2.5

    markdown = build_markdown_report(summary)
    assert "## Pairwise Plugin Comparisons" in markdown
    assert "qwen3_full_vit -> qwen3_chunked_vit_autogaze_sparse" in markdown
    assert "10" in markdown


def test_run_plugin_hlvid_benchmark_writes_predictions_summary_and_markdown(tmp_path):
    manifest = tmp_path / "manifest.json"
    video_root = tmp_path / "videos"
    output_dir = tmp_path / "out"
    video_root.mkdir()
    (video_root / "clip_001.mp4").write_text("fake video")
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "toy",
                    "video_path": "nested/clip_001.mp4",
                    "question": "Question? A. one B. two C. three D. four",
                    "answer": "B",
                }
            ]
        )
    )
    fake_runner = tmp_path / "fake-vila-infer"
    fake_runner.write_text("#!/bin/sh\necho 'Assistant: B'\n")
    os.chmod(fake_runner, 0o755)

    payload = run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=["nvila-video-off"],
        models={"nvila-video": "weight/NVILA-8B-Video"},
        external_mllm_command=str(fake_runner),
        limit=3,
        num_video_frames=8,
        max_tiles_video=1,
        max_new_tokens=4,
    )

    assert payload["summary"]["modes"]["nvila-video-off"]["correct"] == 1
    assert payload["summary"]["modes"]["nvila-video-off"]["accuracy_total"] == 1.0
    assert (output_dir / "plugin_hlvid_predictions.jsonl").is_file()
    assert (output_dir / "plugin_hlvid_summary.json").is_file()
    report = (output_dir / "plugin_hlvid_report.md").read_text()
    assert "nvila-video-off" in report
    assert "accuracy_total" in report


def test_run_plugin_hlvid_benchmark_records_oom_row_and_continues(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    video_root = tmp_path / "videos"
    output_dir = tmp_path / "out"
    video_root.mkdir()
    (video_root / "clip_001.mp4").write_text("fake video")
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "toy",
                    "video_path": "clip_001.mp4",
                    "question": "Question? A. one B. two C. three D. four",
                    "answer": "B",
                }
            ]
        )
    )

    def fake_run_single(args):
        raise RuntimeError("CUDA out of memory in SDPA attention")

    monkeypatch.setattr("repro.plugin_hlvid_benchmark.run_single", fake_run_single)

    payload = run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=["qwen_full_vit"],
        models={"qwen3-vl": "weight/Qwen3-VL"},
        limit=1,
        num_video_frames=8,
        max_tiles_video=1,
        max_new_tokens=4,
    )

    prediction = payload["predictions"][0]
    assert prediction["status"] == "oom"
    assert prediction["runner_status"] == "oom"
    assert prediction["failure"]["stage"] == "llm_prefill_or_generate"
    assert payload["summary"]["modes"]["qwen_full_vit"]["status_counts"] == {"oom": 1}
    assert (output_dir / "runs" / "qwen_full_vit" / "00000.json").is_file()


def test_plugin_hlvid_summary_reports_probe_and_poc_statuses(tmp_path):
    manifest = tmp_path / "manifest.json"
    video_root = tmp_path / "videos"
    output_dir = tmp_path / "out"
    nvila_model = tmp_path / "NVILA-8B-Video"
    longvila_model = tmp_path / "LongVILA"
    video_root.mkdir()
    nvila_model.mkdir()
    longvila_model.mkdir()
    (nvila_model / "config.json").write_text('{"model_type": "llava", "vision_tower": "siglip"}')
    (longvila_model / "config.json").write_text('{"model_type": "llava", "vision_tower": "siglip"}')
    (video_root / "clip_001.mp4").write_text("fake video")
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "category": "toy",
                    "video_path": "clip_001.mp4",
                    "question": "Question? A. one B. two C. three D. four",
                    "answer": "B",
                }
            ]
        )
    )

    payload = run_plugin_hlvid_benchmark(
        manifest=manifest,
        video_root=video_root,
        output_dir=output_dir,
        modes=["nvila-video-autogaze-probe", "longvila-autogaze-probe", "qwen3-vl-autogaze-poc"],
        models={
            "nvila-video": str(nvila_model),
            "longvila": str(longvila_model),
            "qwen3-vl": "weight/Qwen3-VL",
        },
        limit=1,
        num_video_frames=8,
        max_tiles_video=1,
        max_new_tokens=4,
    )

    summary = payload["summary"]["modes"]
    assert summary["nvila-video-autogaze-probe"]["status_counts"] == {"probe_collected": 1}
    assert summary["longvila-autogaze-probe"]["status_counts"] == {"probe_collected": 1}
    assert summary["qwen3-vl-autogaze-poc"]["status_counts"] == {"poc_ready": 1}
    assert summary["nvila-video-autogaze-probe"]["next_action"] == "instrument_vila_remote_code_feature_packing"
    assert summary["longvila-autogaze-probe"]["next_action"] == "instrument_vila_remote_code_feature_packing"
    assert summary["qwen3-vl-autogaze-poc"]["next_action"] == "implement_qwen_visual_feature_prune_generate"
    report = (output_dir / "plugin_hlvid_report.md").read_text()
    assert "status_counts" in report
    assert "instrument_vila_remote_code_feature_packing" in report
    assert "implement_qwen_visual_feature_prune_generate" in report

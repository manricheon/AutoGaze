from pathlib import Path
import json

from repro.flexible_runner import build_inspect_payload, parse_args, run_inspect, run_single
from repro.plugins.mllm_adapters import MllmRunResult, QwenGridMllmAdapter


def test_flexible_runner_builds_longvila_autogaze_plugin_spec():
    args = parse_args(
        [
            "--model-family",
            "longvila",
            "--model-path",
            "weight/longvila",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "longvila-siglip",
            "--mllm-adapter",
            "longvila",
            "--autogaze-integration-level",
            "planned_plugin",
            "--output-json",
            "outputs/autogaze_repro/flexible_longvila_inspect.json",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["runner"] == "flexible_runner"
    assert payload["mode"] == "inspect"
    assert payload["experiment_spec"]["model_family"] == "longvila"
    assert payload["experiment_spec"]["token_selector_kind"] == "autogaze"
    assert payload["experiment_spec"]["integration_level"] == "planned_plugin"
    assert payload["experiment_spec"]["uses_autogaze"] is True
    assert payload["adapter_plan"]["token_selector"]["adapter"] == "autogaze"
    assert payload["adapter_plan"]["vision_encoder"]["adapter"] == "longvila-siglip"
    assert payload["adapter_plan"]["mllm"]["adapter"] == "longvila"
    assert payload["adapter_plan"]["mllm"]["status"] == {
        "value": "external_cli_ready",
        "reason": "official VILA CLI adapter",
    }
    assert payload["implementation_status"] == "inspect_only"


def test_flexible_runner_supports_qwen_family_without_touching_nvila_runner():
    args = parse_args(
        [
            "--model-family",
            "qwen2-vl",
            "--model-path",
            "weight/Qwen2-VL",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen2-vl-vision",
            "--mllm-adapter",
            "qwen2-vl",
            "--autogaze-integration-level",
            "none",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["experiment_spec"]["model_family"] == "qwen2-vl"
    assert payload["adapter_plan"]["token_selector"]["adapter"] == "keep-all"
    assert payload["adapter_plan"]["mllm"]["adapter"] == "qwen2-vl"
    assert payload["paper_baseline_semantics"] == "not_a_paper_baseline"


def test_flexible_runner_records_nvila_video_external_cli_status():
    args = parse_args(
        [
            "--model-family",
            "nvila-video-plugin",
            "--model-path",
            "weight/NVILA-8B-Video",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--mllm-adapter",
            "nvila-video",
            "--autogaze-integration-level",
            "none",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["adapter_plan"]["mllm"]["status"] == {
        "value": "external_cli_ready",
        "reason": "official VILA CLI adapter",
    }
    assert payload["mllm_status_matrix"]["nvila-video-plugin"]["native_off_status"] == "external_cli_ready"


def test_flexible_runner_single_nvila_video_missing_cli_reports_dependency_failure(tmp_path):
    output_json = tmp_path / "nvila_video_missing_cli.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "nvila-video-plugin",
            "--model-path",
            "weight/NVILA-8B-Video",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--mllm-adapter",
            "nvila-video",
            "--autogaze-integration-level",
            "none",
            "--external-mllm-command",
            "definitely_missing_vila_infer_for_test",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["implementation_status"] == "failed_missing_dependency"
    assert payload["generation"]["status"] == "failed_missing_dependency"
    assert payload["generation"]["metrics"]["metric_status"]["reason"] == "vila-infer command was not found"
    assert json.loads(output_json.read_text())["implementation_status"] == "failed_missing_dependency"


def test_flexible_runner_single_longvila_missing_cli_reports_dependency_failure(tmp_path):
    output_json = tmp_path / "longvila_missing_cli.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "longvila",
            "--model-path",
            "weight/LongVILA",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "longvila-siglip",
            "--mllm-adapter",
            "longvila",
            "--autogaze-integration-level",
            "none",
            "--external-mllm-command",
            "definitely_missing_vila_infer_for_test",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["implementation_status"] == "failed_missing_dependency"
    assert payload["generation"]["adapter"] == "longvila"
    assert payload["generation"]["metrics"]["external_cli"]["command"][:3] == [
        "definitely_missing_vila_infer_for_test",
        "--model-path",
        "weight/LongVILA",
    ]
    assert json.loads(output_json.read_text())["generation"]["adapter"] == "longvila"


def test_flexible_runner_supports_qwen25_vl_status_matrix():
    args = parse_args(
        [
            "--model-family",
            "qwen2.5-vl",
            "--model-path",
            "weight/Qwen2.5-VL-7B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen2.5-vl-vision",
            "--mllm-adapter",
            "qwen2.5-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["model_capabilities"]["family_group"] == "qwen_grid_vl"
    assert payload["model_capabilities"]["video_forward_fields"] == ["pixel_values_videos", "video_grid_thw"]
    assert payload["mllm_status_matrix"]["qwen2.5-vl"]["recommended_first_on_mode"] == "post_encoder_token_prune"
    assert payload["four_step_execution_plan"]["post_encoder_token_prune"]["status"] == "candidate_next"


def test_flexible_runner_supports_qwen3_vl_family_and_four_step_plan():
    args = parse_args(
        [
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--pre-encoder-prune-adapter",
            "pixelprune",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["experiment_spec"]["model_family"] == "qwen3-vl"
    assert payload["model_capabilities"]["family_group"] == "qwen3_vl"
    assert payload["model_capabilities"]["video_forward_fields"] == [
        "pixel_values_videos",
        "video_grid_thw",
        "mm_token_type_ids",
    ]
    assert list(payload["four_step_execution_plan"]) == [
        "native_off_baseline",
        "autogaze_standalone_selector",
        "post_encoder_token_prune",
        "pre_encoder_sparse",
    ]
    assert payload["four_step_execution_plan"]["post_encoder_token_prune"]["status"] == "candidate_next"
    assert payload["four_step_execution_plan"]["pre_encoder_sparse"]["status"] == "pixelprune_reference_available"
    assert payload["adapter_plan"]["pre_encoder_prune"]["adapter"] == "pixelprune"
    assert payload["adapter_plan"]["pre_encoder_prune"]["status"]["value"] == "candidate"


def test_flexible_runner_pixelprune_pre_vit_fails_before_dense_qwen_when_hook_missing(monkeypatch, tmp_path):
    output_json = tmp_path / "qwen3_pixelprune_missing.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "pre_encoder_sparse",
            "--pre-encoder-prune-adapter",
            "pixelprune",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    monkeypatch.setattr(
        "repro.flexible_runner.apply_pixelprune_if_available",
        lambda config: {
            "applied": False,
            "reason": "pixelprune package is not installed",
            "model_key": config.model_key,
            "environment": config.apply_environment(),
        },
    )

    payload = run_single(args)

    assert payload["implementation_status"] == "failed_missing_dependency"
    assert payload["pre_encoder_prune_runtime"]["applied"] is False
    assert payload["generation"]["status"] == "failed_missing_dependency"
    assert payload["generation"]["metrics"]["metric_status"]["reason"] == "pixelprune package is not installed"


def test_flexible_runner_pixelprune_pre_vit_runs_qwen_after_hook_applied(monkeypatch, tmp_path):
    output_json = tmp_path / "qwen3_pixelprune_ok.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "pre_encoder_sparse",
            "--pre-encoder-prune-adapter",
            "pixelprune",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    monkeypatch.setattr(
        "repro.flexible_runner.apply_pixelprune_if_available",
        lambda config: {
            "applied": True,
            "reason": None,
            "model_key": config.model_key,
            "environment": config.apply_environment(),
        },
    )

    def fake_run(self, request):
        return MllmRunResult(
            text="ok",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={"latency_ms": {"total": 1.0}, "tokens": {}, "memory_bytes": {}},
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_generate", fake_run)

    payload = run_single(args)

    assert payload["implementation_status"] == "executed"
    assert payload["pre_encoder_prune_runtime"]["applied"] is True
    assert payload["generation"]["text"] == "ok"
    assert payload["generation"]["metrics"]["pre_encoder_prune"]["adapter"] == "pixelprune"


def test_flexible_runner_qwen_autogaze_post_encoder_returns_poc_plan(tmp_path):
    output_json = tmp_path / "qwen3_autogaze_poc.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--gazing-ratio",
            "0.1",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["implementation_status"] == "poc_ready"
    assert payload["generation"]["status"] == "poc_ready"
    assert payload["generation"]["metrics"]["sparse_selection_plan"]["selector_name"] == "autogaze"
    assert payload["generation"]["metrics"]["metric_status"]["value"] == "autogaze_qwen_poc_ready"


def test_flexible_runner_can_enable_qwen_autogaze_prune_generate(monkeypatch, tmp_path):
    output_json = tmp_path / "qwen3_autogaze_prune_generate.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--enable-qwen-prune-generate",
            "--gazing-ratio",
            "0.1",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    def fake_run(self, request):
        return MllmRunResult(
            text="qwen pruned",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={
                "latency_ms": {"total": 10.0},
                "tokens": {"visual_tokens_before_prune": 100, "visual_tokens_after_prune": 10},
                "memory_bytes": {},
                "qwen_prune_generate": {"enabled": True},
            },
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_autogaze_prune_generate", fake_run)

    payload = run_single(args)

    assert payload["implementation_status"] == "executed"
    assert payload["post_encoder_prune_runtime"]["status"] == "experimental_prune_generate_enabled"
    assert payload["generation"]["text"] == "qwen pruned"
    assert payload["generation"]["metrics"]["qwen_prune_generate"]["enabled"] is True


def test_flexible_runner_passes_sparse_selection_plan_json_to_qwen_prune_generate(monkeypatch, tmp_path):
    output_json = tmp_path / "qwen3_autogaze_plan.json"
    plan_json = tmp_path / "sparse_plan.json"
    plan_json.write_text('{"selector_name": "autogaze", "selected_patches": []}')
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--enable-qwen-prune-generate",
            "--sparse-selection-plan-json",
            str(plan_json),
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    def fake_run(self, request):
        assert request.sparse_selection_plan_path == str(plan_json)
        return MllmRunResult(
            text="qwen plan",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={"latency_ms": {"total": 1.0}, "tokens": {}, "memory_bytes": {}},
        )

    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_autogaze_prune_generate", fake_run)

    payload = run_single(args)

    assert payload["generation"]["text"] == "qwen plan"


def test_flexible_runner_runs_direct_autogaze_selector_before_qwen_prune_generate(monkeypatch, tmp_path):
    output_json = tmp_path / "qwen3_direct_autogaze.json"
    generated_plan_json = tmp_path / "generated_sparse_plan.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--enable-qwen-prune-generate",
            "--run-autogaze-selector",
            "--autogaze-selector-output-json",
            str(generated_plan_json),
            "--gazing-ratio",
            "0.1",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    def fake_selector(selector_args):
        assert selector_args.video == "inputs/example.mp4"
        generated_plan_json.write_text('{"selector_name": "autogaze-direct", "selected_patches": []}')
        return {
            "status": "executed",
            "sparse_selection_plan_json": str(generated_plan_json),
            "tokens": {
                "raw_patch_tokens": 100,
                "selected_patch_tokens": 9,
                "reduction_ratio": 100 / 9,
            },
        }

    def fake_run(self, request):
        assert request.sparse_selection_plan_path == str(generated_plan_json)
        return MllmRunResult(
            text="qwen direct autogaze",
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics={
                "latency_ms": {"total": 1.0},
                "tokens": {"visual_tokens_before_prune": 100, "visual_tokens_after_prune": 9},
                "memory_bytes": {},
            },
        )

    monkeypatch.setattr("repro.flexible_runner.run_direct_autogaze_selector", fake_selector)
    monkeypatch.setattr(QwenGridMllmAdapter, "_run_qwen_autogaze_prune_generate", fake_run)

    payload = run_single(args)

    assert payload["implementation_status"] == "executed"
    assert payload["direct_autogaze_selector"]["status"] == "executed"
    assert payload["direct_autogaze_selector"]["sparse_selection_plan_json"] == str(generated_plan_json)
    assert payload["generation"]["text"] == "qwen direct autogaze"


def test_flexible_runner_defaults_qwen_video_nframes_to_num_video_frames():
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "qwen2.5-vl",
            "--model-path",
            "weight/Qwen2.5-VL-7B-Instruct",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen2.5-vl-vision",
            "--mllm-adapter",
            "qwen2.5-vl",
            "--autogaze-integration-level",
            "none",
            "--num-video-frames",
            "32",
            "--video",
            "inputs/large.mp4",
        ]
    )

    assert args.qwen_video_nframes == 32


def test_flexible_runner_accepts_qwen_vit_comparison_mode():
    args = parse_args(
        [
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "pre_encoder_sparse",
            "--pre-encoder-prune-adapter",
            "autogaze-sparse",
            "--qwen-vit-mode",
            "qwen_chunked_vit_autogaze_sparse",
            "--qwen-vit-chunk-frames",
            "8",
            "--qwen-vit-max-spatial-chunks",
            "4",
            "--video",
            "inputs/example.mp4",
            "--num-video-frames",
            "32",
        ]
    )

    assert args.qwen_vit_mode == "qwen_chunked_vit_autogaze_sparse"
    assert args.qwen_vit_chunk_frames == 8
    assert args.qwen_vit_max_spatial_chunks == 4
    assert args.qwen_video_nframes == 32


def test_flexible_runner_supports_llava_onevision_and_internvl_status():
    llava_args = parse_args(
        [
            "--model-family",
            "llava-onevision",
            "--model-path",
            "weight/llava-onevision-qwen2-7b-ov",
            "--token-selector-adapter",
            "autogaze",
            "--vision-encoder-adapter",
            "llava-onevision-siglip",
            "--mllm-adapter",
            "llava-onevision",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
        ]
    )
    internvl_args = parse_args(
        [
            "--model-family",
            "internvl3",
            "--model-path",
            "weight/InternVL3",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "internvl-dynamic-vision",
            "--mllm-adapter",
            "internvl3",
            "--autogaze-integration-level",
            "none",
        ]
    )

    llava_payload = build_inspect_payload(llava_args)
    internvl_payload = build_inspect_payload(internvl_args)

    assert llava_payload["model_capabilities"]["family_group"] == "llava_onevision"
    assert llava_payload["model_capabilities"]["video_token_policy"] == "pooled_196_tokens_per_frame"
    assert llava_payload["mllm_status_matrix"]["llava-onevision"]["pre_encoder_sparse_status"] == "hard"
    assert internvl_payload["model_capabilities"]["family_group"] == "internvl"
    assert internvl_payload["adapter_plan"]["mllm"]["status"]["value"] == "external_cli_ready"
    assert internvl_payload["mllm_status_matrix"]["internvl3"]["native_off_status"] == "external_cli_ready"
    assert internvl_payload["mllm_status_matrix"]["internvl3"]["pre_encoder_sparse_status"] == "probe_required"


def test_flexible_runner_supports_qwen3_vl_moe_family():
    args = parse_args(
        [
            "--model-family",
            "qwen3-vl-moe",
            "--model-path",
            "weight/Qwen3-VL-30B-A3B-Instruct",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen3-vl-moe-vision",
            "--mllm-adapter",
            "qwen3-vl-moe",
            "--autogaze-integration-level",
            "none",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["experiment_spec"]["model_family"] == "qwen3-vl-moe"
    assert payload["model_capabilities"]["family_group"] == "qwen3_vl"
    assert payload["adapter_plan"]["mllm"]["adapter"] == "qwen3-vl-moe"


def test_flexible_runner_keeps_paper_baseline_not_applicable():
    args = parse_args(
        [
            "--model-family",
            "nvila-video-baseline",
            "--model-path",
            "weight/NVILA-8B-Video",
            "--token-selector-adapter",
            "none",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--mllm-adapter",
            "nvila-video",
            "--autogaze-integration-level",
            "none",
        ]
    )

    payload = build_inspect_payload(args)

    assert payload["experiment_spec"]["is_paper_baseline_candidate"] is True
    assert payload["adapter_plan"]["token_selector"]["status"]["value"] == "not_applicable"
    assert payload["paper_baseline_semantics"] == "paper_baseline_candidate"


def test_flexible_runner_writes_inspect_json(tmp_path):
    output_json = tmp_path / "inspect.json"
    args = parse_args(
        [
            "--model-family",
            "nvila-video-plugin",
            "--model-path",
            "weight/NVILA-8B-Video",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "nvila-video-vision",
            "--mllm-adapter",
            "nvila-video",
            "--autogaze-integration-level",
            "none",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_inspect(args)

    assert payload["experiment_spec"]["output_dir"] == str(tmp_path)
    assert Path(payload["output_json"]) == output_json
    assert json.loads(output_json.read_text())["runner"] == "flexible_runner"


def test_flexible_runner_single_dry_run_writes_execution_plan(tmp_path):
    output_json = tmp_path / "single.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--dry-run",
            "--model-family",
            "qwen3-vl",
            "--model-path",
            "weight/Qwen3-VL-8B-Instruct",
            "--token-selector-adapter",
            "keep-all",
            "--vision-encoder-adapter",
            "qwen3-vl-vision",
            "--mllm-adapter",
            "qwen3-vl",
            "--autogaze-integration-level",
            "none",
            "--pre-encoder-prune-adapter",
            "pixelprune",
            "--video",
            "inputs/example.mp4",
            "--attn-implementation",
            "flash_attention_2",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["mode"] == "single"
    assert payload["implementation_status"] == "dry_run"
    assert payload["mllm_runtime"]["adapter"] == "qwen3-vl"
    assert payload["mllm_runtime"]["status"] == "implemented"
    assert payload["mllm_runtime"]["attn_implementation"] == "flash_attention_2"
    assert payload["measurement_plan"]["tokens"]["visual_tokens_before_prune"] is None
    assert payload["post_encoder_prune_runtime"]["status"] == "not_requested"
    assert payload["adapter_plan"]["pre_encoder_prune"]["adapter"] == "pixelprune"
    assert json.loads(output_json.read_text())["implementation_status"] == "dry_run"


def test_flexible_runner_single_dry_run_records_post_prune_probe_plan(tmp_path):
    output_json = tmp_path / "post_prune.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--dry-run",
            "--model-family",
            "qwen2.5-vl",
            "--model-path",
            "weight/Qwen2.5-VL-7B-Instruct",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "qwen2.5-vl-vision",
            "--mllm-adapter",
            "qwen2.5-vl",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["post_encoder_prune_runtime"] == {
        "status": "shape_probe_required",
        "hook": "after vision feature extraction and before MLLM context packing",
        "runtime_behavior": "no pruning applied until visual feature/token shape probe is implemented",
    }
    assert payload["measurement_plan"]["latency_ms"]["generate"] is None


def test_flexible_runner_single_longvila_autogaze_writes_probe_payload(tmp_path):
    output_json = tmp_path / "longvila_probe.json"
    args = parse_args(
        [
            "--mode",
            "single",
            "--model-family",
            "longvila",
            "--model-path",
            "weight/LongVILA",
            "--token-selector-adapter",
            "autogaze",
            "--token-selector-path",
            "weight/AutoGaze",
            "--vision-encoder-adapter",
            "longvila-siglip",
            "--mllm-adapter",
            "longvila",
            "--autogaze-integration-level",
            "post_encoder_token_prune",
            "--video",
            "inputs/example.mp4",
            "--output-json",
            str(output_json),
        ]
    )

    payload = run_single(args)

    assert payload["implementation_status"] == "probe_required"
    assert payload["mllm_runtime"]["status"] == "external_cli_ready"
    assert payload["mllm_runtime"]["feature_packing_probe"]["family_group"] == "longvila"
    assert payload["generation"]["status"] == "probe_required"
    assert payload["generation"]["metrics"]["feature_packing_probe"]["loads_model"] is False
    assert payload["measurement_plan"]["feature_packing_probe"]["required_runtime_checks"] == [
        "load processor/model with trust_remote_code",
        "capture sampled video tensor or frame list contract",
        "capture vision tower output feature shape",
        "capture visual token packing boundary before LLM prefill",
    ]
    assert json.loads(output_json.read_text())["implementation_status"] == "probe_required"

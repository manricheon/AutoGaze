from __future__ import annotations

from dataclasses import dataclass
import math
import re
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import MethodType
from typing import Any

from repro.plugins.gaze_plan import (
    SparseSelectionPlan,
    qwen_visual_indices_from_sparse_plan,
    sparse_selection_plan_from_dict,
)
from repro.vila_feature_probe import run_vila_feature_probe

SIGLIP_DENSE_REFERENCE_TILE_SIZE = 392
SIGLIP_DENSE_REFERENCE_PATCH_SIZE = 14
SIGLIP_DENSE_REFERENCE_TOKEN_SHUFFLE = 9


@dataclass(frozen=True)
class MllmRunRequest:
    model_family: str
    model_path: str
    mllm_adapter: str
    prompt: str
    video: str | None
    image: str | None
    device_map: str
    dtype: str
    attn_implementation: str | None
    trust_remote_code: bool
    max_new_tokens: int
    token_selector_kind: str = "keep-all"
    integration_level: str = "none"
    pre_encoder_prune_adapter: str = "none"
    gazing_ratio: float | None = None
    num_video_frames: int = 128
    num_video_frames_thumbnail: int = 0
    max_tiles_video: int = 1
    external_mllm_command: str = "vila-infer"
    enable_qwen_prune_generate: bool = False
    sparse_selection_plan_path: str | None = None
    qwen_video_nframes: int | None = None
    qwen_video_fps: float | None = None
    qwen_video_max_pixels: int | None = None
    qwen_video_min_pixels: int | None = None
    qwen_vit_mode: str = "qwen_full_vit"
    qwen_vit_chunk_frames: int = 16
    qwen_vit_max_spatial_chunks: int = 1
    qwen_thumbnail_mode: str = "none"
    video_resize_shortest_edge: int | None = None
    video_resize_longest_edge: int | None = None
    video_resize_width: int | None = None
    video_resize_height: int | None = None


@dataclass(frozen=True)
class MllmRunResult:
    text: str | None
    prompt: str
    video: str | None
    image: str | None
    adapter: str
    status: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "video": self.video,
            "image": self.image,
            "adapter": self.adapter,
            "status": self.status,
            "metrics": self.metrics,
        }


class BaseMllmAdapter:
    name = "base"
    runtime_status = "planned"

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "status": self.runtime_status,
            "model_family": request.model_family,
            "model_path": request.model_path,
            "attn_implementation": request.attn_implementation,
            "trust_remote_code": request.trust_remote_code,
            "metric_schema": build_metric_skeleton(request),
        }

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        raise RuntimeError(f"MLLM adapter {self.name!r} is not implemented for execution yet.")


class PlannedMllmAdapter(BaseMllmAdapter):
    runtime_status = "probe_required"

    def __init__(self, name: str):
        self.name = name

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        description = super().describe_runtime(request)
        description["execution_mode"] = "probe_only"
        description["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        return description

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        metrics = build_metric_skeleton(request)
        metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        metrics["metric_status"] = {
            "value": "probe_required",
            "reason": "planned adapter records model-specific feature/token packing checks without loading the MLLM",
        }
        return MllmRunResult(
            text=None,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="probe_required",
            metrics=metrics,
        )


class VilaCliMllmAdapter(BaseMllmAdapter):
    name = "nvila-video"
    runtime_status = "external_cli_ready"

    def __init__(self, name: str = "nvila-video"):
        self.name = name

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        description = super().describe_runtime(request)
        description["execution_mode"] = "external_vila_cli"
        description["external_cli"] = {
            "command": self.build_command(request),
            "max_new_tokens_supported": False,
            "expected_installed_binary": request.external_mllm_command,
            "native_off_only": True,
        }
        description["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        return description

    def build_command(self, request: MllmRunRequest) -> list[str]:
        command = [
            *_split_external_command(request.external_mllm_command),
            "--model-path",
            request.model_path,
            "--conv-mode",
            "auto",
            "--text",
            request.prompt,
        ]
        media = request.video or request.image
        if media:
            command.extend(["--media", media])
        if request.num_video_frames > 0:
            command.extend(["--num_video_frames", str(request.num_video_frames)])
        if request.max_tiles_video > 0:
            command.extend(["--video_max_tiles", str(request.max_tiles_video)])
        return command

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        if request.token_selector_kind == "autogaze" or request.integration_level != "none":
            metrics = build_metric_skeleton(request)
            metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
            vila_probe = run_vila_feature_probe(
                model_path=request.model_path,
                model_family=request.model_family,
                video=request.video or request.image,
                prompt=request.prompt,
                num_video_frames=request.num_video_frames,
                max_tiles_video=request.max_tiles_video,
            )
            metrics["vila_feature_probe"] = vila_probe
            if vila_probe["status"] == "static_probe_collected":
                status = "probe_collected"
                metrics["metric_status"] = {
                    "value": "probe_collected",
                    "reason": "Static VILA-family config probe was collected; runtime feature packing instrumentation is still required.",
                }
            else:
                status = "probe_required"
                metrics["metric_status"] = {
                    "value": "probe_required",
                    "reason": "AutoGaze-on VILA-family integration still requires a feature packing probe",
                }
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status=status,
                metrics=metrics,
            )

        command = self.build_command(request)
        metrics = build_metric_skeleton(request)
        metrics["external_cli"] = {
            "command": command,
            "max_new_tokens_supported": False,
            "stdout_tail": None,
            "stderr_tail": None,
            "returncode": None,
        }
        if not _command_available(command[0]):
            metrics["metric_status"] = {
                "value": "failed_missing_dependency",
                "reason": "vila-infer command was not found",
            }
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_missing_dependency",
                metrics=metrics,
            )

        total_start = time.perf_counter()
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["metric_status"] = {"value": "failed", "reason": str(exc)}
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(total_start)
        metrics["latency_ms"]["total"] = metrics["latency_ms"]["generate"]
        metrics["external_cli"]["returncode"] = int(completed.returncode)
        metrics["external_cli"]["stdout_tail"] = _tail_lines(completed.stdout)
        metrics["external_cli"]["stderr_tail"] = _tail_lines(completed.stderr)
        _record_cuda_memory(metrics)
        text = extract_assistant_text(completed.stdout)
        if completed.returncode != 0:
            metrics["metric_status"] = {
                "value": "failed",
                "reason": "vila-infer returned a non-zero exit code",
            }
            return MllmRunResult(
                text=text,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed",
                metrics=metrics,
            )
        metrics["metric_status"] = {
            "value": "executed",
            "reason": "native/off VILA-family generation was executed through the official VILA CLI",
        }
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )


class InternVL3CliMllmAdapter(BaseMllmAdapter):
    name = "internvl3"
    runtime_status = "external_cli_ready"

    def build_command(self, request: MllmRunRequest) -> list[str]:
        command = [
            *_split_external_command(request.external_mllm_command),
            "--model-path",
            request.model_path,
            "--prompt",
            request.prompt,
        ]
        if request.video:
            command.extend(["--video", request.video])
        if request.image:
            command.extend(["--image", request.image])
        command.extend(["--num-video-frames", str(request.num_video_frames)])
        command.extend(["--max-tiles-video", str(request.max_tiles_video)])
        command.extend(["--max-new-tokens", str(request.max_new_tokens)])
        return command

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        description = super().describe_runtime(request)
        description["execution_mode"] = "external_internvl3_cli"
        description["external_cli"] = {
            "command": self.build_command(request),
            "expected_installed_binary": request.external_mllm_command,
            "native_off_only": True,
        }
        description["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        return description

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        if request.token_selector_kind == "autogaze" or request.integration_level != "none":
            metrics = build_metric_skeleton(request)
            metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
            metrics["metric_status"] = {
                "value": "probe_required",
                "reason": "AutoGaze-on InternVL3 integration still requires a dynamic tiling probe",
            }
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="probe_required",
                metrics=metrics,
            )

        command = self.build_command(request)
        metrics = build_metric_skeleton(request)
        metrics["external_cli"] = {
            "command": command,
            "stdout_tail": None,
            "stderr_tail": None,
            "returncode": None,
        }
        if not _command_available(command[0]):
            metrics["metric_status"] = {
                "value": "failed_missing_dependency",
                "reason": "InternVL3 external command was not found",
            }
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_missing_dependency",
                metrics=metrics,
            )

        total_start = time.perf_counter()
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["metric_status"] = {"value": "failed", "reason": str(exc)}
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(total_start)
        metrics["latency_ms"]["total"] = metrics["latency_ms"]["generate"]
        metrics["external_cli"]["returncode"] = int(completed.returncode)
        metrics["external_cli"]["stdout_tail"] = _tail_lines(completed.stdout)
        metrics["external_cli"]["stderr_tail"] = _tail_lines(completed.stderr)
        _record_cuda_memory(metrics)
        text = extract_assistant_text(completed.stdout)
        if completed.returncode != 0:
            metrics["metric_status"] = {
                "value": "failed",
                "reason": "InternVL3 external command returned a non-zero exit code",
            }
            return MllmRunResult(
                text=text,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed",
                metrics=metrics,
            )
        metrics["metric_status"] = {
            "value": "executed",
            "reason": "native/off InternVL3 generation was executed through the external helper",
        }
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )


class QwenGridMllmAdapter(BaseMllmAdapter):
    name = "qwen-grid"
    runtime_status = "implemented"

    def __init__(self, name: str = "qwen-grid"):
        self.name = name

    def build_messages(self, request: MllmRunRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if request.video:
            video_item: dict[str, Any] = {"type": "video", "video": request.video}
            if request.qwen_video_nframes is not None:
                video_item["nframes"] = int(request.qwen_video_nframes)
            if request.qwen_video_fps is not None:
                video_item["fps"] = float(request.qwen_video_fps)
            if request.qwen_video_max_pixels is not None:
                video_item["max_pixels"] = int(request.qwen_video_max_pixels)
            if request.qwen_video_min_pixels is not None:
                video_item["min_pixels"] = int(request.qwen_video_min_pixels)
            content.append(video_item)
        if request.image:
            content.append({"type": "image", "image": request.image})
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        description = super().describe_runtime(request)
        if request.video:
            description["qwen_video_input"] = {
                "video": request.video,
                "nframes": request.qwen_video_nframes,
                "fps": request.qwen_video_fps,
                "max_pixels": request.qwen_video_max_pixels,
                "min_pixels": request.qwen_video_min_pixels,
                "runner_resize": qwen_runner_resize_summary(request),
                "thumbnail": qwen_thumbnail_summary(request),
                "note": "These constraints are passed to qwen_vl_utils through the video message item.",
            }
        description["qwen_vit"] = {
            "mode": request.qwen_vit_mode,
            "chunk_frames": request.qwen_vit_chunk_frames,
            "max_spatial_chunks": request.qwen_vit_max_spatial_chunks,
            "note": (
                "qwen_full_vit uses the native full-video path; qwen_chunked_vit and "
                "qwen_chunked_vit_autogaze_sparse split Qwen pixel_values_videos after processor build "
                "across temporal chunks and NVILA-style spatial chunks."
            ),
        }
        if request.token_selector_kind == "autogaze" or request.integration_level != "none":
            description["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        return description

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        if request.qwen_vit_mode == "qwen_chunked_vit":
            return self._run_qwen_chunked_vit_generate(request)
        if request.qwen_vit_mode == "qwen_chunked_vit_autogaze_sparse":
            return self._run_qwen_chunked_vit_autogaze_sparse_generate(request)
        if (
            request.token_selector_kind == "autogaze"
            and request.integration_level == "pre_encoder_sparse"
            and request.pre_encoder_prune_adapter == "autogaze-sparse"
        ):
            if request.enable_qwen_prune_generate:
                return self._run_qwen_autogaze_pre_vit_sparse_generate(request)
            return self._run_autogaze_post_encoder_poc(request)
        if request.token_selector_kind == "autogaze" and request.integration_level == "post_encoder_token_prune":
            if request.enable_qwen_prune_generate:
                return self._run_qwen_autogaze_prune_generate(request)
            return self._run_autogaze_post_encoder_poc(request)
        if self._requires_probe_only(request):
            metrics = build_metric_skeleton(request)
            metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
            metrics["metric_status"] = {
                "value": "probe_required",
                "reason": "AutoGaze-on Qwen grid integration requires get_video_features and visual token insertion probe",
            }
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="probe_required",
                metrics=metrics,
            )

        result = self._run_qwen_generate(request)
        if request.pre_encoder_prune_adapter != "none":
            result.metrics["pre_encoder_prune"] = {
                "adapter": request.pre_encoder_prune_adapter,
                "integration_level": request.integration_level,
                "status": "applied_before_model_load",
            }
        return result

    def _run_autogaze_post_encoder_poc(self, request: MllmRunRequest) -> MllmRunResult:
        metrics = build_metric_skeleton(request)
        raw_visual_tokens = _estimate_qwen_visual_tokens(request)
        selected_visual_tokens = _estimate_selected_tokens(raw_visual_tokens, request.gazing_ratio)
        plan = SparseSelectionPlan.placeholder(
            selector_name="autogaze",
            source_path=request.video or request.image,
            raw_patch_tokens=raw_visual_tokens,
            selected_patch_tokens=selected_visual_tokens,
            frame_indices=list(range(max(int(request.num_video_frames), 0))),
            reason=(
                "Qwen AutoGaze post-encoder PoC has a standardized SparseSelectionPlan and "
                "feature packing probe; actual visual feature pruning/insertion still needs "
                "model-specific runtime surgery."
            ),
        )
        metrics["tokens"]["visual_tokens_before_prune"] = raw_visual_tokens
        metrics["tokens"]["visual_tokens_after_prune"] = selected_visual_tokens
        metrics["tokens"]["visual_token_reduction_ratio"] = (
            raw_visual_tokens / selected_visual_tokens if selected_visual_tokens else None
        )
        metrics["sparse_selection_plan"] = plan.to_dict()
        metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        metrics["metric_status"] = {
            "value": "autogaze_qwen_poc_ready",
            "reason": (
                "AutoGaze + Qwen attachment PoC is materialized as a post-encoder SparseSelectionPlan; "
                "it does not yet mutate Qwen visual embeddings for scored generation."
            ),
        }
        return MllmRunResult(
            text=None,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="poc_ready",
            metrics=metrics,
        )

    def _requires_probe_only(self, request: MllmRunRequest) -> bool:
        if request.integration_level == "none" and request.token_selector_kind != "autogaze":
            return False
        if request.integration_level == "pre_encoder_sparse" and request.pre_encoder_prune_adapter == "pixelprune":
            return False
        return True

    def _run_qwen_generate(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for Qwen grid MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation
        metrics = build_metric_skeleton(request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        inputs = _build_qwen_grid_inputs(processor, messages, request)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )

    def _run_qwen_chunked_vit_generate(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for Qwen grid MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation

        metrics = build_metric_skeleton(request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        inputs = _build_qwen_grid_inputs(processor, messages, request)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        try:
            start = time.perf_counter()
            video_features, chunk_metadata = qwen_chunked_video_features(
                model,
                dict(inputs),
                chunk_frames=request.qwen_vit_chunk_frames,
                max_spatial_chunks=request.qwen_vit_max_spatial_chunks,
            )
            packed_inputs = build_qwen_inputs_from_video_features(model, inputs, video_features)
            metrics["latency_ms"]["qwen_vit_prepare"] = _elapsed_ms(start)
        except Exception as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_vit"] = {
                "mode": request.qwen_vit_mode,
                "status": "failed_before_generate",
                "reason": str(exc),
            }
            metrics["metric_status"] = {
                "value": "failed_qwen_chunked_vit",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_chunked_vit",
                metrics=metrics,
            )

        feature_metadata = packed_inputs.pop("qwen_video_feature_metadata")
        metrics["qwen_vit"] = {
            "mode": request.qwen_vit_mode,
            "status": "inputs_embeds_prepared",
            **chunk_metadata,
            **feature_metadata,
        }
        metrics["tokens"]["visual_tokens_before_prune"] = feature_metadata["visual_tokens_before_prune"]
        metrics["tokens"]["visual_tokens_after_prune"] = feature_metadata["visual_tokens_after_prune"]
        metrics["tokens"]["visual_token_reduction_ratio"] = 1.0
        input_ids = packed_inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape"):
            metrics["tokens"]["llm_context_tokens"] = int(input_ids.shape[-1])

        start = time.perf_counter()
        try:
            generated_ids = model.generate(**packed_inputs, max_new_tokens=request.max_new_tokens)
        except Exception as exc:
            metrics["latency_ms"]["generate"] = _elapsed_ms(start)
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_vit"]["status"] = "failed_during_generate"
            metrics["qwen_vit"]["reason"] = str(exc)
            metrics["metric_status"] = {
                "value": "failed_qwen_chunked_vit",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_chunked_vit",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        metrics["metric_status"] = {
            "value": "executed_qwen_chunked_vit",
            "reason": (
                "Qwen ViT features were computed in temporal chunks after processor build, "
                "then packed into the original MLLM visual placeholders."
            ),
        }
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )

    def _run_qwen_chunked_vit_autogaze_sparse_generate(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for Qwen grid MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation

        metrics = build_metric_skeleton(request)
        metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        inputs = _build_qwen_grid_inputs(processor, messages, request)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        try:
            if not request.sparse_selection_plan_path:
                raise ValueError(
                    "qwen_chunked_vit_autogaze_sparse requires --sparse-selection-plan-json "
                    "or --run-autogaze-selector"
                )
            raw_visual_tokens = count_qwen_visual_placeholders(model, inputs)
            mapping = _qwen_mapping_from_sparse_plan_path(
                request.sparse_selection_plan_path,
                model=model,
                inputs=inputs,
            )
            thumbnail_keep, thumbnail_metadata = qwen_thumbnail_visual_keep_indices(
                request,
                video_grid_thw=_qwen_video_grid_thw(inputs),
                spatial_merge_size=qwen_spatial_merge_size(model),
            )
            keep_indices = [
                int(index)
                for index in dict.fromkeys(list(mapping.visual_feature_indices or []) + thumbnail_keep)
                if 0 <= int(index) < int(raw_visual_tokens)
            ]
            if not keep_indices:
                raise ValueError(f"AutoGaze sparse plan did not map to Qwen visual tokens: {mapping.reason}")

            start = time.perf_counter()
            video_features, chunk_metadata = qwen_chunked_video_features(
                model,
                dict(inputs),
                chunk_frames=request.qwen_vit_chunk_frames,
                max_spatial_chunks=request.qwen_vit_max_spatial_chunks,
                keep_indices=keep_indices,
            )
            packed_inputs = build_qwen_inputs_from_video_features(
                model,
                inputs,
                video_features,
                original_keep_indices=keep_indices,
                metadata_key="qwen_chunked_vit_sparse_metadata",
            )
            metrics["latency_ms"]["qwen_vit_prepare"] = _elapsed_ms(start)
        except Exception as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_vit"] = {
                "mode": request.qwen_vit_mode,
                "status": "failed_before_generate",
                "reason": str(exc),
            }
            metrics["metric_status"] = {
                "value": "failed_qwen_chunked_vit_autogaze_sparse",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_chunked_vit_autogaze_sparse",
                metrics=metrics,
            )

        feature_metadata = packed_inputs.pop("qwen_chunked_vit_sparse_metadata")
        metrics["qwen_vit"] = {
            "mode": request.qwen_vit_mode,
            "status": "inputs_embeds_prepared",
            "sparse_selection_plan_path": request.sparse_selection_plan_path,
            "mllm_mapping": mapping.to_dict(),
            "thumbnail_keep_all": thumbnail_metadata,
            **chunk_metadata,
            **feature_metadata,
        }
        metrics["tokens"]["visual_tokens_before_prune"] = feature_metadata["visual_tokens_before_prune"]
        metrics["tokens"]["visual_tokens_after_prune"] = feature_metadata["visual_tokens_after_prune"]
        if feature_metadata["visual_tokens_after_prune"]:
            metrics["tokens"]["visual_token_reduction_ratio"] = (
                feature_metadata["visual_tokens_before_prune"] / feature_metadata["visual_tokens_after_prune"]
            )
        input_ids = packed_inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape"):
            metrics["tokens"]["llm_context_tokens"] = int(input_ids.shape[-1])

        start = time.perf_counter()
        try:
            generated_ids = model.generate(**packed_inputs, max_new_tokens=request.max_new_tokens)
        except Exception as exc:
            metrics["latency_ms"]["generate"] = _elapsed_ms(start)
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_vit"]["status"] = "failed_during_generate"
            metrics["qwen_vit"]["reason"] = str(exc)
            metrics["metric_status"] = {
                "value": "failed_qwen_chunked_vit_autogaze_sparse",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_chunked_vit_autogaze_sparse",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        metrics["metric_status"] = {
            "value": "executed_qwen_chunked_vit_autogaze_sparse",
            "reason": (
                "AutoGaze selected Qwen visual tokens, Qwen ViT ran those sparse tokens in temporal chunks, "
                "and only matching visual placeholders were packed into the MLLM context."
            ),
        }
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )

    def _run_qwen_autogaze_pre_vit_sparse_generate(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for Qwen grid MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation

        metrics = build_metric_skeleton(request)
        metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        inputs = _build_qwen_grid_inputs(processor, messages, request)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        try:
            if not request.sparse_selection_plan_path:
                raise ValueError("Qwen pre-ViT AutoGaze sparse mode requires --sparse-selection-plan-json or --run-autogaze-selector")
            raw_visual_tokens = count_qwen_visual_placeholders(model, inputs)
            mapping = _qwen_mapping_from_sparse_plan_path(
                request.sparse_selection_plan_path,
                model=model,
                inputs=inputs,
            )
            thumbnail_keep, thumbnail_metadata = qwen_thumbnail_visual_keep_indices(
                request,
                video_grid_thw=_qwen_video_grid_thw(inputs),
                spatial_merge_size=qwen_spatial_merge_size(model),
            )
            keep_indices = [
                int(index)
                for index in dict.fromkeys(list(mapping.visual_feature_indices or []) + thumbnail_keep)
                if 0 <= int(index) < int(raw_visual_tokens)
            ]
            if not keep_indices:
                raise ValueError(f"AutoGaze sparse plan did not map to Qwen visual tokens: {mapping.reason}")
            install_qwen_pre_vit_sparse_hook(model, keep_indices)
            start = time.perf_counter()
            pruned_inputs = build_qwen_pre_vit_sparse_visual_inputs(model, inputs, original_keep_indices=keep_indices)
            metrics["latency_ms"]["pre_vit_sparse_prepare"] = _elapsed_ms(start)
        except Exception as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_pre_vit_sparse"] = {
                "enabled": True,
                "status": "failed_before_generate",
                "reason": str(exc),
            }
            metrics["metric_status"] = {
                "value": "failed_qwen_pre_vit_sparse",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_pre_vit_sparse",
                metrics=metrics,
            )

        sparse_metadata = pruned_inputs.pop("qwen_pre_vit_sparse_metadata")
        metrics["qwen_pre_vit_sparse"] = {
            "enabled": True,
            "status": "inputs_embeds_prepared",
            "selection_source": "sparse_selection_plan",
            "sparse_selection_plan_path": request.sparse_selection_plan_path,
            "mllm_mapping": mapping.to_dict(),
            "thumbnail_keep_all": thumbnail_metadata,
            **sparse_metadata,
        }
        metrics["tokens"]["visual_tokens_before_prune"] = sparse_metadata["visual_tokens_before_prune"]
        metrics["tokens"]["visual_tokens_after_prune"] = sparse_metadata["visual_tokens_after_prune"]
        if sparse_metadata["visual_tokens_after_prune"]:
            metrics["tokens"]["visual_token_reduction_ratio"] = (
                sparse_metadata["visual_tokens_before_prune"] / sparse_metadata["visual_tokens_after_prune"]
            )
        input_ids = pruned_inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape"):
            metrics["tokens"]["llm_context_tokens"] = int(input_ids.shape[-1])

        start = time.perf_counter()
        try:
            generated_ids = model.generate(**pruned_inputs, max_new_tokens=request.max_new_tokens)
        except Exception as exc:
            metrics["latency_ms"]["generate"] = _elapsed_ms(start)
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_pre_vit_sparse"]["status"] = "failed_during_generate"
            metrics["qwen_pre_vit_sparse"]["reason"] = str(exc)
            metrics["metric_status"] = {
                "value": "failed_qwen_pre_vit_sparse",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_pre_vit_sparse",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        metrics["metric_status"] = {
            "value": "executed_qwen_pre_vit_sparse",
            "reason": (
                "Qwen visual transformer received AutoGaze-selected sparse merged-token groups, "
                "then the matching visual placeholders were packed into inputs_embeds."
            ),
        }
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )

    def _run_qwen_autogaze_prune_generate(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for Qwen grid MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation

        metrics = build_metric_skeleton(request)
        metrics["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        inputs = _build_qwen_grid_inputs(processor, messages, request)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        try:
            raw_visual_tokens = count_qwen_visual_placeholders(model, inputs)
            keep_indices, selection_metadata = resolve_qwen_visual_keep_indices(
                request,
                model=model,
                inputs=inputs,
                visual_count=raw_visual_tokens,
            )
            pruned_inputs = build_qwen_pruned_visual_inputs(model, inputs, keep_indices)
        except Exception as exc:
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_prune_generate"] = {
                "enabled": True,
                "status": "failed_before_generate",
                "reason": str(exc),
            }
            metrics["metric_status"] = {
                "value": "failed_qwen_prune_generate",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_prune_generate",
                metrics=metrics,
            )

        prune_metadata = pruned_inputs.pop("qwen_prune_generate_metadata")
        metrics["qwen_prune_generate"] = {
            "enabled": True,
            "status": "inputs_embeds_prepared",
            **prune_metadata,
            **selection_metadata,
        }
        metrics["tokens"]["visual_tokens_before_prune"] = prune_metadata["visual_tokens_before_prune"]
        metrics["tokens"]["visual_tokens_after_prune"] = prune_metadata["visual_tokens_after_prune"]
        if prune_metadata["visual_tokens_after_prune"]:
            metrics["tokens"]["visual_token_reduction_ratio"] = (
                prune_metadata["visual_tokens_before_prune"] / prune_metadata["visual_tokens_after_prune"]
            )
        input_ids = pruned_inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape"):
            metrics["tokens"]["llm_context_tokens"] = int(input_ids.shape[-1])

        start = time.perf_counter()
        try:
            generated_ids = model.generate(**pruned_inputs, max_new_tokens=request.max_new_tokens)
        except Exception as exc:
            metrics["latency_ms"]["generate"] = _elapsed_ms(start)
            metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
            metrics["qwen_prune_generate"]["status"] = "failed_during_generate"
            metrics["qwen_prune_generate"]["reason"] = str(exc)
            metrics["metric_status"] = {
                "value": "failed_qwen_prune_generate",
                "reason": str(exc),
            }
            _record_cuda_memory(metrics)
            return MllmRunResult(
                text=None,
                prompt=request.prompt,
                video=request.video,
                image=request.image,
                adapter=self.name,
                status="failed_qwen_prune_generate",
                metrics=metrics,
            )

        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        metrics["metric_status"] = {
            "value": "executed_qwen_prune_generate",
            "reason": "Qwen visual features were pruned after get_video_features and passed to generate through inputs_embeds.",
        }
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )


class LlavaOneVisionMllmAdapter(QwenGridMllmAdapter):
    name = "llava-onevision"

    def build_messages(self, request: MllmRunRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if request.video:
            content.append({"type": "video", "path": request.video})
        if request.image:
            content.append({"type": "image", "path": request.image})
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for LLaVA-OneVision MLLM execution") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": request.device_map,
            "trust_remote_code": request.trust_remote_code,
        }
        if request.dtype != "auto":
            model_kwargs["dtype"] = request.dtype
        if request.attn_implementation:
            model_kwargs["attn_implementation"] = request.attn_implementation
        metrics = build_metric_skeleton(request)
        total_start = time.perf_counter()
        start = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(request.model_path, **model_kwargs)
        metrics["latency_ms"]["model_load"] = _elapsed_ms(start)
        start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(request.model_path, trust_remote_code=request.trust_remote_code)
        metrics["latency_ms"]["processor_load"] = _elapsed_ms(start)
        messages = self.build_messages(request)
        start = time.perf_counter()
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        processor_kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if request.video:
            processor_kwargs["videos"] = request.video
        if request.image:
            processor_kwargs["images"] = request.image
        inputs = processor(**processor_kwargs)
        metrics["latency_ms"]["input_build"] = _elapsed_ms(start)
        _record_input_token_metrics(metrics, inputs)
        model_device = getattr(model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        metrics["latency_ms"]["generate"] = _elapsed_ms(start)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metrics["latency_ms"]["total"] = _elapsed_ms(total_start)
        _record_cuda_memory(metrics)
        return MllmRunResult(
            text=text,
            prompt=request.prompt,
            video=request.video,
            image=request.image,
            adapter=self.name,
            status="executed",
            metrics=metrics,
        )


def resolve_mllm_adapter(name: str) -> BaseMllmAdapter:
    if name in {"nvila-video", "longvila"}:
        return VilaCliMllmAdapter(name)
    if name == "internvl3":
        return InternVL3CliMllmAdapter()
    if name in {"qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen3-vl-moe"}:
        return QwenGridMllmAdapter(name)
    if name == "llava-onevision":
        return LlavaOneVisionMllmAdapter(name)
    return PlannedMllmAdapter(name)


def qwen_runner_resize_enabled(request: MllmRunRequest) -> bool:
    return any(
        value is not None
        for value in (
            request.video_resize_shortest_edge,
            request.video_resize_longest_edge,
            request.video_resize_width,
            request.video_resize_height,
        )
    )


def qwen_runner_resize_summary(request: MllmRunRequest) -> dict[str, Any]:
    enabled = qwen_runner_resize_enabled(request)
    return {
        "enabled": enabled,
        "shortest_edge": request.video_resize_shortest_edge,
        "longest_edge": request.video_resize_longest_edge,
        "width": request.video_resize_width,
        "height": request.video_resize_height,
        "mode": "preloaded_resized_frames" if enabled else "qwen_vl_utils_path_decode",
    }


def qwen_thumbnail_count(request: MllmRunRequest) -> int:
    if request.qwen_thumbnail_mode != "append-video":
        return 0
    return max(0, min(int(request.num_video_frames_thumbnail or 0), qwen_main_video_frame_count(request)))


def qwen_main_video_frame_count(request: MllmRunRequest) -> int:
    return max(1, int(request.qwen_video_nframes or request.num_video_frames or 1))


def qwen_thumbnail_summary(request: MllmRunRequest) -> dict[str, Any]:
    count = qwen_thumbnail_count(request)
    return {
        "mode": request.qwen_thumbnail_mode,
        "requested_frames": int(request.num_video_frames_thumbnail or 0),
        "effective_frames": count,
        "placement": "appended_after_main_video_frames" if count else "none",
        "pruning_policy": "keep_all" if count else "not_applicable",
    }


def read_video_metadata(video: str) -> dict[str, Any]:
    from repro.nvila_runner import read_video_metadata as _read_video_metadata

    return _read_video_metadata(video)


def uniform_sample_indices(total_frames: int, sample_count: int) -> list[int]:
    from repro.nvila_runner import uniform_sample_indices as _uniform_sample_indices

    return _uniform_sample_indices(total_frames, sample_count)


def apply_resize_to_dimensions(**kwargs: Any) -> dict[str, int | str]:
    from repro.nvila_runner import apply_resize_to_dimensions as _apply_resize_to_dimensions

    return _apply_resize_to_dimensions(**kwargs)


def load_sampled_video_frames(
    video: str,
    sample_count: int,
    resize: dict[str, int | str],
    *,
    decode_strategy: str = "auto",
) -> tuple[list[Any], dict[str, Any]]:
    from repro.nvila_runner import load_sampled_video_frames as _load_sampled_video_frames

    return _load_sampled_video_frames(
        video,
        sample_count,
        resize,
        decode_strategy=decode_strategy,
    )


def _qwen_preloaded_video_frames(request: MllmRunRequest) -> tuple[list[Any], dict[str, Any]]:
    if not request.video:
        raise ValueError("Qwen preloaded video frames require request.video")
    sample_count = qwen_main_video_frame_count(request)
    metadata = read_video_metadata(request.video)
    width = metadata.get("width")
    height = metadata.get("height")
    if width is None or height is None:
        raise ValueError("Qwen preloaded video frames require video width/height metadata")
    resize = apply_resize_to_dimensions(
        width=int(width),
        height=int(height),
        shortest_edge=request.video_resize_shortest_edge,
        longest_edge=request.video_resize_longest_edge,
        exact_width=request.video_resize_width,
        exact_height=request.video_resize_height,
    )
    frames, decode_stats = load_sampled_video_frames(
        request.video,
        sample_count,
        resize,
        decode_strategy="auto",
    )
    sampled_indices = uniform_sample_indices(int(metadata.get("frames") or sample_count), sample_count)
    thumbnail_count = qwen_thumbnail_count(request)
    thumbnail_positions = qwen_thumbnail_frame_positions(sample_count, thumbnail_count)
    thumbnail_frames = [frames[position] for position in thumbnail_positions if 0 <= position < len(frames)]
    thumbnail_indices = [sampled_indices[position] for position in thumbnail_positions if 0 <= position < len(sampled_indices)]
    return frames, {
        "mode": "preloaded_resized_frames",
        "sample_count": sample_count,
        "thumbnail_count": len(thumbnail_frames),
        "sampled_frame_indices": sampled_indices,
        "thumbnail_frame_indices": thumbnail_indices,
        "source_metadata": metadata,
        "resize": resize,
        "decode": decode_stats,
        "frames_for_processor": frames + thumbnail_frames,
    }


def qwen_thumbnail_frame_positions(main_count: int, thumbnail_count: int) -> list[int]:
    count = max(0, min(int(thumbnail_count), int(main_count)))
    if count <= 0:
        return []
    if main_count > count:
        step = max(1, main_count // count)
        return list(range(0, main_count, step))[:count]
    return list(range(main_count))


def _build_qwen_grid_inputs(processor: Any, messages: list[dict[str, Any]], request: MllmRunRequest) -> Any:
    if request.video:
        if qwen_runner_resize_enabled(request) or qwen_thumbnail_count(request):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                frames, metadata = _qwen_preloaded_video_frames(request)
            except Exception as exc:
                raise RuntimeError(
                    "Qwen runner-side video resize/decode failed before processor tokenization. "
                    f"{_qwen_video_debug_context(request)}"
                ) from exc
            processor_kwargs: dict[str, Any] = {
                "text": [text],
                "images": None,
                "videos": [metadata["frames_for_processor"]],
                "return_tensors": "pt",
            }
            if not qwen_runner_resize_enabled(request):
                if request.qwen_video_max_pixels is not None:
                    processor_kwargs["max_pixels"] = int(request.qwen_video_max_pixels)
                if request.qwen_video_min_pixels is not None:
                    processor_kwargs["min_pixels"] = int(request.qwen_video_min_pixels)
            if request.qwen_video_fps is not None:
                processor_kwargs["fps"] = float(request.qwen_video_fps)
            try:
                return processor(**processor_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "Qwen processor failed after runner-side video resize/decode. "
                    f"{_qwen_video_debug_context(request)} "
                    "Check that the Qwen processor accepts preloaded video frame lists for this checkpoint."
                ) from exc
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "qwen_vl_utils is required for Qwen-style video inputs. "
                "Install it with `.venv/bin/python -m pip install qwen-vl-utils` "
                "or rerun `.venv/bin/python -m pip install -r requirements-repro.txt`."
            ) from exc
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        except Exception as exc:
            raise RuntimeError(
                "Qwen video decode/preprocess failed before processor tokenization. "
                f"{_qwen_video_debug_context(request)} "
                "For large videos, set --qwen-video-nframes to a small sampled frame count and "
                "--qwen-video-max-pixels to cap per-frame resolution."
            ) from exc
        try:
            return processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
                **video_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen processor failed after video decode/preprocess. "
                f"{_qwen_video_debug_context(request)} "
                "Check that the processor/model family matches the checkpoint and that video frame/resolution "
                "limits are set for large inputs."
            ) from exc
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _qwen_video_debug_context(request: MllmRunRequest) -> str:
    return (
        f"video={request.video!r}, nframes={request.qwen_video_nframes!r}, fps={request.qwen_video_fps!r}, "
        f"max_pixels={request.qwen_video_max_pixels!r}, min_pixels={request.qwen_video_min_pixels!r}, "
        f"runner_resize={qwen_runner_resize_summary(request)!r}, thumbnail={qwen_thumbnail_summary(request)!r}"
    )


def build_qwen_pruned_visual_inputs(model: Any, inputs: Any, keep_indices: list[int]) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Qwen visual feature pruning") from exc

    values = dict(inputs)
    input_ids = values.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise ValueError("Qwen prune-generate requires input_ids in processor outputs")
    if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Qwen prune-generate currently supports a single prompt/video batch")

    video_token_id = qwen_video_token_id(model)
    embeddings = _qwen_input_embeddings(model)(input_ids)
    video_features = _qwen_video_features(model, values)
    if len(video_features.shape) == 3:
        if int(video_features.shape[0]) != 1:
            raise ValueError("Qwen prune-generate currently supports one video feature batch")
        video_features = video_features[0]
    if len(video_features.shape) != 2:
        raise ValueError(f"Expected 2D Qwen video features, got shape={list(video_features.shape)}")

    visual_count = int(video_features.shape[0])
    keep = [index for index in dict.fromkeys(int(item) for item in keep_indices) if 0 <= index < visual_count]
    if not keep and visual_count > 0:
        keep = [0]
    placeholder_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if int(placeholder_positions.numel()) < visual_count:
        raise ValueError(
            "Qwen input has fewer video token placeholders than video features "
            f"({int(placeholder_positions.numel())} < {visual_count})"
        )

    visual_positions = placeholder_positions[:visual_count]
    keep_positions = visual_positions[torch.tensor(keep, device=visual_positions.device, dtype=torch.long)]
    keep_position_set = {int(item) for item in keep_positions.detach().cpu().tolist()}
    sequence_keep_mask = torch.ones(input_ids.shape[-1], dtype=torch.bool, device=input_ids.device)
    for position in visual_positions.detach().cpu().tolist():
        if int(position) not in keep_position_set:
            sequence_keep_mask[int(position)] = False

    pruned: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"}:
            continue
        if hasattr(value, "shape") and len(value.shape) >= 2 and int(value.shape[0]) == 1 and int(value.shape[1]) == int(input_ids.shape[-1]):
            pruned[key] = value[:, sequence_keep_mask, ...]
        else:
            pruned[key] = value

    pruned_embeddings = embeddings[:, sequence_keep_mask, :].clone()
    kept_features = video_features[torch.tensor(keep, device=video_features.device, dtype=torch.long)]
    kept_features = kept_features.to(device=pruned_embeddings.device, dtype=pruned_embeddings.dtype)
    new_positions: list[int] = []
    for original_position in keep_positions.detach().cpu().tolist():
        new_positions.append(int(sequence_keep_mask[: int(original_position) + 1].sum().item()) - 1)
    for row, new_position in enumerate(new_positions):
        pruned_embeddings[0, new_position, :] = kept_features[row]
    pruned["inputs_embeds"] = pruned_embeddings
    pruned["input_ids"] = input_ids[:, sequence_keep_mask]
    if "attention_mask" in values:
        pruned["attention_mask"] = values["attention_mask"][:, sequence_keep_mask]
    pruned["qwen_prune_generate_metadata"] = {
        "video_token_id": video_token_id,
        "visual_tokens_before_prune": visual_count,
        "visual_tokens_after_prune": len(keep),
        "dropped_visual_placeholders": visual_count - len(keep),
        "kept_visual_indices": keep,
        "kept_visual_placeholder_positions": [int(item) for item in keep_positions.detach().cpu().tolist()],
        "llm_context_tokens_after_prune": int(pruned["input_ids"].shape[-1]),
    }
    return pruned


def build_qwen_pre_vit_sparse_visual_inputs(
    model: Any,
    inputs: Any,
    *,
    original_keep_indices: list[int],
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Qwen pre-ViT sparse visual pruning") from exc

    values = dict(inputs)
    input_ids = values.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise ValueError("Qwen pre-ViT sparse generate requires input_ids in processor outputs")
    if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Qwen pre-ViT sparse generate currently supports a single prompt/video batch")

    video_token_id = qwen_video_token_id(model)
    embeddings = _qwen_input_embeddings(model)(input_ids)
    placeholder_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    raw_visual_count = int(placeholder_positions.numel())
    keep = [index for index in dict.fromkeys(int(item) for item in original_keep_indices) if 0 <= index < raw_visual_count]
    if not keep and raw_visual_count > 0:
        keep = [0]

    video_features = _qwen_video_features(model, values)
    if len(video_features.shape) == 3:
        if int(video_features.shape[0]) != 1:
            raise ValueError("Qwen pre-ViT sparse generate currently supports one video feature batch")
        video_features = video_features[0]
    if len(video_features.shape) != 2:
        raise ValueError(f"Expected 2D sparse Qwen video features, got shape={list(video_features.shape)}")
    if int(video_features.shape[0]) != len(keep):
        raise ValueError(
            "Qwen pre-ViT sparse features must match selected original visual placeholders "
            f"({int(video_features.shape[0])} != {len(keep)})"
        )

    keep_positions = placeholder_positions[torch.tensor(keep, device=placeholder_positions.device, dtype=torch.long)]
    keep_position_set = {int(item) for item in keep_positions.detach().cpu().tolist()}
    sequence_keep_mask = torch.ones(input_ids.shape[-1], dtype=torch.bool, device=input_ids.device)
    for position in placeholder_positions.detach().cpu().tolist():
        if int(position) not in keep_position_set:
            sequence_keep_mask[int(position)] = False

    pruned: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"}:
            continue
        if (
            hasattr(value, "shape")
            and len(value.shape) >= 2
            and int(value.shape[0]) == 1
            and int(value.shape[1]) == int(input_ids.shape[-1])
        ):
            pruned[key] = value[:, sequence_keep_mask, ...]
        else:
            pruned[key] = value

    pruned_embeddings = embeddings[:, sequence_keep_mask, :].clone()
    sparse_features = video_features.to(device=pruned_embeddings.device, dtype=pruned_embeddings.dtype)
    new_positions: list[int] = []
    for original_position in keep_positions.detach().cpu().tolist():
        new_positions.append(int(sequence_keep_mask[: int(original_position) + 1].sum().item()) - 1)
    for row, new_position in enumerate(new_positions):
        pruned_embeddings[0, new_position, :] = sparse_features[row]
    pruned["inputs_embeds"] = pruned_embeddings
    pruned["input_ids"] = input_ids[:, sequence_keep_mask]
    if "attention_mask" in values:
        pruned["attention_mask"] = values["attention_mask"][:, sequence_keep_mask]
    pruned["qwen_pre_vit_sparse_metadata"] = {
        "video_token_id": video_token_id,
        "visual_tokens_before_prune": raw_visual_count,
        "visual_tokens_after_prune": len(keep),
        "dropped_visual_placeholders": raw_visual_count - len(keep),
        "kept_original_visual_indices": keep,
        "kept_visual_placeholder_positions": [int(item) for item in keep_positions.detach().cpu().tolist()],
        "llm_context_tokens_after_prune": int(pruned["input_ids"].shape[-1]),
    }
    return pruned


def build_qwen_inputs_from_video_features(
    model: Any,
    inputs: Any,
    video_features: Any,
    *,
    original_keep_indices: list[int] | None = None,
    metadata_key: str = "qwen_video_feature_metadata",
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Qwen visual feature packing") from exc

    values = dict(inputs)
    input_ids = values.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise ValueError("Qwen video feature packing requires input_ids in processor outputs")
    if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Qwen video feature packing currently supports a single prompt/video batch")

    if len(video_features.shape) == 3:
        if int(video_features.shape[0]) != 1:
            raise ValueError("Qwen video feature packing currently supports one video feature batch")
        video_features = video_features[0]
    if len(video_features.shape) != 2:
        raise ValueError(f"Expected 2D Qwen video features, got shape={list(video_features.shape)}")

    video_token_id = qwen_video_token_id(model)
    embeddings = _qwen_input_embeddings(model)(input_ids)
    placeholder_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    raw_visual_count = int(placeholder_positions.numel())
    if original_keep_indices is None:
        keep = list(range(raw_visual_count))
    else:
        keep = [
            index
            for index in dict.fromkeys(int(item) for item in original_keep_indices)
            if 0 <= index < raw_visual_count
        ]
        if not keep and raw_visual_count > 0:
            keep = [0]
    if int(video_features.shape[0]) != len(keep):
        raise ValueError(
            "Qwen video features must match selected visual placeholders "
            f"({int(video_features.shape[0])} != {len(keep)})"
        )

    keep_positions = placeholder_positions[torch.tensor(keep, device=placeholder_positions.device, dtype=torch.long)]
    keep_position_set = {int(item) for item in keep_positions.detach().cpu().tolist()}
    sequence_keep_mask = torch.ones(input_ids.shape[-1], dtype=torch.bool, device=input_ids.device)
    for position in placeholder_positions.detach().cpu().tolist():
        if int(position) not in keep_position_set:
            sequence_keep_mask[int(position)] = False

    packed: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"}:
            continue
        if (
            hasattr(value, "shape")
            and len(value.shape) >= 2
            and int(value.shape[0]) == 1
            and int(value.shape[1]) == int(input_ids.shape[-1])
        ):
            packed[key] = value[:, sequence_keep_mask, ...]
        else:
            packed[key] = value

    packed_embeddings = embeddings[:, sequence_keep_mask, :].clone()
    features = video_features.to(device=packed_embeddings.device, dtype=packed_embeddings.dtype)
    new_positions: list[int] = []
    for original_position in keep_positions.detach().cpu().tolist():
        new_positions.append(int(sequence_keep_mask[: int(original_position) + 1].sum().item()) - 1)
    for row, new_position in enumerate(new_positions):
        packed_embeddings[0, new_position, :] = features[row]
    packed["inputs_embeds"] = packed_embeddings
    packed["input_ids"] = input_ids[:, sequence_keep_mask]
    if "attention_mask" in values:
        packed["attention_mask"] = values["attention_mask"][:, sequence_keep_mask]
    metadata: dict[str, Any] = {
        "video_token_id": video_token_id,
        "visual_tokens_before_prune": raw_visual_count,
        "visual_tokens_after_prune": len(keep),
        "dropped_visual_placeholders": raw_visual_count - len(keep),
        "llm_context_tokens_after_prune": int(packed["input_ids"].shape[-1]),
    }
    if original_keep_indices is not None:
        metadata["kept_original_visual_indices"] = keep
        metadata["kept_visual_placeholder_positions"] = [
            int(item) for item in keep_positions.detach().cpu().tolist()
        ]
    packed[metadata_key] = metadata
    return packed


def resolve_qwen_visual_keep_indices(
    request: MllmRunRequest,
    *,
    model: Any,
    inputs: Any,
    visual_count: int,
) -> tuple[list[int], dict[str, Any]]:
    if request.sparse_selection_plan_path:
        mapping = _qwen_mapping_from_sparse_plan_path(
            request.sparse_selection_plan_path,
            model=model,
            inputs=inputs,
        )
        keep = [
            int(index)
            for index in (mapping.visual_feature_indices or [])
            if 0 <= int(index) < int(visual_count)
        ]
        thumbnail_keep, thumbnail_metadata = qwen_thumbnail_visual_keep_indices(
            request,
            video_grid_thw=_qwen_video_grid_thw(inputs),
            spatial_merge_size=qwen_spatial_merge_size(model),
        )
        keep = [int(index) for index in dict.fromkeys(keep + thumbnail_keep) if 0 <= int(index) < int(visual_count)]
        if keep:
            return keep, {
                "selection_source": "sparse_selection_plan",
                "sparse_selection_plan_path": request.sparse_selection_plan_path,
                "mllm_mapping": mapping.to_dict(),
                "thumbnail_keep_all": thumbnail_metadata,
            }
        return select_qwen_visual_keep_indices(visual_count, request.gazing_ratio), {
            "selection_source": "gazing_ratio_fallback_after_sparse_plan_mapping_failed",
            "sparse_selection_plan_path": request.sparse_selection_plan_path,
            "mllm_mapping": mapping.to_dict(),
        }
    return select_qwen_visual_keep_indices(visual_count, request.gazing_ratio), {
        "selection_source": "gazing_ratio_placeholder_until_autogaze_indices_are_mapped",
    }


def qwen_thumbnail_visual_keep_indices(
    request: MllmRunRequest,
    *,
    video_grid_thw: Any,
    spatial_merge_size: int = 1,
) -> tuple[list[int], dict[str, Any]]:
    thumbnail_count = qwen_thumbnail_count(request)
    if thumbnail_count <= 0 or video_grid_thw is None:
        return [], {
            "enabled": False,
            "reason": "thumbnail append-video mode is not active or video_grid_thw is unavailable",
        }
    t, h, w = _single_qwen_grid(video_grid_thw)
    merge = max(1, int(spatial_merge_size or 1))
    merged_h = max(1, math.ceil(h / merge))
    merged_w = max(1, math.ceil(w / merge))
    main_count = qwen_main_video_frame_count(request)
    total_count = max(main_count + thumbnail_count, 1)
    thumbnail_start = min(max(int(round(main_count * t / total_count)), 0), t)
    per_frame_tokens = merged_h * merged_w
    indices = list(range(thumbnail_start * per_frame_tokens, t * per_frame_tokens))
    return indices, {
        "enabled": True,
        "mode": request.qwen_thumbnail_mode,
        "main_frames": main_count,
        "thumbnail_frames": thumbnail_count,
        "qwen_temporal_tokens": t,
        "thumbnail_temporal_start": thumbnail_start,
        "thumbnail_visual_tokens": len(indices),
        "pruning_policy": "keep_all",
    }


def _qwen_mapping_from_sparse_plan_path(
    path: str,
    *,
    model: Any,
    inputs: Any,
) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        plan = sparse_selection_plan_from_dict(json.load(handle))
    video_grid_thw = _qwen_video_grid_thw(inputs)
    if video_grid_thw is None:
        from repro.plugins.gaze_plan import MllmMapping

        return MllmMapping(
            status="mapping_failed",
            visual_feature_indices=[],
            reason="Qwen processor outputs did not include video_grid_thw",
        )
    return qwen_visual_indices_from_sparse_plan(
        plan,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=qwen_spatial_merge_size(model),
    )


def _qwen_video_grid_thw(inputs: Any) -> Any:
    values = dict(inputs)
    return values.get("video_grid_thw")


def count_qwen_visual_placeholders(model: Any, inputs: Any) -> int:
    values = dict(inputs)
    input_ids = values.get("input_ids")
    if input_ids is None:
        return 0
    video_token_id = qwen_video_token_id(model)
    return int((input_ids == video_token_id).sum().item())


def select_qwen_visual_keep_indices(total_visual_tokens: int, gazing_ratio: float | None) -> list[int]:
    total = max(int(total_visual_tokens), 0)
    if total == 0:
        return []
    if gazing_ratio is None:
        return list(range(total))
    ratio = max(0.0, min(1.0, float(gazing_ratio)))
    count = max(1, min(total, int(round(total * ratio))))
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def qwen_grid_chunk_slices(
    video_grid_thw: Any,
    *,
    spatial_merge_size: int,
    chunk_frames: int,
    max_spatial_chunks: int = 1,
) -> list[dict[str, Any]]:
    t, h, w = _single_qwen_grid(video_grid_thw)
    chunk = max(1, int(chunk_frames))
    merge = max(1, int(spatial_merge_size))
    spatial_merge_unit = merge**2
    if h % merge != 0 or w % merge != 0:
        raise ValueError(
            "Qwen temporal/spatial chunking requires H and W to be divisible by spatial_merge_size "
            f"(H={h}, W={w}, spatial_merge_size={merge})"
        )
    raw_tokens_per_t = h * w
    merged_h = h // merge
    merged_w = w // merge
    merged_tokens_per_t = merged_h * merged_w
    tile_grid = qwen_spatial_tile_grid(
        merged_width=merged_w,
        merged_height=merged_h,
        max_spatial_chunks=max_spatial_chunks,
    )
    col_ranges = _split_index_ranges(merged_w, int(tile_grid["cols"]))
    row_ranges = _split_index_ranges(merged_h, int(tile_grid["rows"]))
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for t_start in range(0, t, chunk):
        t_end = min(t_start + chunk, t)
        for tile_row, (merged_row_start, merged_row_end) in enumerate(row_ranges):
            for tile_col, (merged_col_start, merged_col_end) in enumerate(col_ranges):
                if merged_row_start == merged_row_end or merged_col_start == merged_col_end:
                    continue
                spatial_tile_index = tile_row * int(tile_grid["cols"]) + tile_col
                merged_token_indices: list[int] = []
                merged_token_raw_indices: list[list[int]] = []
                raw_token_indices: list[int] = []
                for t_index in range(t_start, t_end):
                    for merged_row in range(merged_row_start, merged_row_end):
                        for merged_col in range(merged_col_start, merged_col_end):
                            merged_index = t_index * merged_tokens_per_t + merged_row * merged_w + merged_col
                            group_raw = _qwen_raw_indices_for_merged_token(
                                t_index=t_index,
                                merged_row=merged_row,
                                merged_col=merged_col,
                                h=h,
                                w=w,
                                merge=merge,
                            )
                            merged_token_indices.append(merged_index)
                            merged_token_raw_indices.append(group_raw)
                            raw_token_indices.extend(group_raw)
                chunk_payload = {
                    "chunk_index": chunk_index,
                    "t_start": t_start,
                    "t_end": t_end,
                    "t": t_end - t_start,
                    "h": h,
                    "w": w,
                    "spatial_tile_index": spatial_tile_index,
                    "tile_grid_cols": int(tile_grid["cols"]),
                    "tile_grid_rows": int(tile_grid["rows"]),
                    "spatial_tiles": int(tile_grid["tiles"]),
                    "merged_row_start": merged_row_start,
                    "merged_row_end": merged_row_end,
                    "merged_col_start": merged_col_start,
                    "merged_col_end": merged_col_end,
                    "raw_token_indices": raw_token_indices,
                    "merged_token_indices": merged_token_indices,
                    "merged_token_raw_indices": merged_token_raw_indices,
                }
                if int(tile_grid["tiles"]) == 1:
                    chunk_payload.update(
                        {
                            "raw_token_start": t_start * raw_tokens_per_t,
                            "raw_token_end": t_end * raw_tokens_per_t,
                            "merged_token_start": t_start * merged_tokens_per_t,
                            "merged_token_end": t_end * merged_tokens_per_t,
                        }
                    )
                chunks.append(chunk_payload)
                chunk_index += 1
    return chunks


def qwen_spatial_tile_grid(*, merged_width: int, merged_height: int, max_spatial_chunks: int) -> dict[str, int]:
    width = max(1, int(merged_width))
    height = max(1, int(merged_height))
    max_tiles = max(1, min(int(max_spatial_chunks), width * height))
    target_ratios = {
        (cols, rows)
        for n in range(1, max_tiles + 1)
        for cols in range(1, n + 1)
        for rows in range(1, n + 1)
        if 1 <= cols * rows <= max_tiles and cols <= width and rows <= height
    }
    sorted_ratios = sorted(target_ratios, key=lambda item: item[0] * item[1])
    cols, rows = _closest_spatial_ratio(width / height, sorted_ratios, width, height)
    return {"cols": cols, "rows": rows, "tiles": cols * rows}


def _closest_spatial_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _split_index_ranges(length: int, parts: int) -> list[tuple[int, int]]:
    total = max(1, int(length))
    count = max(1, min(int(parts), total))
    return [
        (int(index * total // count), int((index + 1) * total // count))
        for index in range(count)
    ]


def _qwen_raw_indices_for_merged_token(
    *,
    t_index: int,
    merged_row: int,
    merged_col: int,
    h: int,
    w: int,
    merge: int,
) -> list[int]:
    base = int(t_index) * int(h) * int(w)
    row_start = int(merged_row) * int(merge)
    col_start = int(merged_col) * int(merge)
    return [
        base + (row_start + row_offset) * int(w) + col_start + col_offset
        for row_offset in range(int(merge))
        for col_offset in range(int(merge))
    ]


def qwen_chunked_video_features(
    model: Any,
    values: dict[str, Any],
    *,
    chunk_frames: int,
    max_spatial_chunks: int = 1,
    keep_indices: list[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Qwen chunked ViT execution") from exc

    pixel_values_videos = values.get("pixel_values_videos")
    video_grid_thw = values.get("video_grid_thw")
    if pixel_values_videos is None or video_grid_thw is None:
        raise ValueError("Qwen chunked ViT requires pixel_values_videos and video_grid_thw")

    visual = _qwen_visual_module(model)
    spatial_merge_size = qwen_spatial_merge_size(model)
    spatial_merge_unit = max(1, int(spatial_merge_size) ** 2)
    chunks = qwen_grid_chunk_slices(
        video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        chunk_frames=chunk_frames,
        max_spatial_chunks=max_spatial_chunks,
    )
    keep = None
    if keep_indices is not None:
        keep = [int(index) for index in dict.fromkeys(keep_indices)]
    feature_parts: list[Any] = []
    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        merged_token_indices = [int(index) for index in chunk["merged_token_indices"]]
        merged_token_raw_indices = chunk["merged_token_raw_indices"]
        local_keep: list[int] | None
        if keep is None:
            local_keep = list(range(len(merged_token_indices)))
        else:
            local_keep = [
                local_index
                for local_index, global_index in enumerate(merged_token_indices)
                if global_index in keep
            ]
        if not local_keep:
            chunk_records.append({**chunk, "selected_merged_tokens": 0, "status": "skipped_empty_selection"})
            continue

        selected_raw_token_indices = [
            raw_index
            for local_index in local_keep
            for raw_index in merged_token_raw_indices[local_index]
        ]
        chunk_pixels = _slice_qwen_video_pixel_values_by_indices(
            pixel_values_videos,
            selected_raw_token_indices,
        )
        chunk_grid = _make_qwen_grid_like(
            video_grid_thw,
            t=chunk["t"],
            h=chunk["h"],
            w=chunk["w"],
        )
        features = _qwen_sparse_visual_forward(
            visual,
            chunk_pixels,
            chunk_grid,
            list(range(len(local_keep))),
            full_video_grid_thw=video_grid_thw,
            raw_token_indices=selected_raw_token_indices,
        )
        feature_parts.append(features)
        chunk_records.append({**chunk, "selected_merged_tokens": len(local_keep), "status": "executed"})

    if not feature_parts:
        raise ValueError("Qwen chunked ViT produced no features; check AutoGaze sparse keep indices")
    merged_features = torch.cat(feature_parts, dim=0)
    t, h, w = _single_qwen_grid(video_grid_thw)
    visual_tokens_before_prune = (t * h * w) // spatial_merge_unit
    metadata = {
        "chunk_frames": max(1, int(chunk_frames)),
        "chunk_count": len(chunks),
        "executed_chunk_count": len(feature_parts),
        "video_grid_thw": [t, h, w],
        "spatial_merge_size": spatial_merge_size,
        "spatial_merge_unit": spatial_merge_unit,
        "raw_patch_tokens_before_vit": t * h * w,
        "visual_tokens_before_prune": visual_tokens_before_prune,
        "visual_tokens_after_prune": int(merged_features.shape[0]),
        "processor_chunking": "after_qwen_processor_pixel_values_videos",
        "spatial_chunking": {
            "mode": "qwen_processor_grid_spatial_tiles",
            "max_spatial_chunks": max(1, int(max_spatial_chunks)),
            "tile_grid": {
                "cols": int(chunks[0]["tile_grid_cols"]) if chunks else 1,
                "rows": int(chunks[0]["tile_grid_rows"]) if chunks else 1,
                "tiles": int(chunks[0]["spatial_tiles"]) if chunks else 1,
            },
        },
        "position_policy": "full_video_rotary_position_embedding_gathered_by_raw_token_indices",
        "chunks": chunk_records,
    }
    return merged_features, metadata


def _single_qwen_grid(video_grid_thw: Any) -> tuple[int, int, int]:
    if hasattr(video_grid_thw, "detach"):
        grid = video_grid_thw.detach().cpu().tolist()
    elif hasattr(video_grid_thw, "tolist"):
        grid = video_grid_thw.tolist()
    else:
        grid = video_grid_thw
    if isinstance(grid, tuple):
        grid = list(grid)
    if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], (list, tuple)):
        grid = grid[0]
    if not (isinstance(grid, list) and len(grid) == 3):
        raise ValueError(f"Qwen chunked ViT currently supports a single video grid, got {grid!r}")
    return int(grid[0]), int(grid[1]), int(grid[2])


def _slice_qwen_video_pixel_values_by_indices(pixel_values_videos: Any, indices: list[int]) -> Any:
    if not indices:
        raise ValueError("Qwen pixel_values_videos chunk requires at least one raw token index")
    if not hasattr(pixel_values_videos, "shape"):
        return [pixel_values_videos[index] for index in indices]
    import torch

    shape = list(pixel_values_videos.shape)
    max_index = max(int(index) for index in indices)
    if shape and int(shape[0]) > max_index:
        index_tensor = torch.tensor(indices, device=pixel_values_videos.device, dtype=torch.long)
        return pixel_values_videos.index_select(0, index_tensor)
    if len(shape) >= 2 and int(shape[0]) == 1 and int(shape[1]) > max_index:
        index_tensor = torch.tensor(indices, device=pixel_values_videos.device, dtype=torch.long)
        return pixel_values_videos.index_select(1, index_tensor)
    raise ValueError(
        "Qwen pixel_values_videos token dimension is smaller than the requested chunk "
        f"(shape={shape}, max_index={max_index})"
    )


def _make_qwen_grid_like(reference_grid: Any, *, t: int, h: int, w: int) -> Any:
    if hasattr(reference_grid, "detach"):
        import torch

        return torch.tensor(
            [[int(t), int(h), int(w)]],
            device=reference_grid.device,
            dtype=reference_grid.dtype,
        )
    return [[int(t), int(h), int(w)]]


def qwen_video_token_id(model: Any) -> int:
    for target in (getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)):
        value = getattr(target, "video_token_id", None)
        if value is not None:
            return int(value)
    raise ValueError("Qwen model config does not expose video_token_id")


def qwen_spatial_merge_size(model: Any) -> int:
    candidates = (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "config", None), "vision_config", None),
    )
    for target in candidates:
        value = getattr(target, "spatial_merge_size", None)
        if value is not None:
            return max(1, int(value))
    return 1


def _qwen_input_embeddings(model: Any) -> Any:
    for target in (model, getattr(model, "model", None)):
        getter = getattr(target, "get_input_embeddings", None)
        if getter is not None:
            embeddings = getter()
            if embeddings is not None:
                return embeddings
    raise ValueError("Qwen model does not expose get_input_embeddings")


def _qwen_video_features(model: Any, values: dict[str, Any]) -> Any:
    getter = None
    for target in (model, getattr(model, "model", None)):
        getter = getattr(target, "get_video_features", None)
        if getter is not None:
            break
    if getter is None:
        raise ValueError("Qwen model does not expose get_video_features")
    pixel_values_videos = values.get("pixel_values_videos")
    video_grid_thw = values.get("video_grid_thw")
    try:
        output = getter(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw)
    except TypeError:
        output = getter(pixel_values_videos, video_grid_thw)
    return _first_tensor_like(output)


def install_qwen_pre_vit_sparse_hook(model: Any, keep_indices: list[int]) -> dict[str, Any]:
    visual = _qwen_visual_module(model)
    keep = [int(index) for index in dict.fromkeys(keep_indices)]
    if not keep:
        raise ValueError("Qwen pre-ViT sparse hook requires at least one selected visual token")

    def sparse_get_video_features(self: Any, pixel_values_videos: Any, video_grid_thw: Any = None):
        return _qwen_sparse_visual_forward(visual, pixel_values_videos, video_grid_thw, keep)

    patched_targets = []
    for target in (model, getattr(model, "model", None)):
        if target is not None and getattr(target, "get_video_features", None) is not None:
            setattr(target, "get_video_features", MethodType(sparse_get_video_features, target))
            patched_targets.append(type(target).__name__)
    if not patched_targets:
        raise ValueError("Qwen model does not expose get_video_features for sparse hook installation")
    return {
        "patched_targets": patched_targets,
        "selected_merged_tokens": len(keep),
        "spatial_merge_size": qwen_spatial_merge_size(model),
    }


def _qwen_visual_module(model: Any) -> Any:
    for target in (model, getattr(model, "model", None)):
        visual = getattr(target, "visual", None)
        if visual is not None:
            return visual
    raise ValueError("Qwen model does not expose a visual module for pre-ViT sparse pruning")


def _qwen_sparse_visual_forward(
    visual: Any,
    pixel_values_videos: Any,
    video_grid_thw: Any,
    keep_indices: list[int],
    *,
    full_video_grid_thw: Any | None = None,
    raw_token_offset: int = 0,
    raw_token_indices: list[int] | None = None,
) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Qwen pre-ViT sparse visual pruning") from exc

    required = ["patch_embed", "rot_pos_emb", "blocks", "merger", "spatial_merge_unit"]
    missing = [name for name in required if not hasattr(visual, name)]
    if missing:
        raise ValueError(f"Qwen visual module does not support sparse hook; missing {missing}")

    target_dtype = getattr(visual, "dtype", None)
    if target_dtype is not None and hasattr(pixel_values_videos, "type"):
        pixel_values_videos = pixel_values_videos.type(target_dtype)
    hidden_states = visual.patch_embed(pixel_values_videos)
    seq_len, _ = hidden_states.size()
    spatial_merge_unit = int(visual.spatial_merge_unit)
    if seq_len % spatial_merge_unit != 0:
        raise ValueError(
            f"Qwen sparse visual tokens must be divisible by spatial_merge_unit ({seq_len} % {spatial_merge_unit})"
        )
    total_merged_tokens = seq_len // spatial_merge_unit
    keep = [index for index in dict.fromkeys(int(item) for item in keep_indices) if 0 <= index < total_merged_tokens]
    if not keep:
        keep = [0]
    keep_tensor = torch.tensor(keep, device=hidden_states.device, dtype=torch.long)

    rotary_grid = full_video_grid_thw if full_video_grid_thw is not None else video_grid_thw
    rotary_pos_emb = visual.rot_pos_emb(rotary_grid).to(hidden_states.device)
    if raw_token_indices is not None:
        rotary_index = torch.tensor(raw_token_indices, device=hidden_states.device, dtype=torch.long)
        rotary_pos_emb = rotary_pos_emb.index_select(0, rotary_index)
    elif full_video_grid_thw is not None or raw_token_offset:
        start = int(raw_token_offset)
        rotary_pos_emb = rotary_pos_emb[start : start + seq_len]
    hidden_states = hidden_states.reshape(total_merged_tokens, spatial_merge_unit, -1)[keep_tensor]
    hidden_states = hidden_states.reshape(len(keep) * spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(total_merged_tokens, spatial_merge_unit, -1)[keep_tensor]
    rotary_pos_emb = rotary_pos_emb.reshape(len(keep) * spatial_merge_unit, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.tensor([0, hidden_states.shape[0]], device=hidden_states.device, dtype=torch.int32)

    for blk in visual.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    return visual.merger(hidden_states)


def _first_tensor_like(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor_like(item)
            except ValueError:
                continue
    raise ValueError("Qwen get_video_features did not return a tensor-like value")


def build_metric_skeleton(request: MllmRunRequest) -> dict[str, Any]:
    return {
        "latency_ms": {
            "model_load": None,
            "processor_load": None,
            "input_build": None,
            "qwen_vit_prepare": None,
            "generate": None,
            "total": None,
        },
        "tokens": {
            "prompt_tokens_estimated": len(request.prompt.split()),
            "input_ids_tokens": None,
            "visual_tokens_before_prune": None,
            "visual_tokens_after_prune": None,
            "llm_context_tokens": None,
        },
        "memory_bytes": {
            "peak_cuda_allocated": None,
            "peak_cuda_reserved": None,
        },
        "max_new_tokens": request.max_new_tokens,
        "qwen_vit": {
            "mode": request.qwen_vit_mode,
            "chunk_frames": request.qwen_vit_chunk_frames,
            "max_spatial_chunks": request.qwen_vit_max_spatial_chunks,
            "status": "not_started",
        },
        "qwen_thumbnail": qwen_thumbnail_summary(request),
        "processing_budget_summary": build_mllm_processing_budget_summary(request),
    }


def _estimate_qwen_visual_tokens(request: MllmRunRequest) -> int:
    frames = qwen_main_video_frame_count(request) + qwen_thumbnail_count(request)
    tiles = max(int(request.max_tiles_video), 1)
    # Qwen grid models expose the exact visual count through processor outputs at runtime.
    # For a model-load-free AutoGaze attachment PoC, 196 tokens/frame/tile is the same
    # coarse video token unit used by several pooled video MLLM paths and keeps estimates explicit.
    return frames * tiles * 196


def _estimate_selected_tokens(raw_visual_tokens: int, gazing_ratio: float | None) -> int | None:
    if gazing_ratio is None:
        return None
    ratio = max(0.0, min(1.0, float(gazing_ratio)))
    return max(1, int(round(raw_visual_tokens * ratio)))


def build_single_scale_dense_reference_budget(
    *,
    main_frames: int,
    thumbnail_frames: int,
    reference_tile_count: int,
    selected_visual_tokens: int | None,
) -> dict[str, Any]:
    tile_count = max(int(reference_tile_count), 1)
    patches_per_reference_tile = (
        SIGLIP_DENSE_REFERENCE_TILE_SIZE // SIGLIP_DENSE_REFERENCE_PATCH_SIZE
    ) ** 2
    main_tile_frames = main_frames * tile_count
    thumbnail_tile_frames = thumbnail_frames
    estimated_total_patch_tokens = (main_tile_frames + thumbnail_tile_frames) * patches_per_reference_tile
    estimated_llm_visual_tokens = (
        main_frames
        * math.ceil(
            tile_count * patches_per_reference_tile / SIGLIP_DENSE_REFERENCE_TOKEN_SHUFFLE
        )
        + thumbnail_frames
        * math.ceil(patches_per_reference_tile / SIGLIP_DENSE_REFERENCE_TOKEN_SHUFFLE)
    )
    return {
        "comparison_scope": "reference_only_not_model_exact",
        "reference_tile_size_px": SIGLIP_DENSE_REFERENCE_TILE_SIZE,
        "reference_patch_size": SIGLIP_DENSE_REFERENCE_PATCH_SIZE,
        "patch_positions_per_reference_tile_frame": patches_per_reference_tile,
        "reference_tile_count_per_frame": tile_count,
        "reference_tile_count_source": "runner_max_tiles_video",
        "estimated_main_tile_frames": main_tile_frames,
        "estimated_thumbnail_tile_frames": thumbnail_tile_frames,
        "estimated_total_patch_tokens": estimated_total_patch_tokens,
        "estimated_llm_visual_tokens_after_token_shuffle": estimated_llm_visual_tokens,
        "reference_token_shuffle": SIGLIP_DENSE_REFERENCE_TOKEN_SHUFFLE,
        "ratio_over_estimated_visual_tokens_after_prune": (
            estimated_total_patch_tokens / selected_visual_tokens if selected_visual_tokens else None
        ),
        "note": (
            "Reference-only dense SigLIP-style baseline using 392px tiles and patch size 14. "
            "It is exact only for NVILA/SigLIP tile-like pipelines; for Qwen/LongVILA/other adapters "
            "it is included to make cross-run reports comparable, not to describe the native processor."
        ),
    }


def build_mllm_processing_budget_summary(request: MllmRunRequest) -> dict[str, Any]:
    main_frames = qwen_main_video_frame_count(request)
    thumbnail = qwen_thumbnail_summary(request)
    thumbnail_frames = int(thumbnail["effective_frames"] or 0)
    total_frames = main_frames + thumbnail_frames
    estimated_visual = _estimate_qwen_visual_tokens(request) if request.model_family.startswith("qwen") else None
    selected_visual = _estimate_selected_tokens(estimated_visual, request.gazing_ratio) if estimated_visual is not None else None
    source_metadata = _safe_qwen_video_metadata(request.video)
    resize_effective = _safe_qwen_resize_effective(request, source_metadata)
    single_scale_dense = build_single_scale_dense_reference_budget(
        main_frames=main_frames,
        thumbnail_frames=thumbnail_frames,
        reference_tile_count=int(request.max_tiles_video or 1),
        selected_visual_tokens=selected_visual,
    )
    return {
        "runner": "flexible_runner",
        "video": {
            "source_resolution": _format_resolution(
                source_metadata.get("width") if source_metadata else None,
                source_metadata.get("height") if source_metadata else None,
            ),
            "source_width": source_metadata.get("width") if source_metadata else None,
            "source_height": source_metadata.get("height") if source_metadata else None,
            "processor_input_resolution": _format_resolution(
                resize_effective.get("width") if resize_effective else None,
                resize_effective.get("height") if resize_effective else None,
            ),
            "processor_input_width": resize_effective.get("width") if resize_effective else None,
            "processor_input_height": resize_effective.get("height") if resize_effective else None,
            "requested_video_frames": main_frames,
            "runner_resize": qwen_runner_resize_summary(request),
            "qwen_video_max_pixels": request.qwen_video_max_pixels,
            "qwen_video_min_pixels": request.qwen_video_min_pixels,
        },
        "model_processing_unit": {
            "name": "qwen_video_grid_thw" if request.model_family.startswith("qwen") else "adapter_specific_visual_tokens",
            "description": (
                "Qwen processor builds pixel_values_videos and video_grid_thw; exact patch/token counts are filled "
                "from processor outputs at runtime when available."
            ),
        },
        "tiling": {
            "max_tiles_video": request.max_tiles_video,
            "spatial_chunks_per_frame_limit": request.qwen_vit_max_spatial_chunks,
            "qwen_vit_mode": request.qwen_vit_mode,
            "chunk_frames": request.qwen_vit_chunk_frames,
            "note": (
                "For Qwen, spatial chunks split the processor grid for memory/latency measurement; they do not "
                "duplicate the video frames like NVILA spatial tiling."
            ),
        },
        "thumbnail": {
            "enabled": thumbnail_frames > 0,
            **thumbnail,
        },
        "single_scale_dense_vision_budget": single_scale_dense,
        "patch_budget_before_vit": {
            "estimated_main_frames_in_processor": main_frames,
            "estimated_thumbnail_frames_in_processor": thumbnail_frames,
            "estimated_total_frames_in_processor": total_frames,
            "estimated_visual_tokens_before_prune": estimated_visual,
            "estimated_visual_tokens_after_prune": selected_visual,
            "estimated_visual_token_reduction_ratio": (
                estimated_visual / selected_visual if estimated_visual is not None and selected_visual else None
            ),
            "multiscale_policy": "not_applicable_qwen_native_grid"
            if request.model_family.startswith("qwen")
            else "adapter_specific",
        },
    }


def _safe_qwen_video_metadata(video: str | None) -> dict[str, Any] | None:
    if not video:
        return None
    try:
        return read_video_metadata(video)
    except Exception:
        return None


def _safe_qwen_resize_effective(request: MllmRunRequest, source_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not source_metadata:
        return None
    width = source_metadata.get("width")
    height = source_metadata.get("height")
    if width is None or height is None:
        return None
    try:
        return dict(
            apply_resize_to_dimensions(
                width=int(width),
                height=int(height),
                shortest_edge=request.video_resize_shortest_edge,
                longest_edge=request.video_resize_longest_edge,
                exact_width=request.video_resize_width,
                exact_height=request.video_resize_height,
            )
        )
    except Exception:
        return None


def _format_resolution(width: Any, height: Any) -> str | None:
    if width is None or height is None:
        return None
    try:
        return f"{int(width)}x{int(height)}"
    except (TypeError, ValueError):
        return None


def build_feature_packing_probe(adapter_name: str, request: MllmRunRequest) -> dict[str, Any]:
    if adapter_name in {"qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen3-vl-moe", "qwen-grid"}:
        return _planned_probe(
            adapter_name=adapter_name,
            request=request,
            family_group="qwen_grid_vl",
            required_inputs=["pixel_values_videos", "video_grid_thw", "input_ids"],
            post_encoder_hook="after get_video_features output and before visual token insertion into MLLM context",
            pre_encoder_sparse_hook="before get_video_features; requires video_grid_thw and position-id preservation",
            required_runtime_checks=[
                "capture processor video_grid_thw for the sampled video",
                "capture get_video_features output shape",
                "capture visual token insertion indices in input_ids",
                "measure LLM prefill context before and after selected visual token pruning",
            ],
            token_accounting_targets=[
                "processor video patch/grid token count",
                "get_video_features output token count",
                "post-prune visual token count",
                "LLM total context token count",
            ],
            autogaze_applicability="plugin_on_off_experiment",
        )
    if adapter_name == "nvila-video":
        return _planned_probe(
            adapter_name=adapter_name,
            request=request,
            family_group="nvila_video",
            required_inputs=["input_ids", "video frames or processor pixel tensors", "attention_mask"],
            post_encoder_hook="after NVILA-Video vision output and before MLLM visual token packing",
            pre_encoder_sparse_hook="before NVILA-Video vision encoder; requires patch/position alignment probe",
            required_runtime_checks=[
                "load processor/model with trust_remote_code",
                "capture sampled video tensor or frame list contract",
                "capture NVILA-Video vision output feature shape",
                "capture visual token packing boundary before LLM prefill",
                "verify AutoGaze fields are not injected for paper baseline runs",
            ],
            token_accounting_targets=[
                "processor visual input token count",
                "vision feature token count",
                "projected MLLM visual token count",
                "LLM total context token count",
            ],
            autogaze_applicability=(
                "not_applicable_for_paper_baseline"
                if request.model_family == "nvila-video-baseline"
                else "plugin_on_off_experiment"
            ),
            next_probe_command=_vila_feature_probe_command(adapter_name, request),
        )
    if adapter_name == "longvila":
        return _planned_probe(
            adapter_name=adapter_name,
            request=request,
            family_group="longvila",
            required_inputs=["input_ids", "video frames or processor pixel tensors", "attention_mask"],
            post_encoder_hook="after visual feature extraction and before MLLM packing",
            pre_encoder_sparse_hook="before LongVILA vision tower; requires processor and position semantics probe",
            required_runtime_checks=[
                "load processor/model with trust_remote_code",
                "capture sampled video tensor or frame list contract",
                "capture vision tower output feature shape",
                "capture visual token packing boundary before LLM prefill",
            ],
            token_accounting_targets=[
                "LongVILA processor visual token count",
                "vision tower feature token count",
                "long-context packed visual token count",
                "LLM prefill context token count",
            ],
            autogaze_applicability="plugin_on_off_experiment",
            next_probe_command=_vila_feature_probe_command(adapter_name, request),
        )
    if adapter_name == "internvl3":
        return _planned_probe(
            adapter_name=adapter_name,
            request=request,
            family_group="internvl",
            required_inputs=["pixel_values", "num_patches_list"],
            post_encoder_hook="after dynamic visual feature extraction and before language model packing",
            pre_encoder_sparse_hook="before dynamic tile vision encoder; requires num_patches_list remapping probe",
            required_runtime_checks=[
                "load processor/model with trust_remote_code",
                "capture dynamic tile tensor shape",
                "capture num_patches_list per frame/image",
                "capture vision feature shape after dynamic visual encoder",
                "capture language packing boundary before LLM prefill",
            ],
            token_accounting_targets=[
                "dynamic tile count",
                "vision feature token count",
                "packed visual token count",
                "LLM prefill context token count",
            ],
            autogaze_applicability="plugin_on_off_experiment",
        )
    return _planned_probe(
        adapter_name=adapter_name,
        request=request,
        family_group=adapter_name,
        required_inputs=[],
        post_encoder_hook="unknown until model-specific probe is added",
        pre_encoder_sparse_hook="unknown until model-specific probe is added",
        required_runtime_checks=[
            "load processor/model with trust_remote_code",
            "capture processor input contract",
            "capture vision output and language packing boundary",
        ],
        token_accounting_targets=[
            "processor visual token count",
            "vision feature token count",
            "LLM context token count",
        ],
        autogaze_applicability="plugin_on_off_experiment",
    )


def _planned_probe(
    *,
    adapter_name: str,
    request: MllmRunRequest,
    family_group: str,
    required_inputs: list[str],
    post_encoder_hook: str,
    pre_encoder_sparse_hook: str,
    required_runtime_checks: list[str],
    token_accounting_targets: list[str],
    autogaze_applicability: str,
    next_probe_command: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "adapter": adapter_name,
        "status": "probe_required",
        "family_group": family_group,
        "model_family": request.model_family,
        "model_path": request.model_path,
        "required_inputs": required_inputs,
        "post_encoder_hook": post_encoder_hook,
        "pre_encoder_sparse_hook": pre_encoder_sparse_hook,
        "recommended_first_integration_level": "post_encoder_token_prune",
        "required_runtime_checks": required_runtime_checks,
        "token_accounting_targets": token_accounting_targets,
        "autogaze_applicability": autogaze_applicability,
        "next_probe_command": next_probe_command,
        "loads_model": False,
        "notes": [
            "This planned adapter does not run generation yet.",
            "Use it to make the next CUDA probe explicit before wiring AutoGaze into the model-specific processor path.",
        ],
    }


def _vila_feature_probe_command(adapter_name: str, request: MllmRunRequest) -> dict[str, Any]:
    return {
        "goal": "capture_vila_feature_packing",
        "adapter": adapter_name,
        "requires_code_probe": True,
        "suggested_entrypoint": "repro.flexible_runner",
        "model_path": request.model_path,
        "video": request.video,
        "num_video_frames": request.num_video_frames,
        "max_tiles_video": request.max_tiles_video,
        "expected_outputs": [
            "processor video tensor/frame contract",
            "vision feature shape",
            "projector output shape",
            "LLM visual token insertion boundary",
        ],
    }


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ASSISTANT_RE = re.compile(r"^\s*(?:assistant|assistant\s+answer)\s*:\s*(.+)$", re.IGNORECASE)


def extract_assistant_text(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        cleaned = _ANSI_RE.sub("", line).strip()
        if not cleaned:
            continue
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "text", "response", "prediction"):
                value = payload.get(key)
                if value is not None:
                    return str(value).strip()
        match = _ASSISTANT_RE.match(cleaned)
        if match:
            return match.group(1).strip()
    return _last_nonempty_line(text)


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        cleaned = _ANSI_RE.sub("", line).strip()
        if cleaned:
            return cleaned
    return None


def _tail_lines(text: str, *, max_lines: int = 20) -> list[str]:
    lines = [_ANSI_RE.sub("", line).rstrip() for line in text.splitlines()]
    return lines[-max_lines:]


def _split_external_command(command: str) -> list[str]:
    parts = shlex.split(command)
    return parts or [command]


def _command_available(command: str) -> bool:
    if "/" in command:
        return True
    return shutil.which(command) is not None


def _record_input_token_metrics(metrics: dict[str, Any], inputs: Any) -> None:
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    if input_ids is not None and hasattr(input_ids, "shape"):
        metrics["tokens"]["input_ids_tokens"] = int(input_ids.shape[-1])
        metrics["tokens"]["llm_context_tokens"] = int(input_ids.shape[-1])
    for key in ("video_grid_thw", "image_grid_thw"):
        value = inputs.get(key) if isinstance(inputs, dict) else getattr(inputs, key, None)
        if value is not None:
            metrics["tokens"][key] = _shape_list(value)
            if key == "video_grid_thw":
                _record_qwen_actual_grid_budget(metrics, value)


def _record_qwen_actual_grid_budget(metrics: dict[str, Any], video_grid_thw: Any) -> None:
    summary = metrics.get("processing_budget_summary")
    if not isinstance(summary, dict):
        return
    patch_budget = summary.get("patch_budget_before_vit")
    if not isinstance(patch_budget, dict):
        return
    try:
        t, h, w = _single_qwen_grid(video_grid_thw)
    except Exception:
        return
    patch_budget["actual_video_grid_thw"] = [t, h, w]
    patch_budget["actual_raw_patch_tokens_before_vit"] = t * h * w


def _record_cuda_memory(metrics: dict[str, Any]) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if not torch.cuda.is_available():
        return
    metrics["memory_bytes"]["peak_cuda_allocated"] = int(torch.cuda.max_memory_allocated())
    metrics["memory_bytes"]["peak_cuda_reserved"] = int(torch.cuda.max_memory_reserved())


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]

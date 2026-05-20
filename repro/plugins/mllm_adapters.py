from __future__ import annotations

from dataclasses import dataclass
import re
import json
import shlex
import shutil
import subprocess
import time
from typing import Any


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
    num_video_frames: int = 128
    max_tiles_video: int = 1
    external_mllm_command: str = "vila-infer"


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
                status="probe_required",
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
            content.append({"type": "video", "video": request.video})
        if request.image:
            content.append({"type": "image", "image": request.image})
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def describe_runtime(self, request: MllmRunRequest) -> dict[str, Any]:
        description = super().describe_runtime(request)
        if request.token_selector_kind == "autogaze" or request.integration_level != "none":
            description["feature_packing_probe"] = build_feature_packing_probe(self.name, request)
        return description

    def run(self, request: MllmRunRequest) -> MllmRunResult:
        if request.token_selector_kind == "autogaze" or request.integration_level != "none":
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


def _build_qwen_grid_inputs(processor: Any, messages: list[dict[str, Any]], request: MllmRunRequest) -> Any:
    if request.video:
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("qwen_vl_utils is required for Qwen-style video inputs") from exc
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        return processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
            **video_kwargs,
        )
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def build_metric_skeleton(request: MllmRunRequest) -> dict[str, Any]:
    return {
        "latency_ms": {
            "model_load": None,
            "processor_load": None,
            "input_build": None,
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
    }


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
        "loads_model": False,
        "notes": [
            "This planned adapter does not run generation yet.",
            "Use it to make the next CUDA probe explicit before wiring AutoGaze into the model-specific processor path.",
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

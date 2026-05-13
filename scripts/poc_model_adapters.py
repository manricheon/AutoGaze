#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AdapterStatus:
    name: str
    status: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class VisionEncoderAdapter:
    name = "generic_vit"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.status = AdapterStatus(self.name, "not_loaded")
        self.model: Any = None
        self.device = "cpu"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(self.name, "stub", "real vision loading disabled")
            return self.status
        self.device = device
        module_path = self.config.get("module_path") or self.default_module_path()
        class_name = self.config.get("class_name") or self.config.get("class_or_factory") or self.default_class_name()
        checkpoint = self._checkpoint()
        if not module_path or not class_name:
            self.status = AdapterStatus(self.name, "blocked", "module_path and class_name are required for real loading")
            return self.status
        if not checkpoint:
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint_path or model_id is required for real loading")
            return self.status
        if _looks_like_local_path(str(checkpoint)):
            local_checkpoint = _resolve_local_path(checkpoint)
            if not local_checkpoint.exists():
                self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
                return self.status
            missing_shards = _missing_sharded_checkpoint_files(local_checkpoint)
            if missing_shards:
                self.status = AdapterStatus(
                    self.name,
                    "blocked",
                    f"{self.name} checkpoint is incomplete; missing shard files: {', '.join(missing_shards[:8])}",
                    metadata={"checkpoint": checkpoint, "resolved_checkpoint": str(local_checkpoint), "missing_shards": missing_shards},
                )
                return self.status
        checkpoint_ref = _model_reference_for_loading(checkpoint)
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            self.model = factory.from_pretrained(checkpoint_ref, **load_kwargs) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
                    "resolved_checkpoint": checkpoint_ref,
                    "module_path": module_path,
                    "class_name": class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"real vision loading failed: {exc}")
        return self.status

    def preprocess_or_accept_features(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"video": video, "autogaze": autogaze}

    def forward(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        if self.status.status == "blocked":
            raise RuntimeError(self.status.reason or f"{self.name} is blocked")
        if self.model is not None:
            return self._forward_real(video, autogaze=autogaze)
        batch, frames, _channels, height, width = [int(dim) for dim in video.shape]
        grid_h, grid_w = self.patch_grid((height, width))
        token_count = frames * grid_h * grid_w
        output_dim = self.output_dim()
        tokens = torch.zeros((batch, token_count, output_dim), dtype=video.dtype)
        return {
            "status": self.status.status,
            "visual_tokens": tokens,
            "metadata": {
                "adapter": self.name,
                "stub_status": self.status.status,
                "reason": self.status.reason,
                "autogaze_tokens_used": bool(autogaze is not None and self.supports_autogaze_tokens()),
            },
        }

    def _forward_real(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        model_input = video.to(self.device) if hasattr(video, "to") else video
        kwargs = {}
        if autogaze is not None and self.supports_autogaze_tokens():
            kwargs["gazing_info"] = _move_tensor_mapping(autogaze, self.device)
        errors: list[str] = []
        batch_frames: tuple[int, int] | None = None
        if (
            isinstance(model_input, torch.Tensor)
            and model_input.ndim == 5
            and self.config.get("video_input_mode") == "flatten_frames"
        ):
            batch, frames, channels, height, width = [int(dim) for dim in model_input.shape]
            model_candidates = [model_input.reshape(batch * frames, channels, height, width), model_input]
            batch_frames = (batch, frames)
        else:
            model_candidates = [model_input]
        with torch.inference_mode():
            for candidate in model_candidates:
                for call in (
                    lambda candidate=candidate: self.model(pixel_values=candidate, **kwargs),
                    lambda candidate=candidate: self.model(videos=candidate, **kwargs),
                    lambda candidate=candidate: self.model(candidate, **kwargs),
                ):
                    try:
                        output = call()
                        break
                    except TypeError as exc:
                        errors.append(str(exc))
                else:
                    continue
                break
            else:
                raise RuntimeError(f"{self.name} real forward failed for supported call signatures: {errors}")
        tokens = getattr(output, "last_hidden_state", output)
        if isinstance(tokens, Mapping):
            tokens = tokens.get("last_hidden_state") or tokens.get("pooler_output") or tokens.get("hidden_states")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"{self.name} real forward did not return a tensor-like visual output")
        if batch_frames is not None and tokens.ndim >= 3 and int(tokens.shape[0]) == batch_frames[0] * batch_frames[1]:
            batch, frames = batch_frames
            tokens = tokens.reshape(batch, frames * int(tokens.shape[1]), *tokens.shape[2:])
        return {
            "status": "real",
            "visual_tokens": tokens,
            "metadata": {
                "adapter": self.name,
                "status": self.status.to_dict(),
                "autogaze_tokens_used": bool(autogaze is not None and self.supports_autogaze_tokens()),
            },
        }

    def output_dim(self) -> int:
        return int(self.config.get("output_dim", 768))

    def patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        patch_size = int(self.config.get("patch_size", 16))
        if resolution is None:
            resolution = (int(self.config.get("resolution", 224)), int(self.config.get("resolution", 224)))
        return max(1, int(resolution[0]) // patch_size), max(1, int(resolution[1]) // patch_size)

    def supports_autogaze_tokens(self) -> bool:
        return False

    def supports_chop_mode(self) -> bool:
        return False

    def default_module_path(self) -> str | None:
        return None

    def default_class_name(self) -> str | None:
        return None

    def _checkpoint(self) -> Any:
        return self.config.get("checkpoint_path") or self.config.get("model_id")


class ModifiedSiglipAdapter(VisionEncoderAdapter):
    name = "modified_siglip"

    def supports_autogaze_tokens(self) -> bool:
        return True

    def supports_chop_mode(self) -> bool:
        return True


class VanillaSiglipAdapter(VisionEncoderAdapter):
    name = "vanilla_siglip"

    def supports_autogaze_tokens(self) -> bool:
        return bool(self.config.get("experimental_autogaze_adapter", False))


class VJEPA2Adapter(VisionEncoderAdapter):
    name = "vjepa2"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.processor: Any = None

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModel"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory")
        if not processor_class and not self.config.get("processor_path"):
            status.metadata["processor_status"] = "not_configured_tensor_input"
            return status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"real V-JEPA2 processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = processor_class or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            self.processor = factory.from_pretrained(processor_ref, **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
        except Exception as exc:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"real V-JEPA2 processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def preprocess_or_accept_features(self, video: torch.Tensor, autogaze: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        return {
            "video": video,
            "autogaze": None,
            "metadata": {
                "semantics": "[B,T,C,H,W]",
                "note": "AutoGaze patch indices are not assumed to map directly to V-JEPA2 tokens.",
            },
        }


class GenericVitAdapter(VisionEncoderAdapter):
    name = "generic_vit"


class MLLMAdapter:
    name = "generic_mllm"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.status = AdapterStatus(self.name, "not_loaded")
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        if not allow_real_model_loading:
            self.status = AdapterStatus(self.name, "stub", "real MLLM loading disabled")
            return self.status
        self.device = device
        module_path = self.config.get("module_path") or self.default_module_path()
        class_name = self.config.get("class_name") or self.config.get("class_or_factory") or self.default_class_name()
        checkpoint = self._checkpoint()
        if not module_path or not class_name:
            self.status = AdapterStatus(self.name, "blocked", "module_path and class_name are required for real loading")
            return self.status
        if not checkpoint:
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint_path or model_id is required for real loading")
            return self.status
        if _looks_like_local_path(str(checkpoint)):
            local_checkpoint = _resolve_local_path(checkpoint)
            if not local_checkpoint.exists():
                self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
                return self.status
            missing_shards = _missing_sharded_checkpoint_files(local_checkpoint)
            if missing_shards:
                self.status = AdapterStatus(
                    self.name,
                    "blocked",
                    f"{self.name} checkpoint is incomplete; missing shard files: {', '.join(missing_shards[:8])}",
                    metadata={"checkpoint": checkpoint, "resolved_checkpoint": str(local_checkpoint), "missing_shards": missing_shards},
                )
                return self.status
        checkpoint_ref = _model_reference_for_loading(checkpoint)
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            self.model = factory.from_pretrained(checkpoint_ref, **load_kwargs) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to") and "device_map" not in load_kwargs:
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
                    "resolved_checkpoint": checkpoint_ref,
                    "module_path": module_path,
                    "class_name": class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"real MLLM loading failed: {exc}")
        return self.status

    def prepare_inputs(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        return {"query_text": query_text, "video": video, "visual_tokens": visual_tokens, "video_path": video_path}

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required"}
        if self.model is None:
            return {
                "status": self.status.status,
                "answer": None,
                "reason": self.status.reason or "MLLM generation unavailable",
                "query_text_used": True,
            }
        return {
            "status": "blocked",
            "answer": None,
            "reason": f"real generation is not implemented for {self.name}; add a model-specific generation adapter",
            "query_text_used": True,
        }

    def count_visual_tokens(self, visual_tokens: torch.Tensor | None) -> int | None:
        if visual_tokens is None:
            return None
        if visual_tokens.ndim < 2:
            return None
        return int(visual_tokens.shape[-2])

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_official_processor_path(self) -> bool:
        return False

    def default_module_path(self) -> str | None:
        return None

    def default_class_name(self) -> str | None:
        return None

    def _checkpoint(self) -> Any:
        return self.config.get("checkpoint_path") or self.config.get("model_id")


class NVILAAdapter(MLLMAdapter):
    name = "nvila"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        if not processor_path:
            self.status = AdapterStatus(self.name, "blocked", "NVILA processor_path or checkpoint_path is required for official processor loading")
            return self.status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"NVILA official processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            processor_kwargs = _nvila_processor_from_pretrained_kwargs(self.config)
            self.processor = factory.from_pretrained(processor_ref, **processor_kwargs)
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
            status.metadata["official_processor_path"] = True
            status.metadata["processor_autogaze_controls"] = _jsonable_processor_kwargs(processor_kwargs)
            self.status = status
        except Exception as exc:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"NVILA official processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModel"

    def supports_direct_visual_tokens(self) -> bool:
        return False

    def supports_official_processor_path(self) -> bool:
        return True

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        if visual_tokens is not None:
            return {
                "status": "blocked",
                "answer": None,
                "reason": "NVILA direct visual token injection is not verified in this PoC; use official processor video input.",
                "query_text_used": True,
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_official_processor(
                query_text=query_text,
                video=video,
                video_path=video_path,
                max_new_tokens=max_new_tokens,
            )
        return super().generate(
            query_text=query_text,
            video=video,
            visual_tokens=visual_tokens,
            max_new_tokens=max_new_tokens,
            video_path=video_path,
        )

    def _generate_with_official_processor(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        video_path: str | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required", "query_text_used": False}
        try:
            prompt_template = str(self.config.get("prompt_template") or "{video_token}\n\n{prompt}")
            video_token = getattr(getattr(self.processor, "tokenizer", None), "video_token", "<video>")
            prompt = prompt_template.format(prompt=query_text, query=query_text, video_token=video_token)
            target_frame_count = getattr(self.processor, "num_video_frames", None)
            video_input, video_input_kind = _official_video_input(
                video=video,
                video_path=video_path,
                target_frame_count=target_frame_count,
            )
            inputs = self.processor(text=prompt, videos=video_input, return_tensors="pt")
            if isinstance(inputs, Mapping):
                model_device = getattr(self.model, "device", self.device)
                inputs = {
                    key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            decode_input = outputs[:, input_ids.shape[1] :] if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor) else outputs
            decoded = self.processor.batch_decode(decode_input, skip_special_tokens=True)
            answer = str(decoded[0]).strip() if decoded else ""
            return {
                "status": "real",
                "answer": answer,
                "reason": None,
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": {
                    "prompt_template": prompt_template,
                    "video_input_kind": video_input_kind,
                    "max_new_tokens": max_new_tokens,
                    "autogaze_visualizer_status": self.config.get("poc_autogaze_status"),
                    "autogaze_enabled_for_processor": self.config.get("poc_autogaze_enabled"),
                    "processor_autogaze_controls": _jsonable_processor_kwargs(_nvila_processor_from_pretrained_kwargs(self.config)),
                    "autogaze_visual_tokens_injected": False,
                    "note": (
                        "NVILA generation uses the official processor video path. "
                        "The separate PoC AutoGaze stage is retained for visualization and metrics; "
                        "direct visual-token injection into NVILA is not claimed."
                    ),
                },
            }
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"NVILA official processor generation failed: {exc}",
                "query_text_used": True,
                "official_processor_path": True,
            }


class QwenAdapter(MLLMAdapter):
    name = "qwen"

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModelForVision2Seq"

    def load(self, *, allow_real_model_loading: bool = False, device: str = "cpu", dtype: str = "float32") -> AdapterStatus:
        status = super().load(allow_real_model_loading=allow_real_model_loading, device=device, dtype=dtype)
        if status.status != "real":
            return status
        processor_path = self.config.get("processor_path") or self._checkpoint()
        if not processor_path:
            self.status = AdapterStatus(self.name, "blocked", "Qwen processor_path or model_id is required for official processor loading")
            return self.status
        if _looks_like_local_path(str(processor_path)) and not _resolve_local_path(processor_path).exists():
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"Qwen official processor path does not exist: {processor_path}",
                metadata=status.metadata,
            )
            return self.status
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            processor_ref = _model_reference_for_loading(processor_path)
            self.processor = factory.from_pretrained(processor_ref, **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
            status.metadata["resolved_processor_path"] = processor_ref
            status.metadata["processor_module_path"] = processor_module
            status.metadata["processor_class_name"] = processor_class
            status.metadata["official_processor_path"] = True
        except Exception as exc:
            self.status = AdapterStatus(
                self.name,
                "blocked",
                f"Qwen official processor loading failed: {exc}",
                metadata=status.metadata,
            )
            return self.status
        return status

    def supports_official_processor_path(self) -> bool:
        return True

    def generate(
        self,
        *,
        query_text: str,
        video: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        if visual_tokens is not None and not self.supports_direct_visual_tokens():
            return {
                "status": "blocked",
                "answer": None,
                "reason": "Qwen direct visual token injection is unsupported; use official processor/input-level selection.",
                "query_text_used": True,
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_official_processor(query_text=query_text, video=video, max_new_tokens=max_new_tokens)
        return super().generate(query_text=query_text, video=video, visual_tokens=visual_tokens, max_new_tokens=max_new_tokens, video_path=video_path)

    def _generate_with_official_processor(self, *, query_text: str, video: torch.Tensor, max_new_tokens: int) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required", "query_text_used": False}
        try:
            prompt_template = str(self.config.get("prompt_template") or "{prompt}")
            video_token = getattr(getattr(self.processor, "tokenizer", None), "video_token", "<video>")
            prompt = prompt_template.format(prompt=query_text, query=query_text, video_token=video_token)
            frames = _video_tensor_to_pil(video)
            processor_errors: list[str] = []
            for call in (
                lambda: self.processor(text=prompt, videos=frames, return_tensors="pt"),
                lambda: self.processor(text=[prompt], videos=frames, return_tensors="pt"),
                lambda: self.processor(text=prompt, images=frames, return_tensors="pt"),
                lambda: self.processor(text=[prompt], images=frames, return_tensors="pt"),
            ):
                try:
                    inputs = call()
                    break
                except TypeError as exc:
                    processor_errors.append(str(exc))
            else:
                return {
                    "status": "blocked",
                    "answer": None,
                    "reason": f"Qwen official processor did not accept supported video/image signatures: {processor_errors}",
                    "query_text_used": True,
                    "official_processor_path": True,
                }
            if isinstance(inputs, Mapping):
                model_device = getattr(self.model, "device", self.device)
                inputs = {
                    key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            decode_input = outputs[:, input_ids.shape[1] :] if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor) else outputs
            decoded = self.processor.batch_decode(decode_input, skip_special_tokens=True)
            answer = str(decoded[0]).strip() if decoded else ""
            return {
                "status": "real",
                "answer": answer,
                "reason": None,
                "query_text_used": True,
                "official_processor_path": True,
                "metadata": {
                    "prompt_template": prompt_template,
                    "num_video_frames": len(frames),
                    "max_new_tokens": max_new_tokens,
                },
            }
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            return {
                "status": "blocked",
                "answer": None,
                "reason": f"Qwen official processor generation failed: {exc}",
                "query_text_used": True,
                "official_processor_path": True,
            }


class GenericMLLMAdapter(MLLMAdapter):
    name = "generic_mllm"


def _looks_like_local_path(value: str) -> bool:
    if value.startswith((".", "/", "~")):
        return True
    return value.startswith("weights/") or value.startswith("checkpoints/") or value.endswith(".pt") or value.endswith(".safetensors")


def _resolve_local_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _model_reference_for_loading(value: Any) -> str:
    text = str(value)
    if _looks_like_local_path(text):
        return str(_resolve_local_path(text))
    return text


def _missing_sharded_checkpoint_files(path: Path) -> list[str]:
    index_path = path / "model.safetensors.index.json" if path.is_dir() else None
    if index_path is None or not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return []
    expected = sorted({str(name) for name in weight_map.values()})
    return [name for name in expected if not (path / name).exists()]


def _from_pretrained_kwargs(config: Mapping[str, Any], *, dtype: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "trust_remote_code" in config:
        kwargs["trust_remote_code"] = bool(config.get("trust_remote_code"))
    if "local_files_only" in config:
        kwargs["local_files_only"] = bool(config.get("local_files_only"))
    model_dtype = _torch_dtype(dtype)
    if model_dtype is not None:
        kwargs["dtype"] = model_dtype
    for key, value in dict(config.get("from_pretrained_kwargs") or {}).items():
        if key == "torch_dtype":
            key = "dtype"
            value = _dtype_from_config_value(value)
        kwargs[key] = value
    return kwargs


def _processor_from_pretrained_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "trust_remote_code" in config:
        kwargs["trust_remote_code"] = bool(config.get("trust_remote_code"))
    if "local_files_only" in config:
        kwargs["local_files_only"] = bool(config.get("local_files_only"))
    for key, value in dict(config.get("processor_from_pretrained_kwargs") or {}).items():
        kwargs[key] = value
    if "use_fast" not in kwargs:
        kwargs["use_fast"] = False
    return kwargs


def _nvila_processor_from_pretrained_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = _processor_from_pretrained_kwargs(config)
    autogaze_model_id = (
        kwargs.get("autogaze_model_id")
        or config.get("autogaze_model_id")
        or config.get("poc_autogaze_processor_path")
        or config.get("poc_autogaze_checkpoint_path")
    )
    if autogaze_model_id is None and _resolve_local_path("weights/AutoGaze").exists():
        autogaze_model_id = "weights/AutoGaze"
    if autogaze_model_id is not None:
        kwargs["autogaze_model_id"] = _model_reference_for_loading(autogaze_model_id)
    max_num_frames = _local_autogaze_max_num_frames(autogaze_model_id)
    if max_num_frames is not None:
        _set_or_validate_nvila_frame_counts(kwargs, max_num_frames=max_num_frames)
    if not bool(config.get("sync_autogaze_controls_from_config", False)):
        return kwargs

    autogaze_enabled = config.get("poc_autogaze_enabled")
    if autogaze_enabled is True:
        gaze_ratio = config.get("poc_gaze_ratio")
        task_loss_requirement = config.get("poc_task_loss_requirement")
        if gaze_ratio is not None:
            kwargs.setdefault("gazing_ratio_tile", gaze_ratio)
            kwargs.setdefault("gazing_ratio_thumbnail", gaze_ratio)
        if task_loss_requirement is not None:
            kwargs.setdefault("task_loss_requirement_tile", task_loss_requirement)
            kwargs.setdefault("task_loss_requirement_thumbnail", task_loss_requirement)
    elif autogaze_enabled is False:
        kwargs.setdefault("gazing_ratio_tile", None)
        kwargs.setdefault("gazing_ratio_thumbnail", None)
        kwargs.setdefault("task_loss_requirement_tile", None)
        kwargs.setdefault("task_loss_requirement_thumbnail", None)
    return kwargs


def _local_autogaze_max_num_frames(autogaze_model_id: Any) -> int | None:
    if autogaze_model_id in {None, ""}:
        return None
    text = str(autogaze_model_id)
    if not _looks_like_local_path(text):
        return None
    path = _resolve_local_path(text)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("max_num_frames")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _set_or_validate_nvila_frame_counts(kwargs: dict[str, Any], *, max_num_frames: int) -> None:
    value = kwargs.get("num_video_frames")
    if value is None:
        kwargs["num_video_frames"] = max_num_frames
    else:
        num_video_frames = int(value)
        if num_video_frames <= 0 or num_video_frames % max_num_frames != 0:
            raise ValueError(
                "NVILA processor num_video_frames must be a positive multiple of "
                f"AutoGaze max_num_frames ({max_num_frames}); got {num_video_frames}"
            )
        kwargs["num_video_frames"] = num_video_frames

    thumbnail_value = kwargs.get("num_video_frames_thumbnail")
    if thumbnail_value is None:
        kwargs["num_video_frames_thumbnail"] = max_num_frames
    else:
        num_thumbnail_frames = int(thumbnail_value)
        if num_thumbnail_frames <= 0:
            raise ValueError("NVILA processor num_video_frames_thumbnail must be positive")
        kwargs["num_video_frames_thumbnail"] = num_thumbnail_frames


def _jsonable_processor_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.dtype):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _torch_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _dtype_from_config_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return value
    text = str(value)
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    parsed = _torch_dtype(text)
    return parsed if parsed is not None else value


def _official_video_input(
    *,
    video: torch.Tensor,
    video_path: str | None,
    target_frame_count: int | None = None,
) -> tuple[Any, str]:
    if video_path and video_path != "dummy":
        path = Path(video_path).expanduser()
        if path.exists():
            return str(path if path.is_absolute() else path.resolve()), "video_path"
        repo_path = REPO_ROOT / path
        if repo_path.exists():
            return str(repo_path), "video_path"
        return video_path, "video_reference"
    frames = _video_tensor_to_pil(video, target_frame_count=target_frame_count)
    suffix = f"_to_{target_frame_count}" if target_frame_count is not None else ""
    return frames, f"processed_tensor_pil_frames{suffix}"


def _move_tensor_mapping(value: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, item in value.items():
        moved[key] = item.to(device) if isinstance(item, torch.Tensor) else item
    return moved


def _video_tensor_to_pil(video: torch.Tensor, *, target_frame_count: int | None = None) -> list[Any]:
    from PIL import Image

    if video.ndim != 5:
        raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
    frames = video[0].detach().cpu().float().clamp(0, 1)
    if target_frame_count is not None:
        target = int(target_frame_count)
        if target <= 0:
            raise ValueError("target_frame_count must be positive")
        if int(frames.shape[0]) > target:
            if target == 1:
                indices = torch.tensor([0], dtype=torch.long)
            else:
                indices = torch.linspace(0, int(frames.shape[0]) - 1, steps=target).round().to(torch.long)
            frames = frames.index_select(0, indices)
        elif int(frames.shape[0]) < target:
            if int(frames.shape[0]) == 0:
                raise ValueError("cannot pad an empty video tensor")
            pad = frames[-1:].expand(target - int(frames.shape[0]), -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)
    images = []
    for frame in frames:
        array = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(array, mode="RGB"))
    return images

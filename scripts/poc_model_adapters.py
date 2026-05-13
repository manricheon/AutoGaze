#!/usr/bin/env python3
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch


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
        if _looks_like_local_path(str(checkpoint)) and not Path(str(checkpoint)).expanduser().exists():
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
            return self.status
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            load_kwargs = _from_pretrained_kwargs(self.config, dtype=dtype)
            self.model = factory.from_pretrained(str(checkpoint), **load_kwargs) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
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
            kwargs["gazing_info"] = autogaze
        errors: list[str] = []
        with torch.inference_mode():
            for call in (
                lambda: self.model(pixel_values=model_input, **kwargs),
                lambda: self.model(videos=model_input, **kwargs),
                lambda: self.model(model_input, **kwargs),
            ):
                try:
                    output = call()
                    break
                except TypeError as exc:
                    errors.append(str(exc))
            else:
                raise RuntimeError(f"{self.name} real forward failed for supported call signatures: {errors}")
        tokens = getattr(output, "last_hidden_state", output)
        if isinstance(tokens, Mapping):
            tokens = tokens.get("last_hidden_state") or tokens.get("pooler_output") or tokens.get("hidden_states")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"{self.name} real forward did not return a tensor-like visual output")
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
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = processor_class or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            self.processor = factory.from_pretrained(str(processor_path), **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
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
        if _looks_like_local_path(str(checkpoint)) and not Path(str(checkpoint)).expanduser().exists():
            self.status = AdapterStatus(self.name, "blocked", f"{self.name} checkpoint path does not exist: {checkpoint}")
            return self.status
        try:
            factory = getattr(importlib.import_module(str(module_path)), str(class_name))
            self.model = factory.from_pretrained(str(checkpoint), **_from_pretrained_kwargs(self.config, dtype=dtype)) if hasattr(factory, "from_pretrained") else factory()
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.status = AdapterStatus(
                self.name,
                "real",
                metadata={
                    "checkpoint": checkpoint,
                    "module_path": module_path,
                    "class_name": class_name,
                    "local_files_only": bool(self.config.get("local_files_only", False)),
                    "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
                },
            )
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            self.status = AdapterStatus(self.name, "blocked", f"real MLLM loading failed: {exc}")
        return self.status

    def prepare_inputs(self, *, query_text: str, video: torch.Tensor, visual_tokens: torch.Tensor | None = None) -> dict[str, Any]:
        return {"query_text": query_text, "video": video, "visual_tokens": visual_tokens}

    def generate(self, *, query_text: str, video: torch.Tensor, visual_tokens: torch.Tensor | None = None, max_new_tokens: int = 32) -> dict[str, Any]:
        if not query_text:
            return {"status": "blocked", "answer": None, "reason": "query text is required"}
        if self.model is None:
            return {
                "status": self.status.status,
                "answer": None,
                "reason": self.status.reason or "MLLM generation unavailable",
                "query_text_used": True,
            }
        raise NotImplementedError("Real generic MLLM generation requires a model-specific adapter")

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

    def default_module_path(self) -> str | None:
        return "transformers"

    def default_class_name(self) -> str | None:
        return "AutoModel"

    def supports_direct_visual_tokens(self) -> bool:
        return True

    def supports_official_processor_path(self) -> bool:
        return True


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
        processor_module = self.config.get("processor_module_path") or "transformers"
        processor_class = self.config.get("processor_class_name") or self.config.get("processor_class_or_factory") or "AutoProcessor"
        try:
            factory = getattr(importlib.import_module(str(processor_module)), str(processor_class))
            self.processor = factory.from_pretrained(str(processor_path), **_processor_from_pretrained_kwargs(self.config))
            status.metadata["processor_status"] = "real"
            status.metadata["processor_path"] = processor_path
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

    def generate(self, *, query_text: str, video: torch.Tensor, visual_tokens: torch.Tensor | None = None, max_new_tokens: int = 32) -> dict[str, Any]:
        if visual_tokens is not None and not self.supports_direct_visual_tokens():
            return {
                "status": "blocked",
                "answer": None,
                "reason": "Qwen direct visual token injection is unsupported; use official processor/input-level selection.",
                "query_text_used": True,
            }
        if self.model is not None and self.processor is not None:
            return self._generate_with_official_processor(query_text=query_text, video=video, max_new_tokens=max_new_tokens)
        return super().generate(query_text=query_text, video=video, visual_tokens=visual_tokens, max_new_tokens=max_new_tokens)

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


def _from_pretrained_kwargs(config: Mapping[str, Any], *, dtype: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "trust_remote_code" in config:
        kwargs["trust_remote_code"] = bool(config.get("trust_remote_code"))
    if "local_files_only" in config:
        kwargs["local_files_only"] = bool(config.get("local_files_only"))
    torch_dtype = _torch_dtype(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    for key, value in dict(config.get("from_pretrained_kwargs") or {}).items():
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
    return kwargs


def _torch_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _video_tensor_to_pil(video: torch.Tensor) -> list[Any]:
    from PIL import Image

    if video.ndim != 5:
        raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
    frames = video[0].detach().cpu().float().clamp(0, 1)
    images = []
    for frame in frames:
        array = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
        images.append(Image.fromarray(array, mode="RGB"))
    return images

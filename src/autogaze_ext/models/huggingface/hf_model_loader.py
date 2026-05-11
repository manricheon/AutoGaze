from __future__ import annotations

from typing import Any

from autogaze_ext.utils.hf_cache import HFLoadConfig, redacted_hf_config
from autogaze_ext.utils.hf_offline import hf_offline_mode


class HFModelLoader:
    """Lazy Hugging Face model loader with no implicit downloads in tests."""

    MODEL_CLASS_NAMES = {
        "AutoModel": "AutoModel",
        "AutoModelForCausalLM": "AutoModelForCausalLM",
        "AutoModelForVision2Seq": "AutoModelForVision2Seq",
    }

    def __init__(self, config: HFLoadConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, HFLoadConfig) else HFLoadConfig.from_mapping(config or {})
        self.last_load_info: dict[str, Any] | None = None

    def _resolve_model_class(self, transformers_module: Any, model_class: str | None = None) -> Any:
        class_name = model_class or self.config.model_class or "AutoModel"
        if class_name not in self.MODEL_CLASS_NAMES:
            raise ValueError(f"Unsupported HF model_class '{class_name}'")
        return getattr(transformers_module, self.MODEL_CLASS_NAMES[class_name])

    def load_model(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        dtype: Any | None = None,
        model_class: str | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved_model_id = model_id or self.config.model_id
        if not resolved_model_id:
            raise ValueError("model_id is required")

        try:
            import transformers  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers is required for HFModelLoader") from exc

        load_cfg = HFLoadConfig.from_mapping({**self.config.__dict__, "revision": revision or self.config.revision})
        load_kwargs = {**load_cfg.common_kwargs(), **kwargs}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        model_cls = self._resolve_model_class(transformers, model_class=model_class)
        with hf_offline_mode(load_cfg.offline):
            model = model_cls.from_pretrained(resolved_model_id, **load_kwargs)
        if device is not None and hasattr(model, "to"):
            model = model.to(device)
        self.last_load_info = {**redacted_hf_config(load_cfg), "model_id": resolved_model_id, "loader": model_cls.__name__}
        return model

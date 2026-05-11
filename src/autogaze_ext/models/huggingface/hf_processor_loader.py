from __future__ import annotations

from typing import Any

from autogaze_ext.utils.hf_cache import HFLoadConfig, redacted_hf_config
from autogaze_ext.utils.hf_offline import hf_offline_mode


class HFProcessorLoader:
    def __init__(self, config: HFLoadConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, HFLoadConfig) else HFLoadConfig.from_mapping(config or {})
        self.last_load_info: dict[str, Any] | None = None

    def load_processor(self, model_id: str | None = None, revision: str | None = None, **kwargs: Any) -> Any:
        return self._load("AutoProcessor", model_id=model_id, revision=revision, **kwargs)

    def load_tokenizer(self, model_id: str | None = None, revision: str | None = None, **kwargs: Any) -> Any:
        return self._load("AutoTokenizer", model_id=model_id, revision=revision, **kwargs)

    def _load(self, class_name: str, model_id: str | None = None, revision: str | None = None, **kwargs: Any) -> Any:
        resolved_model_id = model_id or self.config.model_id
        if not resolved_model_id:
            raise ValueError("model_id is required")
        try:
            import transformers  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers is required for HFProcessorLoader") from exc

        load_cfg = HFLoadConfig.from_mapping({**self.config.__dict__, "revision": revision or self.config.revision})
        load_kwargs = {**load_cfg.common_kwargs(), **kwargs}
        cls = getattr(transformers, class_name)
        with hf_offline_mode(load_cfg.offline):
            obj = cls.from_pretrained(resolved_model_id, **load_kwargs)
        self.last_load_info = {**redacted_hf_config(load_cfg), "model_id": resolved_model_id, "loader": class_name}
        return obj

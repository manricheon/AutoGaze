from __future__ import annotations

from typing import Any

from repro.plugins.mllm_adapters import qwen_video_token_id


def project_vjepa_features_to_qwen_dim(vjepa_features: Any, *, qwen_hidden_size: int) -> Any:
    if qwen_hidden_size <= 0:
        raise ValueError("qwen_hidden_size must be positive")
    if not hasattr(vjepa_features, "shape"):
        raise ValueError("vjepa_features must be a tensor-like object")
    if len(vjepa_features.shape) != 3:
        raise ValueError("vjepa_features must have shape [batch, tokens, hidden]")

    hidden_size = int(vjepa_features.shape[-1])
    if hidden_size <= 0:
        raise ValueError("vjepa_features hidden size must be positive")
    if hidden_size == int(qwen_hidden_size):
        return vjepa_features
    if hidden_size > int(qwen_hidden_size):
        return vjepa_features[..., : int(qwen_hidden_size)]

    repeats = (int(qwen_hidden_size) + hidden_size - 1) // hidden_size
    repeat_dims = [1] * len(vjepa_features.shape)
    repeat_dims[-1] = repeats
    return vjepa_features.repeat(*repeat_dims)[..., : int(qwen_hidden_size)]


def build_qwen_bridge_inputs_from_vjepa_features(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    projected_vjepa_features: Any,
) -> dict[str, Any]:
    import torch

    if len(projected_vjepa_features.shape) != 3:
        raise ValueError("projected_vjepa_features must have shape [batch, tokens, qwen_hidden]")
    if int(projected_vjepa_features.shape[0]) != 1:
        raise ValueError("Qwen bridge smoke currently supports a single video batch")

    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized.get("attention_mask")
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("Qwen bridge smoke currently supports a single prompt batch")

    video_token_id = qwen_video_token_id(model)
    selected_count = int(projected_vjepa_features.shape[1])
    video_tokens = torch.full(
        (1, selected_count),
        int(video_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    packed_input_ids = torch.cat([input_ids[:, :1], video_tokens, input_ids[:, 1:]], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    packed_attention = torch.cat(
        [
            attention_mask[:, :1],
            torch.ones((1, selected_count), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask[:, 1:],
        ],
        dim=1,
    )

    embeddings = _qwen_input_embeddings(model)(packed_input_ids)
    features = projected_vjepa_features.to(device=embeddings.device, dtype=embeddings.dtype)
    if int(features.shape[-1]) != int(embeddings.shape[-1]):
        raise ValueError(
            "projected_vjepa_features hidden size must match Qwen embedding hidden size "
            f"({int(features.shape[-1])} != {int(embeddings.shape[-1])})"
        )
    if selected_count:
        embeddings[:, 1 : 1 + selected_count, :] = features

    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention,
        "inputs_embeds": embeddings,
        "vjepa_qwen_bridge_metadata": {
            "status": "zero_shot_wiring_probe",
            "visual_tokens_inserted": selected_count,
            "qwen_hidden_size": int(embeddings.shape[-1]),
            "video_token_id": int(video_token_id),
            "projection": "deterministic_repeat_or_truncate_untrained",
            "accuracy_status": "not_claimed",
        },
    }


def run_fake_qwen_bridge_smoke(
    *,
    selected_token_count: int,
    vjepa_hidden_size: int,
    qwen_hidden_size: int,
) -> dict[str, Any]:
    import torch

    class FakeTokenizer:
        def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([[101, 102, 103]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        def batch_decode(
            self,
            token_ids: Any,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = False,
        ) -> list[str]:
            return ["fake decoded answer"]

    class FakeQwenModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"video_token_id": 999})()
            self.embedding = torch.nn.Embedding(1200, int(qwen_hidden_size))
            self.generate_calls: list[dict[str, Any]] = []

        def get_input_embeddings(self) -> Any:
            return self.embedding

        def generate(self, **kwargs: Any) -> Any:
            self.generate_calls.append(kwargs)
            return torch.tensor([[101, 999, 999, 102, 103, 7, 8]], dtype=torch.long)

    if selected_token_count <= 0:
        raise ValueError("selected_token_count must be positive")
    if vjepa_hidden_size <= 0:
        raise ValueError("vjepa_hidden_size must be positive")

    model = FakeQwenModel()
    tokenizer = FakeTokenizer()
    feature_values = torch.arange(
        selected_token_count * vjepa_hidden_size,
        dtype=torch.float32,
    ).reshape(1, int(selected_token_count), int(vjepa_hidden_size))
    projected = project_vjepa_features_to_qwen_dim(feature_values, qwen_hidden_size=qwen_hidden_size)
    packed = build_qwen_bridge_inputs_from_vjepa_features(
        model,
        tokenizer,
        prompt="Describe the video.",
        projected_vjepa_features=projected,
    )
    metadata = packed.pop("vjepa_qwen_bridge_metadata")
    generated_ids = model.generate(**packed)
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    generate_kwargs = model.generate_calls[-1]

    return {
        "status": "passed",
        "bridge_metadata": metadata,
        "generated_text": generated_text,
        "generate_kwargs": {
            "input_ids_shape": list(generate_kwargs["input_ids"].shape),
            "attention_mask_shape": list(generate_kwargs["attention_mask"].shape),
            "inputs_embeds_shape": list(generate_kwargs["inputs_embeds"].shape),
        },
    }


def _qwen_input_embeddings(model: Any) -> Any:
    for target in (model, getattr(model, "model", None)):
        getter = getattr(target, "get_input_embeddings", None)
        if getter is not None:
            embeddings = getter()
            if embeddings is not None:
                return embeddings
    raise ValueError("Qwen model does not expose get_input_embeddings")

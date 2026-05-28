from __future__ import annotations

import inspect
import time
from typing import Any


def gather_vjepa_hidden_states(hidden_states: Any, selected_token_indices: list[int] | tuple[int, ...]):
    import torch

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
    if not selected_token_indices:
        raise ValueError("selected_token_indices must not be empty")
    token_count = int(hidden_states.shape[1])
    selected = [int(index) for index in selected_token_indices]
    invalid = [index for index in selected if index < 0 or index >= token_count]
    if invalid:
        raise ValueError(f"selected_token_indices contain out-of-range indices: {invalid}")
    position_mask = torch.tensor(selected, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    position_mask = position_mask.repeat(int(hidden_states.shape[0]), 1)
    gather_index = position_mask.unsqueeze(-1).expand(-1, -1, int(hidden_states.shape[-1]))
    return torch.gather(hidden_states, dim=1, index=gather_index), position_mask


def run_vjepa_encoder_on_selected_embeddings(
    vjepa_encoder: Any,
    patch_embeddings: Any,
    *,
    selected_token_indices: list[int] | tuple[int, ...],
    output_attentions: bool = False,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    start = time.perf_counter()
    hidden_states, position_mask = gather_vjepa_hidden_states(patch_embeddings, selected_token_indices)
    _sync_tensor_device(hidden_states)
    timings["gather_selected_embeddings"] = _elapsed_ms(start)
    all_self_attentions = () if output_attentions else None

    layers_start = time.perf_counter()
    for layer_module in vjepa_encoder.layer:
        layer_start = time.perf_counter()
        layer_kwargs = _vjepa_layer_kwargs(
            layer_module,
            position_mask=position_mask,
            output_attentions=output_attentions,
        )
        layer_outputs = layer_module(hidden_states, **layer_kwargs)
        hidden_states = layer_outputs[0]
        _sync_tensor_device(hidden_states)
        layer_index = len([key for key in timings if key.startswith("encoder_layer_")])
        timings[f"encoder_layer_{layer_index}"] = _elapsed_ms(layer_start)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
    timings["encoder_layers_total"] = _elapsed_ms(layers_start)

    start = time.perf_counter()
    hidden_states = vjepa_encoder.layernorm(hidden_states)
    _sync_tensor_device(hidden_states)
    timings["layernorm"] = _elapsed_ms(start)
    timings["encoder_total"] = _elapsed_ms(total_start)
    raw_count = int(patch_embeddings.shape[1])
    selected_count = int(hidden_states.shape[1])
    return {
        "last_hidden_state": hidden_states,
        "position_mask": position_mask,
        "attentions": all_self_attentions,
        "metrics": {
            "raw_token_count": raw_count,
            "selected_token_count": selected_count,
            "encoder_token_reduction_ratio": float(raw_count) / float(selected_count) if selected_count else None,
            "stage_timings_ms": timings,
            "sparse_execution_policy": {
                "patch_embedding_scope": "dense_all_vjepa_tokens",
                "encoder_scope": "selected_vjepa_tokens_only",
                "position_policy": "original_vjepa_position_mask",
            },
        },
    }


def _vjepa_layer_kwargs(layer_module: Any, *, position_mask: Any, output_attentions: bool) -> dict[str, Any]:
    parameters = inspect.signature(layer_module.forward).parameters
    kwargs: dict[str, Any] = {}
    if "position_mask" in parameters:
        kwargs["position_mask"] = position_mask
    elif "attention_mask" in parameters:
        kwargs["attention_mask"] = position_mask
    if "head_mask" in parameters:
        kwargs["head_mask"] = None
    if "output_attentions" in parameters:
        kwargs["output_attentions"] = output_attentions
    return kwargs


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _sync_tensor_device(value: Any) -> None:
    device = getattr(value, "device", None)
    if getattr(device, "type", None) == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

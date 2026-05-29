import pytest

torch = pytest.importorskip("torch")

from repro.plugins.vjepa_qwen_bridge import (
    build_qwen_bridge_inputs_from_vjepa_features,
    decode_qwen_new_tokens,
    project_vjepa_features_to_qwen_dim,
    run_fake_qwen_bridge_smoke,
)


class FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        return {
            "input_ids": torch.tensor([[101, 102, 103]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["fake decoded answer"]


class FakeQwenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"video_token_id": 999})()
        self.embedding = torch.nn.Embedding(1200, 8)
        self.generate_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[101, 999, 999, 102, 103, 7, 8]], dtype=torch.long)


def test_project_vjepa_features_to_qwen_dim_repeats_or_truncates():
    small = torch.tensor([[[1.0, 2.0, 3.0]]])
    expanded = project_vjepa_features_to_qwen_dim(small, qwen_hidden_size=8)

    assert list(expanded.shape) == [1, 1, 8]
    assert expanded[0, 0].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]

    large = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10)
    truncated = project_vjepa_features_to_qwen_dim(large, qwen_hidden_size=4)

    assert truncated[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_build_qwen_bridge_inputs_inserts_projected_vjepa_features():
    model = FakeQwenModel()
    tokenizer = FakeTokenizer()
    features = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

    packed = build_qwen_bridge_inputs_from_vjepa_features(
        model,
        tokenizer,
        prompt="Describe the video.",
        projected_vjepa_features=features,
    )

    assert packed["input_ids"].tolist() == [[101, 999, 999, 102, 103]]
    assert packed["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert list(packed["inputs_embeds"].shape) == [1, 5, 8]
    assert packed["inputs_embeds"][0, 1].tolist() == features[0, 0].tolist()
    assert packed["inputs_embeds"][0, 2].tolist() == features[0, 1].tolist()
    assert packed["vjepa_qwen_bridge_metadata"] == {
        "status": "zero_shot_wiring_probe",
        "visual_tokens_inserted": 2,
        "qwen_hidden_size": 8,
        "video_token_id": 999,
        "projection": "deterministic_repeat_or_truncate_untrained",
        "accuracy_status": "not_claimed",
    }


def test_decode_qwen_new_tokens_removes_prompt_prefix():
    class TokenPrinter:
        def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return [" ".join(str(int(token)) for token in token_ids[0])]

    prompt_ids = torch.tensor([[101, 999, 999, 102, 103]], dtype=torch.long)
    generated_ids = torch.tensor([[101, 999, 999, 102, 103, 7, 8]], dtype=torch.long)

    text = decode_qwen_new_tokens(TokenPrinter(), generated_ids, prompt_ids)

    assert text == "7 8"


def test_run_fake_qwen_bridge_smoke_calls_generate():
    payload = run_fake_qwen_bridge_smoke(selected_token_count=3, vjepa_hidden_size=5, qwen_hidden_size=8)

    assert payload["status"] == "passed"
    assert payload["bridge_metadata"]["visual_tokens_inserted"] == 3
    assert payload["generated_text"] == "fake decoded answer"
    assert payload["generate_kwargs"]["inputs_embeds_shape"] == [1, 6, 8]

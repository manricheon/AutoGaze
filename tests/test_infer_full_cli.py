import argparse
import sys

import pytest

import autogaze.infer_full as infer_full


def _args(**overrides):
    base = dict(
        mllm="nvila",
        model_path="weights/NVILA-8B-HD-Video",
        autogaze_path="weights/AutoGaze",
        no_autogaze=False,
        gazing_ratio=0.75,
        integration=None,
        vjepa2_path=None,
        lm_path=None,
        projector_path=None,
        generic_processor_path=None,
        generic_vision_hook=None,
        generic_patch_grid=14,
        generic_has_cls_token=False,
        generic_media_key="images",
        generic_prompt_template="{prompt}",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_args_exposes_current_runner_keys(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer_full.py", "assets/example_input.mp4"])

    parser_args = infer_full.parse_args()

    assert parser_args.mllm == "nvila"
    assert hasattr(parser_args, "integration")
    assert hasattr(parser_args, "vjepa2_path")
    assert hasattr(parser_args, "generic_vision_hook")


def test_vjepa2_qwen25_uses_vjepa2_path_as_model_path():
    args = _args(
        mllm="vjepa2_qwen25",
        model_path="weights/NVILA-8B-HD-Video",
        vjepa2_path="weights/vjepa2-vitl-fpc64-256",
        lm_path="weights/Qwen2.5-7B-Instruct",
        integration="full",
    )

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(args)

    assert model_path == "weights/vjepa2-vitl-fpc64-256"
    assert autogaze_path == "weights/AutoGaze"
    assert ratio == 0.75
    assert kwargs == {
        "integration": "full",
        "lm_path": "weights/Qwen2.5-7B-Instruct",
    }


def test_vjepa2_nvila_keeps_nvila_model_and_forwards_vjepa2_path():
    args = _args(
        mllm="vjepa2_nvila",
        model_path="weights/NVILA-8B-HD-Video",
        vjepa2_path="weights/vjepa2-vitl-fpc64-256",
    )

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(args)

    assert model_path == "weights/NVILA-8B-HD-Video"
    assert autogaze_path == "weights/AutoGaze"
    assert ratio == 0.75
    assert kwargs == {"vjepa2_path": "weights/vjepa2-vitl-fpc64-256"}


def test_nvila_default_baseline_uses_hook_without_autogaze_path():
    args = _args(mllm="nvila", no_autogaze=True)

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(
        args,
        baseline=True,
    )

    assert model_path == "weights/NVILA-8B-HD-Video"
    assert autogaze_path is None
    assert ratio == 1.0
    assert kwargs == {"integration": "hook"}


def test_non_native_baseline_disables_autogaze_path():
    args = _args(mllm="siglip_qwen25", model_path="weights/Qwen2.5-VL-7B-Instruct")

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(
        args,
        baseline=True,
    )

    assert model_path == "weights/Qwen2.5-VL-7B-Instruct"
    assert autogaze_path is None
    assert ratio == 1.0
    assert kwargs == {}


def test_deprecated_full_alias_sets_full_integration():
    args = _args(mllm="qwen25vl_full", model_path="weights/Qwen2.5-VL-7B-Instruct")

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(args)

    assert model_path == "weights/Qwen2.5-VL-7B-Instruct"
    assert autogaze_path == "weights/AutoGaze"
    assert ratio == 0.75
    assert kwargs == {"integration": "full"}


def test_runner_has_autogaze_handles_native_ratio_and_selector():
    native_on = argparse.Namespace(integration="native", gazing_ratio=0.5)
    native_off = argparse.Namespace(integration="native", gazing_ratio=1.0)
    hook_on = argparse.Namespace(integration="hook", selector=object())
    hook_off = argparse.Namespace(integration="hook", selector=None)

    assert infer_full._runner_has_autogaze(native_on)
    assert not infer_full._runner_has_autogaze(native_off)
    assert infer_full._runner_has_autogaze(hook_on)
    assert not infer_full._runner_has_autogaze(hook_off)


def test_missing_vjepa2_path_exits_before_model_load(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer_full.py", "assets/example_input.mp4", "--mllm", "vjepa2_qwen25", "--lm-path", "lm"],
    )
    monkeypatch.setattr(infer_full, "get_video_info", lambda _: (16, 8.0))

    with pytest.raises(SystemExit):
        infer_full.main()


def test_generic_mllm_forwards_generic_options():
    args = _args(
        mllm="generic_mllm",
        model_path="some/model",
        generic_processor_path="some/processor",
        generic_vision_hook="vision_model.embeddings",
        generic_patch_grid=16,
        generic_has_cls_token=True,
        generic_media_key="videos",
        generic_prompt_template="<video>\n{prompt}",
    )

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(args)

    assert model_path == "some/model"
    assert autogaze_path == "weights/AutoGaze"
    assert ratio == 0.75
    assert kwargs == {
        "processor_path": "some/processor",
        "vision_hook": "vision_model.embeddings",
        "patch_grid": 16,
        "has_cls_token": True,
        "media_key": "videos",
        "prompt_template": "<video>\n{prompt}",
    }


def test_generic_mllm_no_autogaze_does_not_require_hook():
    args = _args(
        mllm="generic_mllm",
        model_path="some/model",
        no_autogaze=True,
        generic_vision_hook=None,
    )

    model_path, autogaze_path, ratio, kwargs = infer_full._runner_model_path_and_kwargs(args)

    assert model_path == "some/model"
    assert autogaze_path is None
    assert ratio == 0.75
    assert kwargs["vision_hook"] is None


def test_missing_generic_hook_with_autogaze_exits_before_model_load(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer_full.py", "assets/example_input.mp4", "--mllm", "generic_mllm"],
    )
    monkeypatch.setattr(infer_full, "get_video_info", lambda _: (16, 8.0))

    with pytest.raises(SystemExit):
        infer_full.main()

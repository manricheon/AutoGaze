from repro.failure_logging import classify_exception


def test_classify_exception_marks_missing_dependency():
    failure = classify_exception(
        RuntimeError("qwen_vl_utils is required for Qwen video processing; install with `pip install qwen-vl-utils`."),
        stage="mllm_generate",
    )

    assert failure["kind"] == "failed_missing_dependency"
    assert failure["stage"] == "qwen_video_input_build"

from repro.plugins.token_selector import (
    AutoGazeSelectorPlan,
    KeepAllTokenSelector,
    NoTokenSelector,
    TokenSelectorInput,
)


def make_input(raw_patch_tokens=1060):
    return TokenSelectorInput(
        raw_patch_tokens=raw_patch_tokens,
        frame_indices=[0, 8, 15],
        patch_space={"patch_size": 14, "target_scales": [56, 112, 196, 392]},
        gazing_ratio=0.1,
        task_loss_requirement=0.6,
    )


def test_no_token_selector_marks_autogaze_not_applicable():
    output = NoTokenSelector().select(make_input())

    assert output.selected_positions is None
    assert output.raw_patch_tokens == 1060
    assert output.selected_patch_tokens is None
    assert output.reduction_ratio is None
    assert output.status == "not_applicable"


def test_keep_all_token_selector_preserves_patch_tokens_for_native_off_comparison():
    output = KeepAllTokenSelector().select(make_input(raw_patch_tokens=2120))

    assert output.selected_positions == "keep_all"
    assert output.raw_patch_tokens == 2120
    assert output.selected_patch_tokens == 2120
    assert output.reduction_ratio == 1.0
    assert output.status == "keep_all"


def test_autogaze_selector_plan_marks_probe_required_without_concrete_patch_coordinates():
    output = AutoGazeSelectorPlan().select(make_input(raw_patch_tokens=1000))

    assert output.selected_positions is None
    assert output.selected_patch_tokens == 100
    assert output.reduction_ratio == 10.0
    assert output.status == "probe_required"
    assert output.metric_status["value"] == "probe_required"
    assert output.metric_status["reason"] == "AutoGaze selector has not emitted concrete patch coordinates."
    assert output.sparse_selection_plan is not None
    assert output.sparse_selection_plan.encoder_mapping.status == "not_mapped"
    assert output.sparse_selection_plan.mllm_mapping.status == "not_mapped"

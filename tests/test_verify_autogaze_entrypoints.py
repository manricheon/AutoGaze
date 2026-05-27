from scripts.verify_autogaze_entrypoints import check_plugin_qwen_routing
from scripts.verify_autogaze_entrypoints import entrypoint_matrix
from scripts.verify_autogaze_entrypoints import render_markdown


def test_entrypoint_matrix_covers_required_on_off_families():
    ids = {row["id"] for row in entrypoint_matrix()}

    assert "nvila_single_autogaze" in ids
    assert "nvila_single_keep_all_single" in ids
    assert "qwen_plugin_hlvid" in ids
    assert "vjepa_qwen_single" in ids
    assert "vjepa_qwen_hlvid" in ids
    assert "colab_cuda_smoke_wrapper" in ids


def test_plugin_qwen_routing_check_requires_sparse_mode_to_run_autogaze():
    check = check_plugin_qwen_routing()

    assert check["status"] == "passed"
    evidence = check["evidence"]
    assert evidence["qwen_full_vit"]["runs_autogaze_selector"] is False
    assert evidence["qwen_chunked_vit"]["runs_autogaze_selector"] is False
    assert evidence["qwen_chunked_vit_autogaze_sparse"]["runs_autogaze_selector"] is True
    assert evidence["qwen_chunked_vit_autogaze_sparse"]["forwards_autogaze_target_scales"] is True


def test_entrypoint_verification_markdown_lists_matrix_and_checks():
    payload = {
        "summary": {
            "passed": True,
            "command_count": 1,
            "check_count": 1,
            "elapsed_ms": 1.0,
        },
        "script_matrix": entrypoint_matrix()[:1],
        "checks": [{"name": "sample", "status": "passed", "evidence": {"ok": True}}],
        "commands": [{"name": "cmd", "returncode": 0}],
        "notes": ["note"],
    }

    markdown = render_markdown(payload)

    assert "# AutoGaze Entrypoint Verification" in markdown
    assert "nvila_single_autogaze" in markdown
    assert "| sample | `passed` |" in markdown

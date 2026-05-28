import json

from scripts.write_external_cuda_verification_notebook import build_notebook, main


def test_build_notebook_contains_kaggle_cuda_verification_steps():
    notebook = build_notebook(
        branch="codex/autogaze-vjepa",
        platform="kaggle",
        output_root="/kaggle/working/autogaze_vjepa_outputs",
        weights_root="/kaggle/working/autogaze_weights",
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "codex/autogaze-vjepa" in source
    assert "scripts/verify_autogaze_entrypoints.py" in source
    assert "scripts/run_colab_autogaze_cuda_smoke.py" in source
    assert "RUN_NVILA_SINGLE = True" in source
    assert "RUN_QWEN_SINGLE = True" in source
    assert "RUN_NVILA_HLVID_MINI = True" in source
    assert "RUN_QWEN_PLUGIN_HLVID_MINI = True" in source
    assert "RUN_VJEPA_QWEN_HLVID_MINI = True" in source
    assert "repro.nvila_runner" in source
    assert "repro.flexible_runner" in source
    assert "scripts/run_hlvid_folder_benchmark.py" in source
    assert "--plugin-suite" in source
    assert "64+128+192+224" in source
    assert "--visualization-output-dir" in source
    assert "--visualization-max-frames', '16'" in source
    assert "--dtype" in source
    assert "repro.colab_verification_report" in source
    assert "repro.vjepa_qwen_hlvid_benchmark" in source
    assert "/kaggle/working/autogaze_vjepa_outputs/colab_verification.md" in source
    assert "torch.cuda.is_available()" in source


def test_notebook_cli_writes_valid_ipynb(tmp_path):
    output = tmp_path / "verify.ipynb"

    main(["--output", str(output), "--platform", "colab", "--branch", "codex/autogaze-repro"])

    payload = json.loads(output.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert payload["nbformat"] == 4
    assert "codex/autogaze-repro" in source
    assert "/content/autogaze_vjepa_outputs" in source
    assert "nvidia/NVILA-8B-HD-Video" in source

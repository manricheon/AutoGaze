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
    assert "repro.colab_verification_report" in source
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

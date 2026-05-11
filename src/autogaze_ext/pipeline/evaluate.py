from __future__ import annotations

import argparse
from pathlib import Path

import torch

from autogaze_ext.pipeline.inference import run_inference
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config


def run_evaluate(config_name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR, batch_size: int = 2) -> dict[str, float]:
    cfg = load_config(Path(config_dir), config_name)
    result = run_inference(cfg, batch_size=batch_size)
    if result.task_type == "video_vqa":
        predictions = result.outputs["generated_text"]
        score = sum(pred == "dummy" for pred in predictions) / len(predictions)
        return {"dummy_exact_match": score}

    labels = result.outputs["predicted_labels"]
    if isinstance(labels, torch.Tensor):
        score = float((labels == 0).to(torch.float32).mean().item())
    else:
        score = 0.0
    return {"dummy_accuracy": score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint-free dummy AutoGaze extension inference")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="dummy_video_vqa")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    metrics = run_evaluate(args.config_name, args.config_dir, args.batch_size)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

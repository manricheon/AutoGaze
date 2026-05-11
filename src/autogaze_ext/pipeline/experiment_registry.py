from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    autogaze_enabled: bool
    vision_encoder_type: str
    mllm_type: str
    task_type: str
    integration_mode: str
    compatibility_status: str

    @property
    def autogaze_config(self) -> str:
        return "model/autogaze/on" if self.autogaze_enabled else "model/autogaze/off"

    @property
    def vision_encoder_config(self) -> str:
        return f"model/vision_encoder/{self.vision_encoder_type}"

    @property
    def mllm_config(self) -> str:
        return f"model/mllm/{self.mllm_type}"

    @property
    def task_config(self) -> str:
        return f"task/{self.task_type}/default"


EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "A0": ExperimentSpec(
        experiment_id="A0",
        autogaze_enabled=False,
        vision_encoder_type="vanilla_siglip",
        mllm_type="nvila",
        task_type="video_vqa",
        integration_mode="full_token_baseline",
        compatibility_status="canonical_baseline",
    ),
    "A1": ExperimentSpec(
        experiment_id="A1",
        autogaze_enabled=False,
        vision_encoder_type="modified_siglip",
        mllm_type="nvila",
        task_type="video_vqa",
        integration_mode="modified_siglip_full_token_baseline",
        compatibility_status="canonical_priority_path",
    ),
    "A2": ExperimentSpec(
        experiment_id="A2",
        autogaze_enabled=True,
        vision_encoder_type="modified_siglip",
        mllm_type="nvila",
        task_type="video_vqa",
        integration_mode="original_autogaze_modified_siglip_nvila",
        compatibility_status="canonical_priority_path_requires_original_autogaze",
    ),
    "A3": ExperimentSpec(
        experiment_id="A3",
        autogaze_enabled=True,
        vision_encoder_type="vanilla_siglip",
        mllm_type="nvila",
        task_type="video_vqa",
        integration_mode="experimental_autogaze_vanilla_siglip_nvila",
        compatibility_status="experimental_compatibility_ablation_not_validated",
    ),
}


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    normalized = experiment_id.upper()
    try:
        return EXPERIMENT_REGISTRY[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(EXPERIMENT_REGISTRY))
        raise ValueError(f"Unknown experiment ID '{experiment_id}'. Known experiments: {known}") from exc


def validate_experiment_config(cfg: DictConfig) -> ExperimentSpec:
    experiment_id = str(cfg.experiment.id).upper()
    spec = get_experiment_spec(experiment_id)

    checks = {
        "model.autogaze.enabled": (bool(cfg.model.autogaze.enabled), spec.autogaze_enabled),
        "model.vision_encoder.type": (str(cfg.model.vision_encoder.type), spec.vision_encoder_type),
        "model.mllm.type": (str(cfg.model.mllm.type), spec.mllm_type),
        "task.type": (str(cfg.task.type), spec.task_type),
        "experiment.integration_mode": (str(cfg.experiment.integration_mode), spec.integration_mode),
    }
    mismatches = [
        f"{field}: got {actual!r}, expected {expected!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"Invalid wiring for experiment {experiment_id}: " + "; ".join(mismatches))

    if spec.autogaze_enabled and spec.vision_encoder_type == "vanilla_siglip" and experiment_id != "A3":
        raise ValueError("AutoGaze ON + vanilla SigLIP is only registered as A3 experimental ablation")

    return spec

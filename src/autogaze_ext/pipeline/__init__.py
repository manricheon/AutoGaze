"""Pipeline entry points for the AutoGaze extension PoC."""

from autogaze_ext.pipeline.experiment_registry import (
    EXPERIMENT_REGISTRY,
    ExperimentSpec,
    get_experiment_spec,
    validate_experiment_config,
)

__all__ = [
    "EXPERIMENT_REGISTRY",
    "ExperimentSpec",
    "get_experiment_spec",
    "validate_experiment_config",
]

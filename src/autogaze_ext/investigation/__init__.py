"""Investigation utilities for real AutoGaze canonical-path readiness."""

from autogaze_ext.investigation.canonical_path import (
    CanonicalPathReport,
    ComponentCheck,
    ExperimentReadiness,
    check_canonical_path,
)
from autogaze_ext.investigation.quick_start_reference import QuickStartLocation, locate_quick_start
from autogaze_ext.investigation.model_construction import (
    ComponentConstructionResult,
    ModelConstructionReport,
    run_model_construction_check,
)
from autogaze_ext.investigation.canonical_smoke_inference import (
    SmokeInferenceReport,
    StageReport,
    run_canonical_smoke_inference,
)
from autogaze_ext.investigation.vanilla_siglip_feasibility import (
    ModeFeasibility,
    VanillaSigLIPFeasibilityReport,
    check_vanilla_siglip_feasibility,
)

__all__ = [
    "CanonicalPathReport",
    "ComponentCheck",
    "ComponentConstructionResult",
    "ExperimentReadiness",
    "ModelConstructionReport",
    "ModeFeasibility",
    "QuickStartLocation",
    "SmokeInferenceReport",
    "StageReport",
    "VanillaSigLIPFeasibilityReport",
    "check_canonical_path",
    "check_vanilla_siglip_feasibility",
    "locate_quick_start",
    "run_canonical_smoke_inference",
    "run_model_construction_check",
]

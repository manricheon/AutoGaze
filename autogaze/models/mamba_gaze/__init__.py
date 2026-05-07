# v1 — AR Mamba decoder (replaces LLaMA with Mamba SSM, still autoregressive)
from .configuration_mamba_gaze import MambaGazeConfig, MambaGazeModelConfig
from .modeling_mamba_gaze import (
    MambaGaze,
    MambaGazeModel,
    MambaVisionEncoder,
    MambaGazeDecoder,
    MambaBlock,
    SelectiveSSM,
)

# v2 — Feedforward (non-AR) token selector  (Branch A, mamba_gaze_ref_v0.md)
from .configuration_mamba_gaze_ff import MambaGazeFFConfig
from .modeling_mamba_gaze_ff import (
    MambaGazeFF,
    LightweightCNNEncoder,
    SaliencyHead,
    SpatioTemporalMambaAggregator,
)

__all__ = [
    # v1 AR
    "MambaGazeConfig",
    "MambaGazeModelConfig",
    "MambaGaze",
    "MambaGazeModel",
    "MambaVisionEncoder",
    "MambaGazeDecoder",
    "MambaBlock",
    "SelectiveSSM",
    # v2 Feedforward
    "MambaGazeFFConfig",
    "MambaGazeFF",
    "LightweightCNNEncoder",
    "SaliencyHead",
    "SpatioTemporalMambaAggregator",
]

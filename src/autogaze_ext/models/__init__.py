"""Model adapters for the AutoGaze extension PoC."""

from autogaze_ext.models.autogaze_wrapper import (
    AutoGazeOutput,
    AutoGazeWrapper,
    detect_original_autogaze,
)

__all__ = ["AutoGazeOutput", "AutoGazeWrapper", "detect_original_autogaze"]

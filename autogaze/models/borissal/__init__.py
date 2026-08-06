from .configuration_borissal import BorissalConfig, BorissalV1Config
from .modeling_borissal import Borissal, Selection
from .modeling_borissal_v1 import BorissalV1
from .device import resolve_device, available_devices
from . import adapters

__version__ = "v0.2"
MODEL_TAG = "borissal-v0"          # model-line tag (v0.x share the line)
PRESET_TAG_V02 = "borissal-v0.2"   # the tuned v0.2 preset (BorissalConfig.v0_2())
MODEL_TAG_V1 = "borissal-v1"

__all__ = [
    "BorissalConfig", "Borissal", "Selection",
    "BorissalV1Config", "BorissalV1",
    "resolve_device", "available_devices", "adapters",
    "__version__", "MODEL_TAG", "PRESET_TAG_V02", "MODEL_TAG_V1",
]

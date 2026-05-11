from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def hf_offline_env(offline: bool) -> dict[str, str]:
    if not offline:
        return {}
    return {key: "1" for key in HF_OFFLINE_ENV_VARS}


@contextmanager
def hf_offline_mode(offline: bool) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in HF_OFFLINE_ENV_VARS}
    try:
        if offline:
            for key in HF_OFFLINE_ENV_VARS:
                os.environ[key] = "1"
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

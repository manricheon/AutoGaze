from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool
    cuda_available: bool
    mps_available: bool


def set_seed(seed: int, *, deterministic: bool = False) -> SeedState:
    """Seed common RNGs without enabling deterministic kernels unless requested."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    return SeedState(
        seed=seed,
        deterministic=bool(deterministic),
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=bool(torch.backends.mps.is_available()),
    )


#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autogaze_ext.investigation.vanilla_siglip_feasibility import main


if __name__ == "__main__":
    main()

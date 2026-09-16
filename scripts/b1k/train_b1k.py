#!/usr/bin/env python3
"""Train upstream ACT or CNNMLP from a local BEHAVIOR LeRobot v3 root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_training import main

if __name__ == '__main__':
    main()

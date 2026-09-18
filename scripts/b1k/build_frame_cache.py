#!/usr/bin/env python3
"""Build and verify the uint8 resized-frame cache used by `train_b1k.py --frame-cache`."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_frame_cache import main

if __name__ == '__main__':
    main()

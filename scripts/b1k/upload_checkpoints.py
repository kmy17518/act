#!/usr/bin/env python3
"""Publish queued B1K checkpoints without importing the trainer or CUDA."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_checkpoint_upload import main

if __name__ == '__main__':
    raise SystemExit(main())

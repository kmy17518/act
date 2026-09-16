#!/usr/bin/env python3
"""Serve a standalone ACT/CNNMLP checkpoint with the BEHAVIOR WebSocket protocol."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_server import main

if __name__ == '__main__':
    main()

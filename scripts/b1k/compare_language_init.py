#!/usr/bin/env python3
"""Run the paired ACT baseline/random-FiLM/identity-FiLM initialization diagnostic."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_compare_language_init import main

if __name__ == '__main__':
    main()

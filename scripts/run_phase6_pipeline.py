#!/usr/bin/env python
"""Reproducibility script: Execute Phase 6 regression, OOF modeling, and HSED analysis."""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase6_regression import run_all_phase6

if __name__ == "__main__":
    success = run_all_phase6()
    sys.exit(0 if success else 1)

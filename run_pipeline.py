"""
Root-level convenience entry point.

Usage (from the project root):
    python run_pipeline.py all
    python run_pipeline.py train --max-ratings 500000
    python run_pipeline.py profile

This is equivalent to:
    python -m src.pipeline all
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is always on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_arg_parser, run_pipeline  # noqa: E402

if __name__ == "__main__":
    parser = build_arg_parser()
    run_pipeline(parser.parse_args())

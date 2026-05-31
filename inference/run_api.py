#!/usr/bin/env python3
"""Cross-platform launcher for the KrishiVaidya Inference API.

Delegates entirely to inference.api.main — ngrok is now managed
inside the FastAPI lifespan so it works regardless of entry point.

Usage:
    python run_api.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.api.main import main

if __name__ == "__main__":
    main()

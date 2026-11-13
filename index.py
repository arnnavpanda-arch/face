"""
Vercel Serverless Function Entrypoint
Exports the Flask application instance for @vercel/python runtime.
"""

import sys
from pathlib import Path

# Ensure project root is at the front of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

"""conftest.py — shared pytest configuration."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.paper4...` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

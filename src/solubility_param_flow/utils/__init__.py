"""Utility functions."""

import os
from pathlib import Path


def ensure_dir(path: str) -> str:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> str:
    """Get project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

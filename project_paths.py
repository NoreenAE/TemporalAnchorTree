"""Portable repository paths for saved experimental provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> Path:
    """Resolve a saved project-relative path independently of the working directory."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(value: str | Path) -> str:
    """Represent repository paths without including a local checkout location."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def portable_metadata(value: Any) -> Any:
    """Convert path values recursively without changing numerical metadata."""
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, str) and Path(value).is_absolute():
        return relative_path(value)
    return value

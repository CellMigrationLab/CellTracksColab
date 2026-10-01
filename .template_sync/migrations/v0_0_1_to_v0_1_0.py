from __future__ import annotations

from pathlib import Path
from typing import Any


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Bootstrap migration for the versioned template sync flow.

    The workflow already updates `.template_sync` before executing this script.
    This migration is intentionally a no-op so the initial PR only contains the
    bootstrap files. Future migrations can make targeted repository changes.
    """

    _ = repo_root
    _ = context

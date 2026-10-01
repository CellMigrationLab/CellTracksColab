from __future__ import annotations

from pathlib import Path
from typing import Any


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Record the first-time migration compatibility hardening release.

    The functional changes are in the migration engine's historical notebook and
    documentation handlers. Those handlers now accept initialized repositories
    where project placeholders have already been replaced with real values.
    """

    _ = repo_root
    _ = context
    print("Template migration compatibility hardening is active.")

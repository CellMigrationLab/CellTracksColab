from __future__ import annotations

from pathlib import Path
from typing import Any


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Register a repository that predates template-sync metadata.

    A missing local manifest is represented by version ``0.0.0`` in the
    synchronization workflow. The workflow checks out the current
    ``.template_sync`` directory before running migrations, so no repository
    files need to be changed in this bootstrap step. Advancing to ``0.0.1``
    connects first-time repositories to the existing migration chain.
    """

    _ = repo_root
    _ = context

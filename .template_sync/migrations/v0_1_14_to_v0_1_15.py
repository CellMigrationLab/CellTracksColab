from __future__ import annotations

from pathlib import Path
from typing import Any


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_14_to_v0_1_15"

# Template-owned files that make up the App Centre manifest check. labconstrictor-app.yaml itself
# belongs to the app: it is never created, changed or removed by this migration.
MANAGED_FILES = (
    Path(".tools/schema/labconstrictor-app.schema.json"),
    Path(".tools/python/check_app_manifest.py"),
    Path(".tools/docs/app_manifest.md"),
    Path(".github/workflows/check_app_manifest.yml"),
)


def copy_resource(repo_root: Path, relative_path: Path) -> None:
    """Copy one resource file into the repository, replacing the template-owned copy if it exists.
    Raises FileNotFoundError when the resource is missing from the template."""
    source = RESOURCE_ROOT / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Template resource not found: {source}")
    destination = repo_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Add the App Centre manifest check (script, schema copy, workflow and guide) to the repository."""
    for relative_path in MANAGED_FILES:
        copy_resource(repo_root, relative_path)

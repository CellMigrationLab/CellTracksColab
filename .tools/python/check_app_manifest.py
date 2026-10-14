"""Check labconstrictor-app.yaml, the optional description of an app for the LabConstrictor App Centre.

    python .tools/python/check_app_manifest.py [--repository-root .]

Exit status: 0 = the file is valid, or there is no file (a GitHub warning says how to add one);
1 = the file is invalid (every problem is reported as a GitHub error annotation);
2 = the check itself could not run (for example the schema copy is missing).
The check never changes any file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

MANIFEST_NAME = "labconstrictor-app.yaml"
SCHEMA_RELATIVE = Path(".tools/schema/labconstrictor-app.schema.json")
DOCS_HINT = "See .tools/docs/app_manifest.md."


class CheckError(Exception):
    """The check could not run (not a problem with the manifest itself)."""


def load_schema(root: Path) -> dict[str, Any]:
    """The vendored manifest schema. Raises CheckError when it is missing or not valid JSON."""
    path = root / SCHEMA_RELATIVE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise CheckError(f"cannot read the schema copy {SCHEMA_RELATIVE}: {err}") from err


def load_manifest(path: Path) -> Any:
    """The parsed manifest. Raises ValueError with the YAML message when the file is not valid YAML."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"not valid YAML: {err}") from err


def schema_problems(manifest: Any, schema: dict[str, Any]) -> list[str]:
    """One "path: message" line per way the manifest breaks the schema; empty when it is valid."""
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(str(p) for p in e.absolute_path) or 'manifest'}: {e.message}" for e in errors]


def referenced_paths(manifest: dict[str, Any]) -> list[str]:
    """The repository files the manifest names: its icon and its screenshots."""
    paths = [manifest["icon"]] if "icon" in manifest else []
    return paths + [shot["path"] for shot in manifest.get("screenshots", [])]


def file_problems(manifest: dict[str, Any], root: Path) -> list[str]:
    """One line per file the manifest names that does not exist in the repository."""
    return [f"{path}: file not found in the repository" for path in referenced_paths(manifest) if not (root / path).is_file()]


def annotate(level: str, message: str, title: str) -> None:
    """Print a GitHub Actions annotation (shown on the run and on the file)."""
    print(f"::{level} file={MANIFEST_NAME},title={title}::{message}")


def check(root: Path) -> int:
    """Run the check; returns the exit status described at the top of this file."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        annotate("warning", f"{MANIFEST_NAME} is missing, so this app cannot be listed in the LabConstrictor App Centre. {DOCS_HINT}", "App Centre manifest missing")
        return 0
    schema = load_schema(root)
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as err:
        annotate("error", str(err), "App Centre manifest invalid")
        return 1
    problems = schema_problems(manifest, schema)
    if not problems:
        problems = file_problems(manifest, root)
    for problem in problems:
        annotate("error", problem, "App Centre manifest invalid")
    print(f"{MANIFEST_NAME}: {'invalid' if problems else 'valid'}")
    return 1 if problems else 0


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        return check(args.repository_root.resolve())
    except CheckError as err:
        print(f"::error title=App Centre check could not run::{err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

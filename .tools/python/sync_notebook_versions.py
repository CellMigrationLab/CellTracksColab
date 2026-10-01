#!/usr/bin/env python3
"""
Extract the `current_version` variable from each notebook and update
`notebooks/notebook_latest_versions.yaml` accordingly.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUT_FILE = NOTEBOOKS_DIR / "notebook_latest_versions.yaml"


@dataclass
class NotebookVersion:
    category: str
    name: str
    version: str
    path: Path


def iter_notebooks(paths: Iterable[Path] | None) -> Iterable[Path]:
    """Yield notebook paths under NOTEBOOKS_DIR."""
    if paths:
        for explicit in paths:
            explicit = explicit.resolve()
            if not explicit.exists():
                raise SystemExit(f"Notebook not found: {explicit}")
            yield explicit
        return

    for ipynb in NOTEBOOKS_DIR.glob("**/*.ipynb"):
        if ipynb.name.startswith("."):
            continue
        yield ipynb


def extract_current_version(nb_path: Path) -> str | None:
    """Return the value assigned to `current_version` inside the notebook."""
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Failed reading {nb_path}: {exc}") from exc

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        raw_source = "".join(cell.get("source", []))
        # Replace IPython magics and shell commands so ast.parse doesn't choke.
        sanitized_lines = []
        for line in raw_source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("!") or stripped.startswith("%"):
                indent = line[: len(line) - len(stripped)]
                sanitized_lines.append(f"{indent}pass")
            else:
                sanitized_lines.append(line)
        source = "\n".join(sanitized_lines)

        try:
            module = ast.parse(source)
        except SyntaxError:
            print(f"Your notebook {nb_path} contains invalid syntax. Skipping.")
            continue

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "current_version":
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
    return "0.0.0"


def discover_versions(paths: Iterable[Path] | None) -> tuple[list[NotebookVersion], list[str]]:
    """Collect NotebookVersion objects for all notebooks of interest."""
    results: list[NotebookVersion] = []
    missing: list[str] = []

    for nb_path in iter_notebooks(paths):
        # Check if its a .ipynb file
        if nb_path.suffix != ".ipynb":
            continue

        try:
            relative = nb_path.relative_to(NOTEBOOKS_DIR)
        except ValueError:
            raise SystemExit(f"{nb_path} is outside {NOTEBOOKS_DIR}")

        if relative.parts[0].startswith("."):
            # Skip hidden directories
            continue

        if len(relative.parts) < 2:
            # Expect at least category/file
            category = "root"
        else:
            category = relative.parts[0]

        version = extract_current_version(nb_path)
        if not version:
            missing.append(str(relative))
            continue

        name = nb_path.stem
        results.append(NotebookVersion(category, name, version, nb_path))

    return results, missing


def render_global_yaml(grouped: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    for category in sorted(grouped):
        lines.append(f'"{category}":')
        for name, version in grouped[category].items():
            lines.append(f'  "{name}": "{version}"')
    return "\n".join(lines) + "\n"


def group_versions(versions: Iterable[NotebookVersion]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for item in versions:
        bucket = grouped.setdefault(item.category, {})
        bucket[item.name] = item.version
    # Sort names to keep output stable
    for category in grouped:
        grouped[category] = dict(sorted(grouped[category].items()))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update notebook_latest_versions.yaml from notebook metadata."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional notebook paths to update (defaults to all notebooks).",
    )
    args = parser.parse_args()

    versions, missing = discover_versions(args.paths)
    grouped = group_versions(versions)

    global_yaml = render_global_yaml(grouped)
    OUTPUT_FILE.write_text(global_yaml, encoding="utf-8")
    for item in versions:
        print(f"{item.category}/{item.name}: {item.version}")
    print(f"updated {OUTPUT_FILE.relative_to(ROOT)}")
    if missing:
        joined = "\n".join(f"- {item}" for item in sorted(missing))
        print(
            "Notebooks missing `current_version` were skipped:\n"
            f"{joined}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stop a release whose installer would ship a template placeholder (CellMigrationLab/LabConstrictor#40).

The repository initialisation (the LabConstrictor web form) replaces the template's placeholders with the app's names. A placeholder
it misses survives into the installer: `import PYTHON_PROJ_NAME` in post_install.sh / .bat made every installer of an app with Python
code in src/ fail at its last step. This script runs in the release workflow after the versioned files are updated and before anything
is committed or built, and fails with the file, the line and the fix when an initialisation placeholder is left in a file the
installer ships (app/**, construct.yaml, setup.py, environment.yaml), or when the repository is the raw, uninitialised template.

Placeholders that are filled in later on purpose are not checked: BASE_PATH_KEYWORD (at install time), PYTHON_VERSION and
JUPYTERLAB_VERSION (by the requirements workflow).

Usage: python .tools/python/check_placeholders.py [--repository-root PATH]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_23.md"

# The placeholders the initialisation replaces (keys of conversion_table in LabConstrictor_streamlit's placeholders.py,
# plus the launcher icon paths it fills in itself). VERSION_NUMBER is also rewritten by bump_version.py, which runs before this check.
INIT_PLACEHOLDERS = (
    "LOWER_PROJ_NAME",
    "PYTHON_PROJ_NAME",
    "UNDERSCORED_PROJECT_NAME",
    "PROJECT_NAME",
    "VERSION_NUMBER",
    "WELCOME_IMAGE",
    "HEADER_IMAGE",
    "ICON_IMAGE",
    "GITHUB_OWNER",
    "GITHUB_REPO_NAME",
    "HIDE_CODE_DISABLED",
    "ICON_IMAGE_PATH",
    "ICON_ICO_IMAGE_PATH",
    "ICON_ICNS_IMAGE_PATH",
)
SHIPPED_FILES = ("construct.yaml", "setup.py", "environment.yaml")
SHIPPED_FOLDERS = ("app",)
WELCOME_NOTEBOOK = "app/menuinst/Welcome.ipynb"
WELCOME_TEMPLATE = ".tools/templates/Welcome_template.ipynb"


def placeholder_pattern(name: str) -> re.Pattern[str]:
    """A placeholder on its own or followed by '_' (PYTHON_PROJ_NAME_lc_tools), never inside a longer name (UNDERSCORED_PROJECT_NAME)."""
    return re.compile(r"(?<![A-Za-z0-9_])" + name + r"(?![A-Za-z0-9])")


PATTERNS = {name: placeholder_pattern(name) for name in INIT_PLACEHOLDERS}


def shipped_files(root: Path) -> list[Path]:
    files = [root / name for name in SHIPPED_FILES if (root / name).is_file()]
    for folder in SHIPPED_FOLDERS:
        if (root / folder).is_dir():
            files.extend(sorted(p for p in (root / folder).rglob("*") if p.is_file()))
    return files


def leftover_placeholders(root: Path) -> list[str]:
    problems = []
    for path in shipped_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # images, icons and other binary files
        rel = path.relative_to(root).as_posix()
        # The Welcome notebook is rebuilt from its template at every release: the fix belongs in the template.
        where = f" (fix it in {WELCOME_TEMPLATE}, which this notebook is made from)" if rel == WELCOME_NOTEBOOK else ""
        for number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    problems.append(f"{rel}:{number}: the template placeholder {name} was never replaced{where}: {line.strip()[:160]}")
    return problems


def check(root: Path) -> list[str]:
    construct = root / "construct.yaml"
    if construct.is_file() and re.search(r"^name:\s*UNDERSCORED_PROJECT_NAME\s*$", construct.read_text(encoding="utf-8"), re.MULTILINE):
        return ["construct.yaml: this is the raw LabConstrictor template; initialise the repository with the LabConstrictor form before releasing it"]
    return leftover_placeholders(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when a template placeholder is left in a file the installer ships.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    problems = check(args.repository_root.resolve())
    if not problems:
        print("No template placeholder is left in the files the installer ships.")
        return 0
    print("The installer would be broken: these files still hold template placeholders.", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        "Replace each placeholder with your app's value (PYTHON_PROJ_NAME: the Python package of the app, usually its folder in src/; "
        "PROJECT_NAME: the app name), "
        f"commit, and publish the release again. See {DOCS_URL}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

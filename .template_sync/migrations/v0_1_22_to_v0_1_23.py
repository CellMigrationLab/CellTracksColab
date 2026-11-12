"""Template 0.1.22 -> 0.1.23: no template placeholder in the installer (CellMigrationLab/LabConstrictor#40).

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_23.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. Placeholders the repository initialisation missed are replaced with the app's values, only where they are still there:
   - app/bash_bat_scripts/post_install.sh / .bat: PYTHON_PROJ_NAME (the import check and the <package>_lc_tools module). An app
     initialised on template 0.1.16 or later with Python code in src/ could not be installed: `No module named 'PYTHON_PROJ_NAME'`.
   - app/python_scripts/launch_jupyter.py: PROJECT_NAME (PROJECT_DISPLAY_NAME, the launcher's window titles and error messages).
   - .tools/templates/Welcome_template.ipynb: PROJECT_NAME and PYTHON_PROJ_NAME ("Welcome to PROJECT_NAME!" in the Playground).
   The package is found as in the 0.1.16 migration (the single <package>_lc_tools module or package in src/, then setup.py, then
   pyproject.toml); the app name is the menu name of app/menuinst/notebook_launcher.json, which the initialisation wrote.
2. .tools/python/check_placeholders.py (new) and two lines in .github/workflows/release_constructor.yml that run it after the
   versioned files are updated: a release whose installer would still ship a placeholder stops with the file, the line and the fix.

Rules this migration keeps: surgical (only placeholders, one new file and the known workflow lines), idempotent (a second run reports
no changes), never raises for a customised file (the sync pull request must still be created), the line endings of each file are
kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import json
import re
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_23.md"
TAG = "[migration 0.1.23]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_22_to_v0_1_23"

POST_INSTALL_SH = Path("app/bash_bat_scripts/post_install.sh")
POST_INSTALL_BAT = Path("app/bash_bat_scripts/post_install.bat")
LAUNCHER = Path("app/python_scripts/launch_jupyter.py")
WELCOME_TEMPLATE = Path(".tools/templates/Welcome_template.ipynb")
NOTEBOOK_LAUNCHER = Path("app/menuinst/notebook_launcher.json")
CHECK_SCRIPT = Path(".tools/python/check_placeholders.py")
RELEASE_WORKFLOW = Path(".github/workflows/release_constructor.yml")

# Which placeholders each file may still hold, and where their value comes from ("package" or "name").
PLACEHOLDER_FILES: dict[Path, tuple[tuple[str, str], ...]] = {
    POST_INSTALL_SH: (("PYTHON_PROJ_NAME", "package"),),
    POST_INSTALL_BAT: (("PYTHON_PROJ_NAME", "package"),),
    LAUNCHER: (("PROJECT_NAME", "name"),),
    WELCOME_TEMPLATE: (("PYTHON_PROJ_NAME", "package"), ("PROJECT_NAME", "name")),
}

# The release workflow runs bump_Welcome_notebook.py last in both jobs; the check goes right after it, with the same log file.
WELCOME_STEP = re.compile(
    r"(?P<indent>[ ]+)bash \.tools/bash/run_with_context\.sh \\\n"
    r"(?P=indent)  (?P<log>\S+) \\\n"
    r"(?P=indent)  \"bump_Welcome_notebook\.py\" \\\n"
    r"(?P=indent)  \.tools/python/bump_Welcome_notebook\.py \\\n"
    r"(?P=indent)  python \.tools/python/bump_Welcome_notebook\.py [^\n]*\n"
)
CHECK_STEP = (
    "{indent}bash .tools/bash/run_with_context.sh \\\n"
    "{indent}  {log} \\\n"
    '{indent}  "check_placeholders.py" \\\n'
    "{indent}  .tools/python/check_placeholders.py \\\n"
    "{indent}  python .tools/python/check_placeholders.py\n"
)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load(name: str, alias: str) -> ModuleType:
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def placeholder_pattern(name: str) -> re.Pattern[str]:
    """A placeholder on its own or followed by '_' (PYTHON_PROJ_NAME_lc_tools), never inside a longer name (UNDERSCORED_PROJECT_NAME)."""
    return re.compile(r"(?<![A-Za-z0-9_])" + name + r"(?![A-Za-z0-9])")


def app_name(helpers: Any, repo_root: Path) -> str | None:
    """The app name the initialisation wrote for PROJECT_NAME: the menu name of notebook_launcher.json, else PROJECT_DISPLAY_NAME."""
    path = repo_root / NOTEBOOK_LAUNCHER
    if path.is_file():
        try:
            name = json.loads(helpers.read_text(path)).get("menu_name")
        except (ValueError, AttributeError):
            name = None
        if isinstance(name, str) and name.strip() and not placeholder_pattern("PROJECT_NAME").search(name):
            return name.strip()
    path = repo_root / LAUNCHER
    if path.is_file():
        match = re.search(r'^PROJECT_DISPLAY_NAME = "([^"\n]+)"', helpers.read_text(path), re.MULTILINE)
        if match and match.group(1) != "PROJECT_NAME":
            return match.group(1)
    return None


def _replace_placeholders(helpers: Any, path: Path, wanted: tuple[tuple[str, str], ...], values: dict[str, str | None]) -> str:
    text = helpers.read_text(path)
    present = [(name, source) for name, source in wanted if placeholder_pattern(name).search(text)]
    if not present:
        return "unchanged: no placeholder left"
    missing = sorted({source for _, source in present if not values[source]})
    if missing:
        what = " and ".join("the Python package (one package in src/, or a name in setup.py)" if m == "package" else "the app name" for m in missing)
        names = ", ".join(name for name, _ in present)
        return f"SKIPPED: {names} is still there, but {what} could not be found; replace it by hand. See {DOCS_URL}"
    done = []
    for name, source in present:
        value = values[source]
        assert value is not None
        text, count = placeholder_pattern(name).subn(value.replace("\\", "\\\\"), text)
        done.append(f"{name} -> {value} ({count}x)")
    helpers.write_text(path, text)
    return "updated: " + ", ".join(done)


def _create_check_script(helpers: Any, repo_root: Path) -> str:
    path = repo_root / CHECK_SCRIPT
    new = (RESOURCE_ROOT / CHECK_SCRIPT).read_text(encoding="utf-8")
    if path.is_file():
        if helpers.read_text(path).replace("\r\n", "\n") == new:
            return "already the 0.1.23 template file: unchanged"
        return "SKIPPED: the repository has its own check_placeholders.py, left as it is"
    if not path.parent.is_dir():
        return "SKIPPED: the repository has no .tools/python folder (nothing was created)"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(new)
    return "updated: created"


def _add_check_step(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lf = text.replace("\r\n", "\n")
    if "check_placeholders.py" in lf:
        return "unchanged: already runs check_placeholders.py"
    steps = list(WELCOME_STEP.finditer(lf))
    if not steps:
        return f"SKIPPED: the bump_Welcome_notebook.py command was not found (customised workflow); add the check by hand. See {DOCS_URL}"
    out, last = [], 0
    for match in steps:
        out.append(lf[last : match.end()])
        out.append(CHECK_STEP.format(indent=match.group("indent"), log=match.group("log")))
        last = match.end()
    out.append(lf[last:])
    helpers.write_text(path, "".join(out).replace("\n", newline))
    return f"updated: the check runs after bump_Welcome_notebook.py in {len(steps)} job(s)"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Replace the placeholders the initialisation missed and add the release check (idempotent, surgical, never raises for a customised file)."""
    _ = context
    helpers = _load("v0_1_9_to_v0_1_10.py", "template_sync_helpers_0_1_23")
    construct = repo_root / "construct.yaml"
    if construct.is_file() and re.search(r"^name:\s*UNDERSCORED_PROJECT_NAME\s*$", helpers.read_text(construct), re.MULTILINE):
        log("SKIPPED: this repository is not initialised (construct.yaml still names UNDERSCORED_PROJECT_NAME); nothing was changed.")
        return
    package = _load("v0_1_15_to_v0_1_16.py", "template_sync_detect_0_1_23").detect_package(helpers, repo_root)
    values = {"package": package, "name": app_name(helpers, repo_root)}
    changed = 0

    def report(name: Path, step: Callable[[], str]) -> None:
        nonlocal changed
        if not (repo_root / name).is_file() and name != CHECK_SCRIPT:
            log(f"{name}: not present in this repository (nothing was created)")
            return
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    for relative, wanted in PLACEHOLDER_FILES.items():
        report(relative, partial(_replace_placeholders, helpers, repo_root / relative, wanted, values))
    report(CHECK_SCRIPT, lambda: _create_check_script(helpers, repo_root))
    report(RELEASE_WORKFLOW, lambda: _add_check_step(helpers, repo_root / RELEASE_WORKFLOW))

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the fixes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

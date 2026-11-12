"""Template 0.1.19 -> 0.1.20: no lost pushes between the two workflows, lock files for the pip requirements, a launcher that starts fresh.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_20.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. Workflows. update_on_notebook_change.yml and update_on_src_change.yaml are replaced by the 0.1.20 files ONLY when the app's file is still
   identical to the 0.1.19 template file (line endings are ignored when comparing and kept when writing). The new files rebase their
   commit onto the latest branch and retry a rejected push, have one concurrency group each, revert a commit only when the conda test
   failed (a merge commit with -m 1, any other commit as it is), and the notebook workflow runs lock_requirements.py.
2. Lock files. .tools/python/lock_requirements.py is created (requirements*.txt -> requirements*.lock.txt with uv), `uv==<pin>` is added to
   .tools/requirements.txt, .tools/bash/test_conda_environment.sh (replaced when identical to 0.1.19) tests the lock file,
   post_install.sh / .bat install the lock file when the installer bundles one (known blocks only), and bump_constructor.py gets the
   function that bundles the lock files (inserted at two known lines).
3. Launcher. app/python_scripts/launch_jupyter.py (replaced when identical to the rendered 0.1.19 file) asks to press Enter when JupyterLab
   stops and gives JupyterLab an empty workspace folder at every launch (only the Welcome notebook opens). app/menuinst/notebook_launcher.json
   gets a Windows command that ends with `& exit`, so the terminal window closes after Enter (cmd /K kept it open).
4. Notebook usage guide. The sentence on stopping JupyterLab in .tools/docs/notebook_usage.md is replaced when it is still the 0.1.19 text.

Rules this migration keeps: surgical (only known lines or known files), idempotent (a second run reports no changes), never raises for a
customised file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_19_to_v0_1_20"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"
DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_20.md"
TAG = "[migration 0.1.20]"

NOTEBOOK_WORKFLOW = Path(".github/workflows/update_on_notebook_change.yml")
SRC_WORKFLOW = Path(".github/workflows/update_on_src_change.yaml")
TEST_CONDA = Path(".tools/bash/test_conda_environment.sh")
LAUNCHER = Path("app/python_scripts/launch_jupyter.py")
LOCK_SCRIPT = Path(".tools/python/lock_requirements.py")
TOOLS_REQUIREMENTS = Path(".tools/requirements.txt")
POST_INSTALL_SH = Path("app/bash_bat_scripts/post_install.sh")
POST_INSTALL_BAT = Path("app/bash_bat_scripts/post_install.bat")
BUMP_CONSTRUCTOR = Path(".tools/python/bump_constructor.py")
MENU = Path("app/menuinst/notebook_launcher.json")
USAGE_DOC = Path(".tools/docs/notebook_usage.md")

# Files replaced as a whole when the app's file is still the (rendered) 0.1.19 template file.
REPLACED_FILES = (NOTEBOOK_WORKFLOW, SRC_WORKFLOW, TEST_CONDA, LAUNCHER)
# Resource files that are copies of the 0.1.20 template files (a test checks they stay identical to the template).
MANAGED_FILES = (*REPLACED_FILES, LOCK_SCRIPT)

UV_PIN = "uv==0.13.3"
UV_LINE = re.compile(r"^\s*uv\s*(?:[=<>!~;\[]|$)", re.IGNORECASE)
UV_LINES = "# lock_requirements.py (requirements*.txt -> requirements*.lock.txt)\n" + UV_PIN + "\n"

# post_install.sh: (0.1.19 text, 0.1.20 text), LF.
SH_BLOCKS = (
    (
        'echo "Installing requirements from $SELECTED_REQUIREMENTS" >> "$LOG_FILE"\n'
        '"$PYTHON_EXE" -m pip install -r "$SELECTED_REQUIREMENTS" >> "$LOG_FILE" 2>&1\n',
        "# A lock file (requirements*.lock.txt next to the requirements file: every package with its exact version, written by\n"
        "# .tools/python/lock_requirements.py) is installed instead of the requirements file when the installer bundles one.\n"
        "locked_requirements() {\n"
        '    if [ -f "${1%.txt}.lock.txt" ]; then echo "${1%.txt}.lock.txt"; else echo "$1"; fi\n'
        "}\n"
        'SELECTED_REQUIREMENTS="$(locked_requirements "$SELECTED_REQUIREMENTS")"\n'
        'echo "Installing requirements from $SELECTED_REQUIREMENTS" >> "$LOG_FILE"\n'
        '"$PYTHON_EXE" -m pip install -r "$SELECTED_REQUIREMENTS" >> "$LOG_FILE" 2>&1\n',
    ),
    (
        '        "$PYTHON_EXE" -m pip install -r "$PROJECT_ROOT/requirements-macos.txt" >> "$LOG_FILE" 2>&1\n',
        '        "$PYTHON_EXE" -m pip install -r "$(locked_requirements "$PROJECT_ROOT/requirements-macos.txt")" >> "$LOG_FILE" 2>&1\n',
    ),
    (
        '        "$PYTHON_EXE" -m pip install -r "$PROJECT_ROOT/requirements-linux.txt" >> "$LOG_FILE" 2>&1\n',
        '        "$PYTHON_EXE" -m pip install -r "$(locked_requirements "$PROJECT_ROOT/requirements-linux.txt")" >> "$LOG_FILE" 2>&1\n',
    ),
)
SH_DONE = "locked_requirements()"

BAT_BLOCKS = (
    (
        'echo Installing requirements from "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%"\n'
        '"%PYTHON_EXE%" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%" 2>&1\n',
        "REM A lock file (requirements*.lock.txt next to the requirements file: every package with its exact version, written by\n"
        "REM .tools/python/lock_requirements.py) is installed instead of the requirements file when the installer bundles one.\n"
        'SET "SELECTED_LOCK=%SELECTED_REQUIREMENTS:~0,-4%.lock.txt"\n'
        'IF EXIST "%SELECTED_LOCK%" SET "SELECTED_REQUIREMENTS=%SELECTED_LOCK%"\n'
        'echo Installing requirements from "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%"\n'
        '"%PYTHON_EXE%" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%" 2>&1\n',
    ),
    (
        'IF EXIST "%PROJECT_ROOT%\\requirements-windows.txt" (\n'
        '    echo Installing Windows-specific requirements. >> "%LOG_FILE%"\n'
        '    "%PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%\\requirements-windows.txt" >> "%LOG_FILE%" 2>&1\n',
        'SET "WINDOWS_REQUIREMENTS=%PROJECT_ROOT%\\requirements-windows.txt"\n'
        'IF EXIST "%PROJECT_ROOT%\\requirements-windows.lock.txt" SET "WINDOWS_REQUIREMENTS=%PROJECT_ROOT%\\requirements-windows.lock.txt"\n'
        'IF EXIST "%PROJECT_ROOT%\\requirements-windows.txt" (\n'
        '    echo Installing Windows-specific requirements from "%WINDOWS_REQUIREMENTS%". >> "%LOG_FILE%"\n'
        '    "%PYTHON_EXE%" -m pip install -r "%WINDOWS_REQUIREMENTS%" >> "%LOG_FILE%" 2>&1\n',
    ),
)
BAT_DONE = "SELECTED_LOCK="

BUMP_FUNCTION_ANCHOR = "def ensure_extra_files("
BUMP_CALL_OLD = "    ensure_requirements_in_extra_files(data)\n"
BUMP_CALL_NEW = "    ensure_requirements_in_extra_files(data)\n    ensure_lock_files_in_extra_files(data)\n"
BUMP_DONE = "def ensure_lock_files_in_extra_files("
BUMP_NEEDS = ("def extract_project_folder(", "def debug(", "from pathlib import Path")

WIN_EXIT = ["&", "exit"]

USAGE_OLD = (
    "When you have, click on the terminal window and press `Ctrl+C` to stop the Jupyter Lab session and close the terminal."
)
USAGE_NEW = (
    "When you have, stop the Jupyter Lab session with **File > Shut Down** in Jupyter Lab (or click on the terminal window and press `Ctrl+C`). "
    "The terminal then asks you to press `Enter`, and closes."
)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers() -> Any:
    """Reuse the helpers of the 0.1.10 migration (token replacement, newline detection, file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_20", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resource(root: Path, relative: Path) -> str:
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(f"Template resource not found: {source}")
    return source.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _render(helpers: Any, text: str, replacements: dict[str, str]) -> str:
    return str(helpers.render_template(text, replacements, "\n"))


def _write_lf(helpers: Any, path: Path, text_lf: str, newline: str) -> None:
    helpers.write_text(path, text_lf.replace("\n", newline))


# --------------------------------------------------------------------------------------------------------------------------------
# Whole files (workflows, conda test, launcher)
# --------------------------------------------------------------------------------------------------------------------------------
def _replace_file(helpers: Any, repo_root: Path, relative: Path, replacements: dict[str, str]) -> str:
    path = repo_root / relative
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(path)
    current_lf = current.replace("\r\n", "\n")
    new = _render(helpers, _resource(RESOURCE_ROOT, relative), replacements)
    if current_lf == new:
        return "already the 0.1.20 template file: unchanged"
    if current_lf != _render(helpers, _resource(PREVIOUS_ROOT, relative), replacements):
        return f"SKIPPED: customised (not the 0.1.19 template file), so it was left unchanged; apply the 0.1.20 changes by hand, see {DOCS_URL}"
    _write_lf(helpers, path, new, helpers.detect_newline(current))
    return "updated: replaced by the 0.1.20 template file"


# --------------------------------------------------------------------------------------------------------------------------------
# Lock files
# --------------------------------------------------------------------------------------------------------------------------------
def _create_lock_script(helpers: Any, repo_root: Path) -> str:
    path = repo_root / LOCK_SCRIPT
    new = _resource(RESOURCE_ROOT, LOCK_SCRIPT)
    if path.is_file():
        if helpers.read_text(path).replace("\r\n", "\n") == new:
            return "already the 0.1.20 template file: unchanged"
        return "SKIPPED: the repository has its own lock_requirements.py, left as it is"
    if not path.parent.is_dir():
        return "SKIPPED: the repository has no .tools/python folder (nothing was created)"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(new)
    return "updated: created"


def _add_uv(helpers: Any, repo_root: Path) -> str:
    path = repo_root / TOOLS_REQUIREMENTS
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(path)
    if any(UV_LINE.match(line) for line in current.splitlines()):
        return "already installs uv: unchanged"
    newline = helpers.detect_newline(current)
    text = current if not current or current.endswith(("\n", "\r")) else current + newline
    helpers.write_text(path, text + UV_LINES.replace("\n", newline))
    return f"updated: added {UV_PIN}"


def _replace_blocks(helpers: Any, path: Path, blocks: tuple[tuple[str, str], ...], done: str, what: str) -> str:
    current = helpers.read_text(path)
    lf = current.replace("\r\n", "\n")
    if done in lf:
        return f"already {what}: unchanged"
    missing = [old.splitlines()[0].strip() for old, _ in blocks if lf.count(old) != 1]
    if missing:
        shown = "; ".join(missing)
        return f"SKIPPED: customised, the known lines were not found once ({shown}); left unchanged, apply the change by hand, see {DOCS_URL}"
    for old, new in blocks:
        lf = lf.replace(old, new)
    _write_lf(helpers, path, lf, helpers.detect_newline(current))
    return f"updated: {what}"


def _bump_constructor(helpers: Any, repo_root: Path) -> str:
    path = repo_root / BUMP_CONSTRUCTOR
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(path)
    lf = current.replace("\r\n", "\n")
    if BUMP_DONE in lf:
        return "already bundles the lock files: unchanged"
    if lf.count(BUMP_FUNCTION_ANCHOR) != 1 or lf.count(BUMP_CALL_OLD) != 1 or not all(need in lf for need in BUMP_NEEDS):
        return f"SKIPPED: customised (the known lines were not found once), left unchanged; add ensure_lock_files_in_extra_files by hand, see {DOCS_URL}"
    lf = lf.replace(BUMP_FUNCTION_ANCHOR, BUMP_SNIPPET + BUMP_FUNCTION_ANCHOR, 1).replace(BUMP_CALL_OLD, BUMP_CALL_NEW, 1)
    _write_lf(helpers, path, lf, helpers.detect_newline(current))
    return "updated: bundles the lock files (ensure_lock_files_in_extra_files)"


# The LOCK_FILES tuple and ensure_lock_files_in_extra_files of the 0.1.20 bump_constructor.py, with the blank lines before the next function
# (a test checks that the template script holds this text once).
BUMP_SNIPPET = '''LOCK_FILES = (
    "requirements.lock.txt",
    "requirements_gpu.lock.txt",
    "requirements-linux.lock.txt",
    "requirements-windows.lock.txt",
    "requirements-macos.lock.txt",
)


def ensure_lock_files_in_extra_files(construct_data: dict) -> None:
    """Bundle the lock files (requirements*.lock.txt, written by lock_requirements.py) next to their requirements files.

    post_install installs a lock file instead of its requirements file when the installer bundles one. A mapping whose lock file is gone
    (the requirements could not be locked) is removed, because constructor fails on a missing file.
    """
    extra_files = construct_data.setdefault("extra_files", [])
    project_folder = extract_project_folder(extra_files)
    for lock_file in LOCK_FILES:
        mapped = [item for item in extra_files if isinstance(item, dict) and lock_file in item]
        if Path(lock_file).exists():
            if not mapped:
                extra_files.append({lock_file: f"{project_folder}/{lock_file}"})
                debug(f"Added mapping for {lock_file}")
            else:
                debug(f"{lock_file} mapping already present")
        elif mapped:
            for item in mapped:
                extra_files.remove(item)
            debug(f"Removed the mapping of {lock_file}: the file does not exist (the installer installs the requirements file instead)")


'''


# --------------------------------------------------------------------------------------------------------------------------------
# Launcher shortcut (Windows: close the terminal after Enter)
# --------------------------------------------------------------------------------------------------------------------------------
def _menu(helpers: Any, repo_root: Path) -> str:
    path = repo_root / MENU
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(path)
    lf = current.replace("\r\n", "\n")
    try:
        data = json.loads(lf)
    except json.JSONDecodeError as exc:
        return f"SKIPPED: not valid JSON ({exc}), left unchanged"
    items = data.get("menu_items") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        return f"SKIPPED: not the template shape (one menu item), left unchanged; see {DOCS_URL}"
    item = items[0]
    command = item.get("command")
    win = (item.get("platforms") or {}).get("win")
    if not isinstance(command, list) or not isinstance(win, dict) or not item.get("terminal"):
        return f"SKIPPED: not the template shape (a terminal command with a win platform), left unchanged; see {DOCS_URL}"
    if "command" in win:
        if win["command"] == command + WIN_EXIT:
            return "already closes the Windows terminal after Enter: unchanged"
        return "SKIPPED: has its own Windows command, left unchanged; end it with \"&\", \"exit\" to close the terminal after Enter"
    if json.dumps(data, indent=2, ensure_ascii=False) + "\n" != lf:
        return f"SKIPPED: formatted differently from the template, left unchanged; add the Windows command by hand, see {DOCS_URL}"
    win["command"] = command + WIN_EXIT
    _write_lf(helpers, path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", helpers.detect_newline(current))
    return "updated: the Windows command ends with `& exit` (the terminal closes after Enter)"


def _usage_doc(helpers: Any, repo_root: Path) -> str:
    path = repo_root / USAGE_DOC
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(path)
    if USAGE_NEW in current:
        return "unchanged"
    if current.count(USAGE_OLD) != 1:
        return "SKIPPED: the 0.1.19 sentence on stopping Jupyter Lab was not found (customised), left unchanged"
    helpers.write_text(path, current.replace(USAGE_OLD, USAGE_NEW))
    return "updated: how to stop Jupyter Lab (File > Shut Down, then Enter)"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Replace the two workflows and the launcher, add the lock files (idempotent, surgical, never raises for a customised file)."""
    _ = context
    helpers = _load_helpers()
    try:
        replacements = helpers.load_replacements(repo_root)
    except (OSError, ValueError) as exc:
        log(f"could not read the app name from construct.yaml ({exc}): the launcher is compared without app names.")
        replacements = {}

    changed = 0

    def report(name: Path, step: Any) -> None:
        nonlocal changed
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    for relative in REPLACED_FILES:
        report(relative, lambda relative=relative: _replace_file(helpers, repo_root, relative, replacements))
    report(LOCK_SCRIPT, lambda: _create_lock_script(helpers, repo_root))
    report(TOOLS_REQUIREMENTS, lambda: _add_uv(helpers, repo_root))
    for relative, blocks, done in ((POST_INSTALL_SH, SH_BLOCKS, SH_DONE), (POST_INSTALL_BAT, BAT_BLOCKS, BAT_DONE)):
        if not (repo_root / relative).is_file():
            log(f"{relative}: not present in this repository (nothing was created)")
            continue
        report(relative, lambda relative=relative, blocks=blocks, done=done: _replace_blocks(
            helpers, repo_root / relative, blocks, done, "installs the lock file when the installer bundles one"
        ))
    report(BUMP_CONSTRUCTOR, lambda: _bump_constructor(helpers, repo_root))
    report(MENU, lambda: _menu(helpers, repo_root))
    report(USAGE_DOC, lambda: _usage_doc(helpers, repo_root))

    if changed:
        log(
            f"done: {changed} file(s) changed. Review the pull request before merging. The lock files are written by the next run of "
            "'Updates on notebook change' (Actions > Run workflow); the launcher changes reach users with the next release of the app."
        )
    else:
        log("done: no repository changes were required.")

"""Template 0.1.15 -> 0.1.16: add the LabConstrictor Tools bridge hooks to an app repository.

What this changes (in the app's *repository*, through the template sync pull request; an installed copy of the app is
only ever changed by installing a newer installer, see
https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_16.md):

* app/bash_bat_scripts/post_install.sh / .bat   register the app with the Tools bridge after install
* app/bash_bat_scripts/pre_uninstall.sh / .bat  unregister it on uninstall
* app/bash_bat_scripts/uninstall.sh             report a failed pre-uninstall or a failed removal
* .tools/docs/tools_bridge.md and its entry in .tools/docs/README.md
* the App Centre manifest check (.tools/schema/labconstrictor-app.schema.json, .tools/python/check_app_manifest.py, .tools/docs/app_manifest.md,
  .github/workflows/check_app_manifest.yml), whose 0.1.15 schema copy was stale; labconstrictor-app.yaml itself is never touched

Rules this migration keeps:

* It edits surgically: a block is inserted before a known line of the app's own script, so changes the app made to its scripts
  stay. A script that already has the hook (it mentions labconstrictor_tools) is left unchanged, so running it twice, or on an
  app that carried the hooks before (bridge-test apps), changes nothing.
* A hook that is already there (apps that got it from the bridge-test branch) is not re-inserted. Only if its Tools source is exactly the
  old unpinned `.../LabConstrictor-Tools/archive/refs/heads/main.zip` is that source rewritten to the pinned tag form of the template
  (an LC_TOOLS_PIN line plus the tag URL); any other text is left alone with a SKIPPED line that says what to do by hand.
* A script that is missing or too different to edit safely is skipped with a message that says what to do by hand. The
  migration never raises for that reason: the sync pull request must still be created.
* Nothing is deleted. Only the files listed above are written; notebooks, src/, requirements, construct.yaml and
  labconstrictor-app.yaml are not touched. Line endings of each file are kept.
* The hooks are inert for an app without a <package>_lc_tools module (post_install finds no module and skips the step;
  pre_uninstall only warns when a registry entry exists), so the migration is applied to every app.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_15_to_v0_1_16"
SCRIPTS = Path("app/bash_bat_scripts")
POST_INSTALL_SH = SCRIPTS / "post_install.sh"
POST_INSTALL_BAT = SCRIPTS / "post_install.bat"
PRE_UNINSTALL_SH = SCRIPTS / "pre_uninstall.sh"
PRE_UNINSTALL_BAT = SCRIPTS / "pre_uninstall.bat"
UNINSTALL_SH = SCRIPTS / "uninstall.sh"
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
DOCS_INDEX = Path(".tools/docs/README.md")

# Resource files: copies of the template files of 0.1.16 (a test checks they stay identical to the template).
# Scripts are used as the source of the hook blocks; the doc is copied.
# Template-owned App Centre manifest check (as in 0.1.15). The 0.1.15 copy of the schema rejected the fields inputs, outputs, limitations,
# status and distribution.platforms. labconstrictor-app.yaml itself belongs to the app and is never created, changed or removed here.
MANIFEST_CHECK_FILES = (
    Path(".tools/schema/labconstrictor-app.schema.json"),
    Path(".tools/python/check_app_manifest.py"),
    Path(".tools/docs/app_manifest.md"),
    Path(".github/workflows/check_app_manifest.yml"),
)
MANAGED_FILES = (POST_INSTALL_SH, POST_INSTALL_BAT, PRE_UNINSTALL_SH, PRE_UNINSTALL_BAT, UNINSTALL_SH, TOOLS_DOC) + MANIFEST_CHECK_FILES

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_16.md"
TOOLS_ARCHIVE = "https://github.com/CellMigrationLab/LabConstrictor-Tools/archive/refs/"
UNPINNED_URL = TOOLS_ARCHIVE + "heads/main.zip"
INDEX_ENTRY = "- [Expose tools to Napari and Fiji (experimental)](tools_bridge.md)"
HOOK_MARKER = "labconstrictor_tools"
TAG = "[migration 0.1.16]"

# uninstall.sh of template 0.1.15 and the replacement of 0.1.16 (exact text only: a changed uninstall.sh is left alone).
OLD_UNINSTALL = 'if [ -f "$PREFIX/pre_uninstall.sh" ]; then\n    bash "$PREFIX/pre_uninstall.sh"\nfi\nrm -rf "$PREFIX"\n'
NEW_UNINSTALL = (
    'if [ -f "$PREFIX/pre_uninstall.sh" ]; then\n'
    "    # best effort: shortcut/registry cleanup must not stop the removal, but a failure must be visible\n"
    '    if ! bash "$PREFIX/pre_uninstall.sh"; then\n'
    '        echo "WARNING: pre-uninstall cleanup failed (see the messages above); shortcuts or the tools registry entry may remain." >&2\n'
    "    fi\n"
    "fi\n"
    'if ! rm -rf -- "$PREFIX"; then\n'
    '    echo "ERROR: could not remove $PREFIX; PROJECT_NAME is only partly uninstalled. Close anything using it and delete that folder manually." >&2\n'
    "    exit 1\n"
    "fi\n"
)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers():
    """Reuse the rendering helpers of the 0.1.10 migration (token replacement, newline detection, file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_16", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resource_lines(helpers: Any, relative: Path, replacements: dict[str, str]) -> list[str]:
    source = RESOURCE_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Template resource not found: {source}")
    return helpers.render_template(helpers.read_text(source), replacements, "\n").split("\n")


def _find(lines: list[str], predicate: Callable[[str], bool], start: int = 0) -> int | None:
    for index in range(start, len(lines)):
        if predicate(lines[index]):
            return index
    return None


def _block(resource: list[str], starts: Callable[[str], bool], ends: Callable[[str], bool], lead: int = 0) -> list[str]:
    """Lines of the resource from the line where `starts` is true (minus `lead` lines before it) up to, not including, the line where `ends` is true."""
    first = _find(resource, starts)
    last = _find(resource, ends, (first or 0) + 1) if first is not None else None
    if first is None or last is None:
        raise ValueError("the hook block was not found in the template resource")
    return resource[first - lead : last]


def _insert_before(helpers: Any, path: Path, block: list[str], anchor: Callable[[str], bool], required: list[str]) -> str:
    """Insert `block` before the first line of the file at `path` for which `anchor` is true. Returns a status text."""
    text = helpers.read_text(path)
    if HOOK_MARKER in text:
        return "already has the Tools bridge hook: unchanged"
    missing = [item for item in required if item not in text]
    if missing:
        return f"SKIPPED: the script does not define {', '.join(missing)}, so the hook cannot be added safely; add it by hand from .tools/docs/tools_bridge.md"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    index = _find(lines, anchor)
    if index is None:
        return "SKIPPED: the line the hook goes before was not found (the script was customised); add the hook by hand from .tools/docs/tools_bridge.md"
    if lines[index - 1 : index] and not lines[index - 1].endswith(("\n", "\r")):
        lines[index - 1] += newline
    lines[index:index] = [line + newline for line in block]
    helpers.write_text(path, "".join(lines))
    return "hook added"


def _template_pin(resource: list[str]) -> str:
    """The tag the template scripts pin Tools to (read from the template resource, so the two cannot drift apart)."""
    for line in resource:
        match = re.match(r'^(?:LC_TOOLS_PIN="|SET "LC_TOOLS_PIN=)([^"]+)"$', line.strip())
        if match:
            return match.group(1)
    raise ValueError("LC_TOOLS_PIN was not found in the template resource")


def _repin(helpers: Any, path: Path, resource: list[str], batch: bool) -> str:
    """For a script that already has the hook: rewrite the known unpinned Tools source to the pinned tag form of the template.

    Only the exact old text is rewritten (the `${LC_TOOLS_SPEC:-<url>}` default in post_install.sh, the `IF NOT DEFINED LC_TOOLS_SPEC SET`
    line in post_install.bat); a pin line is added next to it. Anything else is left unchanged. Idempotent, line endings kept."""
    unchanged = "already has the Tools bridge hook: unchanged"
    text = helpers.read_text(path)
    pin = _template_pin(resource)
    if UNPINNED_URL not in text:
        if "LC_TOOLS_PIN" in text or "LabConstrictor-Tools/archive/" not in text:
            return unchanged
        return f"{unchanged}. SKIPPED the pin: its Tools source is not the known unpinned main.zip (customised); keep it or set a tag by hand, see {DOCS_URL}"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    if batch:
        exact = 'IF NOT DEFINED LC_TOOLS_SPEC SET "LC_TOOLS_SPEC=' + UNPINNED_URL + '"'
        hits = [i for i, l in enumerate(lines) if l.strip() == exact]
        new_source = exact.replace(UNPINNED_URL, TOOLS_ARCHIVE + "tags/%LC_TOOLS_PIN%.zip")
        pin_lines = [
            "REM Pinned release tag of labconstrictor-tools (moving the pin: change only this tag).",
            f'SET "LC_TOOLS_PIN={pin}"',
        ]
        anchor_prefix = None
    else:
        exact = "${LC_TOOLS_SPEC:-" + UNPINNED_URL + "}"
        hits = [i for i, l in enumerate(lines) if exact in l]
        new_source = exact.replace(UNPINNED_URL, TOOLS_ARCHIVE + "tags/${LC_TOOLS_PIN}.zip")
        pin_lines = [
            "# Pinned release tag of labconstrictor-tools (moving the pin: change only this tag).",
            f'LC_TOOLS_PIN="{pin}"',
        ]
        anchor_prefix = "LC_TOOLS_MODULE="
    if len(hits) != 1 or text.count(UNPINNED_URL) != 1:
        return f"{unchanged}. SKIPPED the pin: the unpinned Tools source is not in the known form (customised); replace refs/heads/main.zip by refs/tags/{pin}.zip by hand, see {DOCS_URL}"
    index = hits[0]
    insert_at = index
    if anchor_prefix:
        anchors = [i for i, l in enumerate(lines) if l.startswith(anchor_prefix)]
        if len(anchors) != 1 or anchors[0] > index:
            return (f"{unchanged}. SKIPPED the pin: no single {anchor_prefix} line before the pip install line (customised); add "
                    f'LC_TOOLS_PIN="{pin}" and use refs/tags/{pin}.zip by hand, see {DOCS_URL}')
        insert_at = anchors[0]
    lines[index] = lines[index].replace(exact, new_source)
    if lines[insert_at - 1 : insert_at] and not lines[insert_at - 1].endswith(("\n", "\r")):
        lines[insert_at - 1] += newline
    lines[insert_at:insert_at] = [l + newline for l in pin_lines]
    helpers.write_text(path, "".join(lines))
    return f"updated: unpinned Tools source (refs/heads/main.zip) rewritten to the pinned tag {pin}"


def _repin_only(helpers: Any, path: Path, resource: list[str], batch: bool) -> str:
    if HOOK_MARKER not in helpers.read_text(path):
        return "SKIPPED: no hook added (the app's Python package is unknown); add it by hand from .tools/docs/tools_bridge.md"
    return _repin(helpers, path, resource, batch)


def _migrate_post_install_sh(helpers: Any, repo_root: Path, resource: list[str]) -> str:
    path = repo_root / POST_INSTALL_SH
    block = _block(resource, lambda l: l.startswith("# --- Optional: expose the app's tools"), lambda l: "include_path.py" in l and "notebook_launcher.json" in l, lead=1)
    status = _insert_before(
        helpers, path, block,
        lambda l: "include_path.py" in l and "notebook_launcher.json" in l,
        ['PYTHON_EXE="', 'PROJECT_ROOT="', 'LOG_FILE="'],
    )
    return _repin(helpers, path, resource, batch=False) if status.startswith("already has") else status


def _migrate_post_install_bat(helpers: Any, repo_root: Path, resource: list[str]) -> str:
    path = repo_root / POST_INSTALL_BAT
    block = _block(resource, lambda l: l.startswith("REM Optional: expose the app's tools"), lambda l: "include_path.py" in l and "notebook_launcher.json" in l)
    status = _insert_before(
        helpers, path, block,
        lambda l: "include_path.py" in l and "notebook_launcher.json" in l,
        ['SET "PYTHON_EXE=', 'SET "PROJECT_ROOT=', 'SET "LOG_FILE='],
    )
    if status.startswith("already has"):
        return _repin(helpers, path, resource, batch=True)
    if status == "hook added":
        # The hook reads LC_TOOLS_SPEC and the version into variables: keep a "!" in a value from being re-interpreted.
        text = helpers.read_text(path)
        old, new = "SETLOCAL EnableExtensions", "SETLOCAL EnableExtensions DisableDelayedExpansion"
        newline = helpers.detect_newline(text)
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.strip().upper() == old.upper():
                lines[index] = new + (newline if line.endswith(("\n", "\r")) else "")
                helpers.write_text(path, "".join(lines))
                status += " (SETLOCAL now DisableDelayedExpansion)"
                break
    return status


def _migrate_pre_uninstall(helpers: Any, repo_root: Path, resource: list[str], relative: Path, starts: Callable[[str], bool]) -> str:
    block = _block(resource, starts, lambda l: "menuinst.api import remove" in l)
    return _insert_before(
        helpers, repo_root / relative, block,
        lambda l: "menuinst.api import remove" in l,
        ["BASE_PATH"] if relative.suffix == ".sh" else [],
    )


def _migrate_uninstall_sh(helpers: Any, repo_root: Path, replacements: dict[str, str]) -> str:
    path = repo_root / UNINSTALL_SH
    text = helpers.read_text(path)
    newline = helpers.detect_newline(text)
    normalised = text.replace("\r\n", "\n")
    new = helpers.render_template(NEW_UNINSTALL, replacements, "\n")
    if new in normalised:
        return "already up to date: unchanged"
    if OLD_UNINSTALL not in normalised:
        return "SKIPPED: uninstall.sh differs from the template of 0.1.15 (customised); left unchanged. Optional: the new text only reports a failed pre-uninstall or a failed removal, see " + DOCS_URL
    helpers.write_text(path, normalised.replace(OLD_UNINSTALL, new, 1).replace("\n", newline))
    return "updated (pre-uninstall failures and a failed removal are reported)"


def _migrate_docs(helpers: Any, repo_root: Path, replacements: dict[str, str]) -> list[str]:
    results = []
    source = RESOURCE_ROOT / TOOLS_DOC
    destination = repo_root / TOOLS_DOC
    rendered = helpers.render_template(helpers.read_text(source), replacements, helpers.detect_newline(helpers.read_text(destination)) if destination.exists() else "\n")
    if destination.exists() and helpers.read_text(destination) == rendered:
        results.append(f"{TOOLS_DOC}: unchanged")
    else:
        helpers.write_text(destination, rendered)
        results.append(f"{TOOLS_DOC}: written")

    index_path = repo_root / DOCS_INDEX
    if not index_path.is_file():
        results.append(f"{DOCS_INDEX}: not present, index entry skipped")
        return results
    text = helpers.read_text(index_path)
    if "tools_bridge.md" in text:
        results.append(f"{DOCS_INDEX}: entry already present, unchanged")
        return results
    entry = INDEX_ENTRY
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    anchor = _find(lines, lambda l: "external_code_upload.md" in l)
    if entry is None or anchor is None:
        results.append(f"{DOCS_INDEX}: SKIPPED (no place for the entry; the page is still reachable as tools_bridge.md)")
        return results
    if not lines[anchor].endswith(("\n", "\r")):
        lines[anchor] += newline
    lines.insert(anchor + 1, entry + newline)
    helpers.write_text(index_path, "".join(lines))
    results.append(f"{DOCS_INDEX}: entry added")
    return results


def detect_package(helpers: Any, repo_root: Path) -> str | None:
    """The app's Python package name. The 0.1.10 helper gives up when src/ holds two packages, which is exactly the case of an
    app that has a <package>_lc_tools module next to its package (and apps may have several shim packages), so the name of the
    single <package>_lc_tools module wins, then the single package that is not a tools module, then the 0.1.10 helper (setup.py), then the name in pyproject.toml."""
    src = repo_root / "src"
    if src.is_dir():
        # The tools module is the one thing the hook registers, so its name is the best answer when there is exactly one.
        tools = []
        for child in src.iterdir():
            module = child.name if child.is_dir() and (child / "__init__.py").is_file() else child.stem if child.suffix == ".py" else ""
            if module.endswith("_lc_tools") and len(module) > len("_lc_tools"):
                tools.append(module[: -len("_lc_tools")])
        tools = sorted(set(tools))
        if len(tools) == 1:
            return tools[0]
        candidates = sorted(
            child.name
            for child in src.iterdir()
            if child.is_dir()
            and child.name.isidentifier()
            and not child.name.endswith("_lc_tools")
            and (child / "__init__.py").is_file()
        )
        if len(candidates) == 1:
            return candidates[0]
    name = helpers.detect_python_package_name(repo_root)
    if name != "PYTHON_PROJ_NAME":
        return name
    return _package_from_pyproject(repo_root)


def _package_from_pyproject(repo_root: Path) -> str | None:
    """Last resort (an app with several packages in src/ and a setup.py shim without a name): the [project] name of pyproject.toml,
    hyphens turned into underscores, accepted only if src/<that name>/ exists. The 0.1.10 helper does not read pyproject.toml."""
    path = repo_root / "pyproject.toml"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    name = None
    try:
        import tomllib  # Python 3.11+

        project = tomllib.loads(text).get("project", {})
        name = project.get("name") if isinstance(project, dict) else None
    except ImportError:
        section = re.search(r"^\[project\]\s*$(?P<body>.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
        match = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', section.group("body"), re.MULTILINE) if section else None
        name = match.group(1) if match else None
    except ValueError:  # tomllib.TOMLDecodeError: a pyproject.toml that does not parse decides nothing
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    package = name.strip().replace("-", "_")
    if package.isidentifier() and (repo_root / "src" / package).is_dir():
        return package
    return None


def _migrate_manifest_check(repo_root: Path) -> list[str]:
    """Bring the template-owned manifest check files to this version (byte copies; unchanged files are not rewritten)."""
    results = []
    for relative in MANIFEST_CHECK_FILES:
        source = RESOURCE_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Template resource not found: {source}")
        destination = repo_root / relative
        data = source.read_bytes()
        if destination.is_file() and destination.read_bytes() == data:
            results.append(f"{relative}: unchanged")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        results.append(f"{relative}: written")
    return results


def has_tools_module(repo_root: Path) -> bool:
    src = repo_root / "src"
    if not src.is_dir():
        return False
    return any(src.glob("*_lc_tools/__init__.py")) or any(src.glob("*_lc_tools.py"))


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Add the Tools bridge hooks to the installer scripts of the repository (idempotent, never deletes, never raises for a customised file)."""
    _ = context
    helpers = _load_helpers()
    try:
        replacements = helpers.load_replacements(repo_root)
    except (OSError, ValueError) as exc:
        log(f"SKIPPED: could not read the app name from construct.yaml ({exc}); nothing was changed.")
        return

    package = detect_package(helpers, repo_root)
    if package:
        replacements["PYTHON_PROJ_NAME"] = package
    else:
        log("could not tell the app's Python package from src/, setup.py or pyproject.toml: the post_install hook is skipped (add it by hand), the other files are migrated.")

    log("app has a <package>_lc_tools module: the hooks will register it." if has_tools_module(repo_root)
        else "app has no <package>_lc_tools module: the hooks are added but stay inert until you add one (see .tools/docs/tools_bridge.md).")

    steps: list[tuple[Path, Callable[[], str]]] = []
    resources = {path: _resource_lines(helpers, path, replacements) for path in (POST_INSTALL_SH, POST_INSTALL_BAT, PRE_UNINSTALL_SH, PRE_UNINSTALL_BAT)}
    if package:  # without the package name the block would carry an unreplaced token and silently do nothing
        steps.append((POST_INSTALL_SH, lambda: _migrate_post_install_sh(helpers, repo_root, resources[POST_INSTALL_SH])))
        steps.append((POST_INSTALL_BAT, lambda: _migrate_post_install_bat(helpers, repo_root, resources[POST_INSTALL_BAT])))
    else:  # the hook is not added, but a hook that is already there still gets its pin
        steps.append((POST_INSTALL_SH, lambda: _repin_only(helpers, repo_root / POST_INSTALL_SH, resources[POST_INSTALL_SH], False)))
        steps.append((POST_INSTALL_BAT, lambda: _repin_only(helpers, repo_root / POST_INSTALL_BAT, resources[POST_INSTALL_BAT], True)))
    steps.append((PRE_UNINSTALL_SH, lambda: _migrate_pre_uninstall(helpers, repo_root, resources[PRE_UNINSTALL_SH], PRE_UNINSTALL_SH, lambda l: l.startswith("# best effort: a failure must not stop"))))
    steps.append((PRE_UNINSTALL_BAT, lambda: _migrate_pre_uninstall(helpers, repo_root, resources[PRE_UNINSTALL_BAT], PRE_UNINSTALL_BAT, lambda l: "labconstrictor_tools unregister" in l)))
    steps.append((UNINSTALL_SH, lambda: _migrate_uninstall_sh(helpers, repo_root, replacements)))

    changed = 0
    for relative, step in steps:
        if not (repo_root / relative).is_file():
            log(f"{relative}: SKIPPED, file not found in this repository (nothing was created).")
            continue
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}"
        if status.startswith(("hook added", "updated")):
            changed += 1
        log(f"{relative}: {status}")

    try:
        for line in _migrate_docs(helpers, repo_root, replacements):
            if line.endswith(("written", "entry added")):
                changed += 1
            log(line)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log(f"docs SKIPPED: {exc}")

    try:
        for line in _migrate_manifest_check(repo_root):
            if line.endswith("written"):
                changed += 1
            log(line)
    except OSError as exc:
        log(f"manifest check files SKIPPED: {exc}")

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the tools register after a NEW installer of the app is installed.")
    else:
        log("done: no repository changes were required.")

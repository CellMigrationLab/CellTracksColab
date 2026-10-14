"""Template 0.1.16 -> 0.1.17: install labconstrictor-tools BEFORE the import check of the app package in post_install.

Problem: post_install of 0.1.16 ran `import <package>` right after installing the package and only afterwards installed
labconstrictor-tools, so a package that imports `labconstrictor_tools` at module level failed with ModuleNotFoundError and the installer
aborted. Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_17.md

New order in app/bash_bat_scripts/post_install.sh / .bat: install the package -> install Tools (only if the package ships a
<package>_lc_tools module; a failure is a logged WARNING) -> import check -> register the app (only if the Tools install worked).

What this changes (only the two post_install scripts of the app repository, plus the Tools guide):

* a block that installs Tools is inserted before the import check line (sh) / a `CALL :install_lc_tools` is inserted there and the
  subroutine is appended to the file (bat; a subroutine because %VAR% in a parenthesised block is expanded when the block is read);
* the lines that defined the pin / module / Tools source in the registration block are MOVED into that block (a one-line comment stays),
  with the app's own values (its module name, its pin, its Tools source: nothing is replaced by a template default);
* the single `pip install` line of the registration block becomes a check of LC_TOOLS_READY.
* .tools/docs/tools_bridge.md is refreshed (it still carries the 0.1.16 text in the app repository).

Rules this migration keeps:

* Surgical: only lines of the known hook are touched; a script that does not have the known shape is skipped with a message that says what
  to do by hand (it never raises: the sync pull request must still be created); other customisations stay.
* Idempotent: a script that already mentions LC_TOOLS_READY is left unchanged. Nothing is deleted except the moved lines.
* A script without the Tools hook is left alone (the 0.1.16 migration adds the hook, in the old position, when the chain runs both,
  and this migration then moves it). Line endings of each file are kept.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_16_to_v0_1_17"
SCRIPTS = Path("app/bash_bat_scripts")
POST_INSTALL_SH = SCRIPTS / "post_install.sh"
POST_INSTALL_BAT = SCRIPTS / "post_install.bat"
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
# Resource files: copies of the template files of 0.1.17 (a test checks they stay identical to the template).
MANAGED_FILES = (POST_INSTALL_SH, POST_INSTALL_BAT, TOOLS_DOC)

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_17.md"
TAG = "[migration 0.1.17]"
DONE_MARKER = "LC_TOOLS_READY"
UNCHANGED = "already installs Tools before the import check: unchanged"

# Tools source of the template (as in the resources): replaced by the app's own source when that differs.
SH_TEMPLATE_SPEC = '"${LC_TOOLS_SPEC:-https://github.com/CellMigrationLab/LabConstrictor-Tools/archive/refs/tags/${LC_TOOLS_PIN}.zip}"'
BAT_PIP_LINE = (
    '"%PYTHON_EXE%" -c "import os, subprocess, sys; sys.exit(subprocess.call([sys.executable, \'-m\', \'pip\', \'install\', '
    "os.environ['LC_TOOLS_SPEC']]))\" >> \"%LOG_FILE%\" 2>&1"
)
# Comment lines of the hook header that move together with the variables they describe.
SH_MOVED_COMMENT = re.compile(r"^# (Default source|because the package|is not on PyPI|Pinned release tag)")
BAT_MOVED_COMMENT = re.compile(r"^REM (Default source|because the package|is not on PyPI|Pinned release tag)")


class Unsupported(Exception):
    """The script does not have the shape this migration knows: it is skipped with this message."""


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers():
    """Reuse the rendering helpers of the 0.1.10 migration (token replacement, newline detection, file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_17", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find(lines: list[str], predicate: Callable[[str], bool], start: int = 0, stop: int | None = None) -> int | None:
    for index in range(start, len(lines) if stop is None else stop):
        if predicate(lines[index]):
            return index
    return None


def _only(lines: list[str], predicate: Callable[[str], bool], what: str, start: int = 0, optional: bool = False) -> int | None:
    hits = [i for i in range(start, len(lines)) if predicate(lines[i])]
    if len(hits) == 1:
        return hits[0]
    if not hits and optional:
        return None
    raise Unsupported(f"{what} was {'not found' if not hits else 'found more than once'}")


def _resource_lines(helpers: Any, relative: Path, replacements: dict[str, str]) -> list[str]:
    source = RESOURCE_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Template resource not found: {source}")
    return helpers.render_template(helpers.read_text(source), replacements, "\n").split("\n")


def _resource_line(resource: list[str], predicate: Callable[[str], bool], what: str) -> int:
    index = _find(resource, predicate)
    if index is None:
        raise ValueError(f"{what} was not found in the template resource")
    return index


def _is_package_import(line: str, bat: bool) -> bool:
    return bool(re.match(r'^\s+"(%PYTHON_EXE%|\$PYTHON_EXE)" -c "import [A-Za-z_]\w*;', line))


def _package_install_index(lines: list[str]) -> int:
    return _only(lines, lambda l: "pip install --no-deps --no-build-isolation" in l, "the package install line (pip install --no-deps --no-build-isolation)")  # type: ignore[return-value]


def _mark_moved(stripped: list[str], removed: set[int], index: int, pattern: re.Pattern[str]) -> None:
    """Mark line `index` and the moved comment lines directly above it (a run the 0.1.15/0.1.16 hook wrote) as removed."""
    removed.add(index)
    j = index - 1
    while j >= 0 and (j in removed or pattern.match(stripped[j])):
        removed.add(j)
        j -= 1


def _rebuild(lines: list[str], removed: set[int], replaced: dict[int, str], inserted: dict[int, list[str]]) -> list[str]:
    """The file after the edits: `inserted` lines go before the given index, `replaced` lines take the place of a line, `removed` lines are dropped."""
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in inserted:
            if out and not out[-1].endswith(("\n", "\r")):
                out[-1] += "\n"
            out.extend(inserted[index])
        if index in replaced:
            out.append(replaced[index])
        elif index not in removed:
            out.append(line)
    return out


def _with_newlines(lines: list[str], newline: str) -> list[str]:
    return [line + newline for line in lines]


def _mark_moved(stripped: list[str], removed: set[int], index: int, pattern: re.Pattern[str]) -> None:
    """Mark line `index` and the moved comment lines directly above it (a run the 0.1.15/0.1.16 hook wrote) as removed."""
    removed.add(index)
    j = index - 1
    while j >= 0 and (j in removed or pattern.match(stripped[j])):
        removed.add(j)
        j -= 1


def _rebuild(lines: list[str], removed: set[int], replaced: dict[int, str], inserted: dict[int, list[str]]) -> list[str]:
    """The file after the edits: `inserted` lines go before the given index, `replaced` lines take the place of a line, `removed` lines are dropped."""
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in inserted:
            if out and not out[-1].endswith(("\n", "\r")):
                out[-1] += "\n"
            out.extend(inserted[index])
        if index in replaced:
            out.append(replaced[index])
        elif index not in removed:
            out.append(line)
    return out


def _with_newlines(lines: list[str], newline: str) -> list[str]:
    return [line + newline for line in lines]


def _ensure_ends_with_newline(lines: list[str], index: int, newline: str) -> None:
    if 0 <= index - 1 < len(lines) and not lines[index - 1].endswith(("\n", "\r")):
        lines[index - 1] += newline


# --------------------------------------------------------------------------------------------------------------------------------
# post_install.sh
# --------------------------------------------------------------------------------------------------------------------------------
def _migrate_sh(helpers: Any, path: Path, resource: list[str]) -> str:
    text = helpers.read_text(path)
    if DONE_MARKER in text:
        return UNCHANGED
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to move"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)

    start = _package_install_index(lines)
    imp = _find(lines, lambda l: _is_package_import(l, False), start + 1)
    if imp is None:
        raise Unsupported("the import check line after the package install was not found")
    stripped = [l.strip() for l in lines]
    module_re = re.compile(r'^LC_TOOLS_MODULE="([^"]+)"$')
    mod = _only(stripped, lambda l: bool(module_re.match(l)), "the LC_TOOLS_MODULE=\"...\" line", imp + 1)
    assert mod is not None
    module = module_re.match(stripped[mod]).group(1)  # type: ignore[union-attr]
    pin_re = re.compile(r'^LC_TOOLS_PIN="([^"]+)"$')
    pin = _only(stripped, lambda l: bool(pin_re.match(l)), "the LC_TOOLS_PIN=\"...\" line", imp + 1, optional=True)
    pip_re = re.compile(r'^if "\$PYTHON_EXE" -m pip install ("\$\{LC_TOOLS_SPEC:-.*\}") >> "\$LOG_FILE" 2>&1 \\$')
    pip = _only(stripped, lambda l: bool(pip_re.match(l)), "the Tools `pip install ... \\` line of the registration block", mod + 1)
    assert pip is not None
    if not stripped[pip + 1].startswith('&& "$PYTHON_EXE" -m labconstrictor_tools register'):
        raise Unsupported("the line after the Tools pip install is not the `labconstrictor_tools register` line")
    spec = pip_re.match(stripped[pip]).group(1)  # type: ignore[union-attr]
    uses_pin = "LC_TOOLS_PIN" in spec

    # The new block, from the template resource, with the app's own module, pin and Tools source.
    first = _resource_line(resource, lambda l: l.startswith("    # Tools first"), "the Tools install block")
    last = _resource_line(resource, lambda l: _is_package_import(l, False), "the import check line")
    block: list[str] = []
    for line in resource[first:last]:
        if line.strip().startswith("LC_TOOLS_MODULE="):
            block.append("    " + stripped[mod])
        elif line.strip().startswith("LC_TOOLS_PIN="):
            if uses_pin:  # an app whose Tools source has no pin variable gets no pin line
                block.append("    " + (stripped[pin] if pin is not None else line.strip()))
        elif SH_TEMPLATE_SPEC in line:
            block.append(line.replace(SH_TEMPLATE_SPEC, spec))
        elif not uses_pin and re.match(r"^\s*# (Default source|because the package)", line):
            continue  # these comment lines describe the pinned tag the app's source does not use
        else:
            block.append(line)
    comment = resource[_resource_line(resource, lambda l: l.startswith("# labconstrictor-tools is installed above"), "the moved-variables comment")]
    ready = resource[_resource_line(resource, lambda l: l.lstrip().startswith('if [ "${LC_TOOLS_READY:-0}" = 1 ]'), "the LC_TOOLS_READY check")]

    removed: set[int] = set()
    if pin is not None:
        _mark_moved(stripped, removed, pin, SH_MOVED_COMMENT)
    _mark_moved(stripped, removed, mod, SH_MOVED_COMMENT)
    removed.discard(mod)
    indent = lines[pip][: len(lines[pip]) - len(lines[pip].lstrip())]
    replaced = {mod: comment + newline, pip: indent + ready.strip() + newline}
    lines = _rebuild(lines, removed, replaced, {imp: _with_newlines(block, newline)})
    helpers.write_text(path, "".join(lines))
    return f"updated: Tools are installed before the import check (module {module})"


# --------------------------------------------------------------------------------------------------------------------------------
# post_install.bat
# --------------------------------------------------------------------------------------------------------------------------------
def _migrate_bat(helpers: Any, path: Path, resource: list[str]) -> str:
    text = helpers.read_text(path)
    if DONE_MARKER in text:
        return UNCHANGED
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to move"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    stripped = [l.strip() for l in lines]
    last_line = next((s for s in reversed(stripped) if s), "")
    if not re.match(r"^(GOTO :EOF|EXIT /B)", last_line, re.IGNORECASE):
        raise Unsupported("the file does not end with GOTO :EOF or EXIT /B, so a subroutine cannot be appended safely")

    start = _package_install_index(lines)
    imp = _find(lines, lambda l: _is_package_import(l, True), start + 1)
    if imp is None:
        raise Unsupported("the import check line after the package install was not found")
    spec_re = re.compile(r'^IF NOT DEFINED LC_TOOLS_SPEC SET "LC_TOOLS_SPEC=.*"$', re.IGNORECASE)
    spec = _only(stripped, lambda l: bool(spec_re.match(l)), "the `IF NOT DEFINED LC_TOOLS_SPEC SET ...` line", imp + 1)
    assert spec is not None
    pin_re = re.compile(r'^SET "LC_TOOLS_PIN=[^"]+"$', re.IGNORECASE)
    pin = _only(stripped, lambda l: bool(pin_re.match(l)), 'the SET "LC_TOOLS_PIN=..." line', imp + 1, optional=True)
    uses_pin = "%LC_TOOLS_PIN%" in stripped[spec]
    find_re = re.compile(r"""^"%PYTHON_EXE%" -c "import importlib\.util, sys; sys\.exit\(0 if importlib\.util\.find_spec\('([^']+)'\) else 1\)" >> "%LOG_FILE%" 2>&1$""")
    find = _only(stripped, lambda l: bool(find_re.match(l)), "the find_spec line of the registration block", spec + 1)
    assert find is not None
    module = find_re.match(stripped[find]).group(1)  # type: ignore[union-attr]
    pip = _only(stripped, lambda l: l == BAT_PIP_LINE, "the Tools pip install line (through Python) of the registration block", find + 1)
    assert pip is not None

    # Template resource: the call block, the replacement lines and the subroutine.
    first = _resource_line(resource, lambda l: l.startswith("    REM Tools first"), "the Tools call block")
    last = _resource_line(resource, lambda l: _is_package_import(l, True), "the import check line")
    call_block = resource[first:last]
    comment = resource[_resource_line(resource, lambda l: l.startswith("REM labconstrictor-tools is installed by :install_lc_tools"), "the moved-variables comment")]
    ready_at = _resource_line(resource, lambda l: l.lstrip().startswith("REM labconstrictor-tools was installed before the import check"), "the LC_TOOLS_READY check")
    ready = resource[ready_at : ready_at + 2]
    sub_first = _resource_line(resource, lambda l: l.strip().lower() == ":install_lc_tools", "the subroutine")
    subroutine = resource[sub_first:]
    while subroutine and not subroutine[-1].strip():
        subroutine.pop()
    sub = []
    for line in subroutine:
        if line.startswith("SET \"LC_TOOLS_PIN="):
            if uses_pin:
                sub.append(stripped[pin] if pin is not None else line)
        elif line.startswith("IF NOT DEFINED LC_TOOLS_SPEC SET"):
            sub.append(stripped[spec])
        else:
            sub.append(line.replace("PYTHON_PROJ_NAME_lc_tools", module))
    if not uses_pin:  # no pin variable in the app's source: the comment that talks about "the TAG below" would mislead
        sub = [l for l in sub if not l.startswith(("REM Default source", "REM because the package"))]

    # The pin / spec / header comments of the registration block are replaced by one comment, the pip line by the READY check.
    removed: set[int] = set()
    if pin is not None:
        _mark_moved(stripped, removed, pin, BAT_MOVED_COMMENT)
    _mark_moved(stripped, removed, spec, BAT_MOVED_COMMENT)
    removed.discard(spec)
    replaced = {spec: comment + newline, pip: "".join(_with_newlines(["    " + r.strip() for r in ready], newline))}
    if stripped[pip - 1].startswith("REM LC_TOOLS_SPEC goes to pip as one argument"):
        removed.add(pip - 1)
    lines = _rebuild(lines, removed, replaced, {imp: _with_newlines(call_block, newline)})
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    lines.extend(_with_newlines([""] + sub, newline))
    helpers.write_text(path, "".join(lines))
    return f"updated: Tools are installed before the import check (module {module})"


def _migrate_docs(helpers: Any, repo_root: Path, replacements: dict[str, str]) -> str:
    source = RESOURCE_ROOT / TOOLS_DOC
    destination = repo_root / TOOLS_DOC
    if not destination.exists():
        return "not present in this repository (nothing was created)"
    rendered = helpers.render_template(helpers.read_text(source), replacements, helpers.detect_newline(helpers.read_text(destination)))
    if helpers.read_text(destination) == rendered:
        return "unchanged"
    helpers.write_text(destination, rendered)
    return "updated: written"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Install Tools before the import check in the post_install scripts (idempotent, surgical, never raises for a customised file)."""
    _ = context
    helpers = _load_helpers()
    try:
        replacements = helpers.load_replacements(repo_root)
    except (OSError, ValueError) as exc:
        log(f"could not read the app name from construct.yaml ({exc}): messages of the inserted block name the app generically.")
        replacements = {"PROJECT_NAME": "the app"}

    changed = 0
    for relative, step in ((POST_INSTALL_SH, _migrate_sh), (POST_INSTALL_BAT, _migrate_bat)):
        path = repo_root / relative
        if not path.is_file():
            log(f"{relative}: SKIPPED, file not found in this repository (nothing was created).")
            continue
        try:
            # the module name of the hook decides PYTHON_PROJ_NAME in the block, so it is rendered after reading the hook
            module = _hook_module(helpers, path)
            local = dict(replacements)
            if module:
                local["PYTHON_PROJ_NAME"] = module
            status = step(helpers, path, _resource_lines(helpers, relative, local))
        except Unsupported as exc:
            status = f"SKIPPED: {exc}; the script was left unchanged. Move the Tools install before the import check by hand, see {DOCS_URL}"
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; the script was left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{relative}: {status}")

    try:
        status = _migrate_docs(helpers, repo_root, replacements)
        if status.startswith("updated"):
            changed += 1
        log(f"{TOOLS_DOC}: {status}")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log(f"{TOOLS_DOC}: SKIPPED: {exc}")

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the change reaches users with the next release of the app.")
    else:
        log("done: no repository changes were required.")


def _hook_module(helpers: Any, path: Path) -> str | None:
    """The package name of the existing hook (its <package>_lc_tools module, minus the suffix), if the hook is there."""
    text = helpers.read_text(path)
    match = re.search(r"LC_TOOLS_MODULE=\"(\w+)_lc_tools\"|find_spec\('(\w+)_lc_tools'\)", text)
    if not match:
        return None
    return match.group(1) or match.group(2)

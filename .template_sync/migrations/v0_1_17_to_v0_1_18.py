"""Template 0.1.17 -> 0.1.18: new Tools pin, GitHub Actions pinned by commit SHA, install rules in the install guide, app hosts for Tools.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_18.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. Tools pin. `LC_TOOLS_PIN` in app/bash_bat_scripts/post_install.sh / .bat moves from the old template value `v0.2.0-rc.1` to the new one
   (read from the resource copy of the template script, `v0.3.0-alpha.1`), ONLY when the line still holds exactly the old template value.
   An app with its own pin, a script without a pin line (unpinned source, own `LC_TOOLS_SPEC`) or without the hook stays as it is, with a note.
2. Pinned actions. The five template workflows (check_app_manifest.yml, release_constructor.yml, sync_template.yml,
   update_on_notebook_change.yml, update_on_src_change.yaml) are replaced by the pinned version ONLY when the app's file is still identical to
   the previous template version (the resource with the pins removed; line endings are ignored when comparing and kept when writing).
   A file that already is the pinned template file is unchanged; a file that is customised (or pinned in part by the app) is skipped with a
   message that lists its still unpinned `uses:` lines. A workflow file that is not in the repository is not created.
3. Install guide. `.tools/docs/install.md`, if it exists, gets the section "Before you install" (path without a space) and the section
   "Update an installed app" (macOS/Windows uninstall first, Linux `-u`), inserted before "## Install on Windows" and before "## Next".
4. Tools guide. `.tools/docs/tools_bridge.md` is refreshed (it names the pinned tag) only if it still is the 0.1.17 text.
5. App hosts. The installer passes the app's `hosts:` line to Tools (`register --app-manifest`): `construct.yaml` gets the `extra_files` entry that
   bundles `labconstrictor-app.yaml` (only if that file exists in the repository, and only in the known block shape), and the Tools hook of
   post_install.sh / .bat gets the lines that pass the bundled file and log when it is missing (only in the known hook shape; the app's own
   register arguments stay). A hook that already passes `--app-manifest` or `--hosts` is unchanged; any other shape is SKIPPED with a note.

Rules this migration keeps: surgical (only known lines or known files), idempotent (a second run reports no changes), never raises for a
customised file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_17_to_v0_1_18"
SCRIPTS = Path("app/bash_bat_scripts")
POST_INSTALL_SH = SCRIPTS / "post_install.sh"
POST_INSTALL_BAT = SCRIPTS / "post_install.bat"
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
INSTALL_DOC = Path(".tools/docs/install.md")
WORKFLOWS = (
    Path(".github/workflows/check_app_manifest.yml"),
    Path(".github/workflows/release_constructor.yml"),
    Path(".github/workflows/sync_template.yml"),
    Path(".github/workflows/update_on_notebook_change.yml"),
    Path(".github/workflows/update_on_src_change.yaml"),
)
# Resource files: copies of the template files of 0.1.18 (a test checks they stay identical to the template).
MANAGED_FILES = (POST_INSTALL_SH, POST_INSTALL_BAT, TOOLS_DOC, INSTALL_DOC, *WORKFLOWS)

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_18.md"
TAG = "[migration 0.1.18]"
OLD_PIN = "v0.2.0-rc.1"
NEW_PIN_RE = re.compile(r'LC_TOOLS_PIN="([^"]+)"')

# The pinned third-party action lines, `uses: owner/repo@<40 hex> # <tag>`; removing the pin gives `uses: owner/repo@<tag>`.
PINNED_USES = re.compile(r"^(\s*(?:-\s+)?uses:\s*[\w.-]+/[\w./-]+)@[0-9a-f]{40}[ \t]+#[ \t]*(\S+)[ \t]*$")
ANY_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*([\w.-]+/[\w./-]+)@(\S+)")

INSTALL_BEFORE_HEADING = "## Before you install"
INSTALL_UPDATE_HEADING = "## Update an installed app"
INSTALL_HOSTS_HEADING = "## Where the app is listed"
INSTALL_BEFORE_ANCHOR = "## Install on Windows"
INSTALL_UPDATE_ANCHOR = "## Next"
CONSTRUCT = Path("construct.yaml")
APP_MANIFEST = Path("labconstrictor-app.yaml")


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers() -> Any:
    """Reuse the helpers of the 0.1.10 migration (token replacement, newline detection, file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_18", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resource_text(helpers: Any, relative: Path) -> str:
    source = RESOURCE_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Template resource not found: {source}")
    return str(helpers.read_text(source)).replace("\r\n", "\n")


def new_pin(helpers: Any) -> str:
    """The pin of the template (the resource copy of post_install.sh is the single source of the value)."""
    match = NEW_PIN_RE.search(_resource_text(helpers, POST_INSTALL_SH))
    if not match:
        raise ValueError("LC_TOOLS_PIN was not found in the template resource")
    return match.group(1)


def _unpin(text: str) -> str:
    """The text of a workflow before its actions were pinned (the previous template version)."""
    lines = []
    for line in text.split("\n"):
        match = PINNED_USES.match(line)
        lines.append(f"{match.group(1)}@{match.group(2)}" if match else line)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------
# 1. Tools pin
# --------------------------------------------------------------------------------------------------------------------------------
SH_PIN_LINE = re.compile(r'^([ \t]*LC_TOOLS_PIN=")([^"]*)(".*)$')
BAT_PIN_LINE = re.compile(r'^([ \t]*SET "LC_TOOLS_PIN=)([^"]*)(".*)$', re.IGNORECASE)


def _move_pin(helpers: Any, path: Path, pattern: re.Pattern[str], pin: str) -> str:
    text = helpers.read_text(path)
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to move"
    lines = text.splitlines(keepends=True)
    hits = []
    for index, line in enumerate(lines):
        match = pattern.match(line.rstrip("\r\n"))
        if match:
            hits.append((index, match))
    if not hits:
        return "has no LC_TOOLS_PIN line (unpinned source or its own Tools source): left as it is"
    if len(hits) > 1:
        return f"has {len(hits)} LC_TOOLS_PIN lines: left as it is, set the pin by hand to {pin}"
    index, match = hits[0]
    current = match.group(2)
    if current == pin:
        return f"already pins Tools {pin}: unchanged"
    if current != OLD_PIN:
        return f"has its own pin {current}: left as it is (the template pin is now {pin})"
    ending = lines[index][len(lines[index].rstrip("\r\n")) :]
    lines[index] = f"{match.group(1)}{pin}{match.group(3)}{ending}"
    helpers.write_text(path, "".join(lines))
    return f"updated: Tools pin {OLD_PIN} -> {pin}"


# --------------------------------------------------------------------------------------------------------------------------------
# 2. Pinned actions
# --------------------------------------------------------------------------------------------------------------------------------
def _unpinned_uses(text: str) -> list[str]:
    """`uses:` lines of third-party actions that are not pinned by SHA (local `./` actions and `docker://` need no pin)."""
    found = []
    for line in text.replace("\r\n", "\n").split("\n"):
        match = ANY_USES.match(line)
        if match and not re.fullmatch(r"[0-9a-f]{40}", match.group(2)) and not match.group(1).startswith(("./", "docker:")):
            found.append(line.strip())
    return found


def _migrate_workflow(helpers: Any, repo_root: Path, relative: Path) -> str:
    path = repo_root / relative
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    new = _resource_text(helpers, relative)
    current = helpers.read_text(path)
    current_lf = current.replace("\r\n", "\n")
    if current_lf == new:
        return "already the pinned template file: unchanged"
    if current_lf == _unpin(new):
        newline = helpers.detect_newline(current)
        helpers.write_text(path, new.replace("\n", newline))
        return "updated: every third-party action is pinned by commit SHA"
    left = _unpinned_uses(current)
    if not left:
        return "customised and already pinned: left as it is"
    shown = "; ".join(left[:6]) + ("; ..." if len(left) > 6 else "")
    return f"SKIPPED: customised, so it was left unchanged ({len(left)} third-party action(s) not pinned by SHA: {shown}). Pin them by hand, see {DOCS_URL}"


# --------------------------------------------------------------------------------------------------------------------------------
# 3. Install guide
# --------------------------------------------------------------------------------------------------------------------------------
def _section(text: str, heading: str) -> str:
    """The section of `heading` in the (LF) resource text, up to the next `## ` heading, with its trailing blank line."""
    match = re.search(rf"^{re.escape(heading)}\n.*?(?=^## )", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"section {heading!r} was not found in the template resource")
    return match.group(0)


def _migrate_install_doc(helpers: Any, repo_root: Path) -> str:
    path = repo_root / INSTALL_DOC
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    resource = _resource_text(helpers, INSTALL_DOC)
    current = helpers.read_text(path)
    newline = helpers.detect_newline(current)
    lf = current.replace("\r\n", "\n")
    done = []
    for heading, anchor in (
        (INSTALL_BEFORE_HEADING, INSTALL_BEFORE_ANCHOR),
        (INSTALL_UPDATE_HEADING, INSTALL_UPDATE_ANCHOR),
        (INSTALL_HOSTS_HEADING, INSTALL_UPDATE_ANCHOR),
    ):
        if re.search(rf"^{re.escape(heading)}\s*$", lf, re.MULTILINE):
            continue
        position = re.search(rf"^{re.escape(anchor)}\s*$", lf, re.MULTILINE)
        if not position:
            return f"SKIPPED: no line {anchor!r} to insert {heading!r} before; add the sections by hand from {DOCS_URL}"
        lf = lf[: position.start()] + _section(resource, heading) + lf[position.start() :]
        done.append(heading[3:])
    if not done:
        return "already has the install rules: unchanged"
    helpers.write_text(path, lf.replace("\n", newline))
    return "updated: added " + " and ".join(repr(d) for d in done)


# --------------------------------------------------------------------------------------------------------------------------------
# 4. Tools guide
# --------------------------------------------------------------------------------------------------------------------------------
def _migrate_tools_doc(helpers: Any, repo_root: Path, replacements: dict[str, str], pin: str) -> str:
    destination = repo_root / TOOLS_DOC
    if not destination.is_file():
        return "not present in this repository (nothing was created)"
    current = helpers.read_text(destination)
    newline = helpers.detect_newline(current)
    new = helpers.render_template(_resource_text(helpers, TOOLS_DOC), replacements, newline)
    if current == new:
        return "unchanged"
    old = helpers.render_template(_resource_text(helpers, TOOLS_DOC).replace(pin, OLD_PIN), replacements, newline)
    if current != old:
        return "SKIPPED: not the 0.1.17 text (customised), so it was left unchanged; it names the pinned Tools tag " + pin
    helpers.write_text(destination, new)
    return "updated: written"


# --------------------------------------------------------------------------------------------------------------------------------
# 5. App hosts (labconstrictor-app.yaml bundled by the installer, passed to `register --app-manifest`)
# --------------------------------------------------------------------------------------------------------------------------------
PASSES_HOSTS = re.compile(r"--app-manifest|--hosts")
EXTRA_FILES_HEAD = re.compile(r"^extra_files:\s*$")
EXTRA_FILES_ITEM = re.compile(r"^- [^\s#][^:]*:\s*\S.*$")
MANIFEST_MAPPING = re.compile(r"^\s*-\s*['\"]?labconstrictor-app\.yaml['\"]?\s*:")
WELCOME_MAPPING = re.compile(r"^- [^:]*Welcome\.ipynb:\s*['\"]?([^/'\"\s]+)/")
REGISTER_NAME = re.compile(r'--name "([^"]+)"')
SH_BLOCK_START = "    # Hosts of the app:"
SH_BLOCK_END = '    if [ "${LC_TOOLS_READY:-0}" = 1 ]'
BAT_PRE_START = "REM Hosts of the app:"
BAT_LOGS_START = "    REM Which hosts list the app"
BAT_LOGS_END = "    REM labconstrictor-tools was installed before"
BAT_SPEC_CALL = "importlib.util.find_spec('PYTHON_PROJ_NAME_lc_tools')"
BAT_REGISTER = re.compile(r'^(\s*IF NOT ERRORLEVEL 1 )("%PYTHON_EXE%" -m labconstrictor_tools register .*?)( >> "%LOG_FILE%" 2>&1)\s*$')
SH_LOG_REDIRECT = ' >> "$LOG_FILE" 2>&1'
SH_HOSTS_ARGS = '${LC_HOSTS_ARGS[@]+"${LC_HOSTS_ARGS[@]}"}'


def _slice(resource: list[str], start: str, end: str, what: str) -> list[str]:
    """The lines of the template resource from the first line that starts with `start` up to (not including) the next line that starts with `end`."""
    begin = next((i for i, line in enumerate(resource) if line.startswith(start)), None)
    if begin is None:
        raise ValueError(f"{what}: the block was not found in the template resource")
    stop = next((i for i in range(begin, len(resource)) if resource[i].startswith(end)), None)
    if stop is None:
        raise ValueError(f"{what}: the end of the block was not found in the template resource")
    return resource[begin:stop]


def _only_line(lines: list[str], pattern: str, what: str) -> int:
    hits = [i for i, line in enumerate(lines) if pattern in line]
    if len(hits) != 1:
        raise ValueError(f"{what} was {'not found' if not hits else 'found more than once'}")
    return hits[0]


def _app_name(line: str) -> str:
    match = REGISTER_NAME.search(line)
    if not match:
        raise ValueError('the register line has no --name "..." argument')
    return match.group(1)


def _sh_hook_lines(lines: list[str]) -> tuple[int, int, str]:
    """(index of the first line of the register statement, index of the line that ends it, the app name) in a known post_install.sh hook."""
    reg = _only_line(lines, "labconstrictor_tools register", "the register line")
    if not any(line.lstrip().startswith("lc_note()") for line in lines[:reg]):
        raise ValueError("lc_note is not defined before the register line")
    first = reg
    while first > 0 and lines[first - 1].rstrip("\r\n").rstrip().endswith("\\"):
        first -= 1
    if not lines[first].lstrip().startswith("if "):
        raise ValueError("the register command is not the condition of an `if` statement")
    last = next((i for i in range(reg, min(reg + 4, len(lines))) if SH_LOG_REDIRECT in lines[i]), None)
    if last is None or lines[last].count(SH_LOG_REDIRECT) != 1:
        raise ValueError(f"the end of the register command (`{SH_LOG_REDIRECT.strip()}`) was not found")
    return first, last, _app_name("".join(lines[first : reg + 1]))


def _migrate_hosts_sh(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to do"
    if PASSES_HOSTS.search(text):
        return "already passes the hosts of the app to Tools: unchanged"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    first, last, name = _sh_hook_lines(lines)
    indent = lines[first][: len(lines[first]) - len(lines[first].lstrip())]
    block = _slice(_resource_text(helpers, POST_INSTALL_SH).split("\n"), SH_BLOCK_START, SH_BLOCK_END, "post_install.sh")
    inserted = [(indent + line[4:] if line.startswith("    ") else line).replace("PROJECT_NAME", name) + newline for line in block]
    lines[last] = lines[last].replace(SH_LOG_REDIRECT, f" {SH_HOSTS_ARGS}{SH_LOG_REDIRECT}", 1)
    helpers.write_text(path, "".join(lines[:first] + inserted + lines[first:]))
    return "updated: the Tools hook passes --app-manifest and logs when labconstrictor-app.yaml is missing"


def _bat_hook_lines(lines: list[str]) -> tuple[int, int, int, re.Match[str]]:
    """(index of the line before which the manifest lookup goes, index after which the log lines go, register index, its match) in a known post_install.bat hook."""
    reg = _only_line(lines, "labconstrictor_tools register", "the register line")
    match = BAT_REGISTER.match(lines[reg].rstrip("\r\n"))
    if not match:
        raise ValueError("the register line does not have the known shape (IF NOT ERRORLEVEL 1 ... >> LOG_FILE 2>&1)")
    found = max((i for i in range(reg) if re.match(r"^\s*echo Found \S+_lc_tools: registering", lines[i])), default=None)
    block = max((i for i in range(reg) if lines[i].rstrip("\r\n") == "IF NOT ERRORLEVEL 1 ("), default=None)
    if found is None or block is None or block == 0 or "find_spec(" not in lines[block - 1] or block > found:
        raise ValueError("the `IF NOT ERRORLEVEL 1 (` block of the hook after the find_spec check was not found")
    return block - 1, found, reg, match


def _migrate_hosts_bat(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to do"
    if PASSES_HOSTS.search(text):
        return "already passes the hosts of the app to Tools: unchanged"
    newline = helpers.detect_newline(text)
    lines = text.splitlines(keepends=True)
    before, found, reg, match = _bat_hook_lines(lines)
    name = _app_name(lines[reg])
    resource = _resource_text(helpers, POST_INSTALL_BAT).split("\n")
    pre = _slice(resource, BAT_PRE_START, "\"%PYTHON_EXE%\" -c \"import importlib.util, sys; sys.exit(0 if " + BAT_SPEC_CALL, "post_install.bat")
    logs = _slice(resource, BAT_LOGS_START, BAT_LOGS_END, "post_install.bat")
    head, command, tail = match.groups()
    register = [
        head + "IF DEFINED LC_APP_MANIFEST " + command + ' --app-manifest "%LC_APP_MANIFEST%"' + tail,
        head + "IF NOT DEFINED LC_APP_MANIFEST " + command + tail,
    ]

    def lines_of(texts: list[str]) -> list[str]:
        return [item.replace("PROJECT_NAME", name) + newline for item in texts]

    lines[reg : reg + 1] = lines_of(register)
    lines[found + 1 : found + 1] = lines_of(logs)
    lines[before:before] = lines_of(pre)
    helpers.write_text(path, "".join(lines))
    return "updated: the Tools hook passes --app-manifest and logs when labconstrictor-app.yaml is missing"


def _migrate_construct(helpers: Any, repo_root: Path) -> str:
    """Add the extra_files entry that bundles labconstrictor-app.yaml (only if the file exists, in the known block shape)."""
    path = repo_root / CONSTRUCT
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    text = helpers.read_text(path)
    lines = text.splitlines(keepends=True)
    stripped = [line.rstrip("\r\n") for line in lines]
    if any(MANIFEST_MAPPING.match(line) for line in stripped):
        return "already bundles labconstrictor-app.yaml: unchanged"
    if not (repo_root / APP_MANIFEST).is_file():
        return (
            "SKIPPED: the repository has no labconstrictor-app.yaml, so it is not bundled and the app is registered without hosts (listed in every host). "
            f"Add the file, then add `- labconstrictor-app.yaml: <project folder>/labconstrictor-app.yaml` to extra_files. See {DOCS_URL}"
        )
    heads = [i for i, line in enumerate(stripped) if EXTRA_FILES_HEAD.match(line)]
    if len(heads) != 1:
        raise ValueError(f"expected one `extra_files:` block in the usual block shape, found {len(heads)}; add the mapping by hand")
    last = heads[0]
    while last + 1 < len(stripped) and EXTRA_FILES_ITEM.match(stripped[last + 1]):
        last += 1
    folders = {m.group(1) for m in (WELCOME_MAPPING.match(line) for line in stripped[heads[0] + 1 : last + 1]) if m}
    if last == heads[0] or len(folders) != 1:
        raise ValueError("the project folder could not be read from the Welcome.ipynb entry of extra_files; add the mapping by hand")
    newline = helpers.detect_newline(text)
    entry = f"- labconstrictor-app.yaml: {folders.pop()}/labconstrictor-app.yaml{newline}"
    if not lines[last].endswith(("\n", "\r")):
        lines[last] += newline
    lines.insert(last + 1, entry)
    helpers.write_text(path, "".join(lines))
    return "updated: extra_files bundles labconstrictor-app.yaml"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Move the template Tools pin, pin the workflow actions, add the install rules (idempotent, surgical, never raises for a customised file)."""
    _ = context
    helpers = _load_helpers()
    pin = new_pin(helpers)
    try:
        replacements = helpers.load_replacements(repo_root)
    except (OSError, ValueError) as exc:
        log(f"could not read the app name from construct.yaml ({exc}): the guides are written without app names.")
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

    for relative, pattern in ((POST_INSTALL_SH, SH_PIN_LINE), (POST_INSTALL_BAT, BAT_PIN_LINE)):
        if not (repo_root / relative).is_file():
            log(f"{relative}: SKIPPED, file not found in this repository (nothing was created).")
            continue
        report(relative, lambda relative=relative, pattern=pattern: _move_pin(helpers, repo_root / relative, pattern, pin))
    for relative in WORKFLOWS:
        report(relative, lambda relative=relative: _migrate_workflow(helpers, repo_root, relative))
    report(CONSTRUCT, lambda: _migrate_construct(helpers, repo_root))
    for relative, step in ((POST_INSTALL_SH, _migrate_hosts_sh), (POST_INSTALL_BAT, _migrate_hosts_bat)):
        if (repo_root / relative).is_file():
            report(relative, lambda relative=relative, step=step: step(helpers, repo_root / relative))
    report(INSTALL_DOC, lambda: _migrate_install_doc(helpers, repo_root))
    report(TOOLS_DOC, lambda: _migrate_tools_doc(helpers, repo_root, replacements, pin))

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the pin reaches users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

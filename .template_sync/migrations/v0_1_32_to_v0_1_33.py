"""Template 0.1.32 -> 0.1.33: Tools v0.3.0-alpha.3 (Fiji opens results of 3 or 4 planes); the Windows Apps list shows the right version and icon.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_33.md

Tools v0.3.0-alpha.2 wrote a result image of 3 or 4 planes (3 channels, 3 timepoints) as one page of colour samples, which Fiji cannot open
unless it is 8-bit (LabConstrictor-Tools #163); v0.3.0-alpha.3 writes it as grey pages. What this changes in an app repository (each step is
independent, idempotent and never deletes anything):

1. `LC_TOOLS_PIN` in app/bash_bat_scripts/post_install.sh and post_install.bat moves from `v0.3.0-alpha.2` to `v0.3.0-alpha.3`, ONLY when the
   line still holds exactly the old template value. An app with its own pin, a script without a pin line or without the Tools hook stays as it is,
   with a note.

2. post_install.bat registered the app in the Windows Apps list with the version written at initialisation (`DISPLAY_VERSION`; no
   release ever changed it) and the icon `logo.ico`, a file no installer contains. The version is now read at install time from the
   bundled construct.yaml (the setup-time value stays as a fallback), and the icon is the first .ico the installer copied into the app
   folder; without one, no icon entry is written. Only the template's exact lines are replaced; a changed script stays as it is, with a note.

Since template 0.1.32 the Tools guide no longer names the pin, so no document changes. Both changes reach users with the app's next release.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_33.md"
TAG = "[migration 0.1.33]"
OLD_PIN = "v0.3.0-alpha.2"
NEW_PIN = "v0.3.0-alpha.3"
POST_INSTALL_SH = Path("app/bash_bat_scripts/post_install.sh")
POST_INSTALL_BAT = Path("app/bash_bat_scripts/post_install.bat")
SH_PIN_LINE = re.compile(r'^([ \t]*LC_TOOLS_PIN=")([^"]*)(".*)$')
# The Windows Apps list block of post_install.bat (CRLF in the script; matched without line endings). The app's own setup-time version is kept.
OLD_ARP = re.compile(r'^SET "DISPLAY_ICON=%PROJECT_ROOT%\\logo\.ico"\r?\nSET "DISPLAY_VERSION=([^"\r\n]*)"(?=\r?\n)', re.MULTILINE)
NEW_ARP_LINES = (
    "REM Windows Apps list: the version of this installer (the version: line of the bundled construct.yaml, written by each release),",
    "REM and the app icon the installer copied (the first .ico in the app folder); no icon entry when there is none.",
    'SET "DISPLAY_VERSION={version}"',
    'IF EXIST "%PROJECT_ROOT%\\construct.yaml" FOR /F "usebackq tokens=1,* delims=: " %%A IN (`findstr /B /C:"version:" "%PROJECT_ROOT%\\construct.yaml"`) DO SET "DISPLAY_VERSION=%%~B"',
    'SET "DISPLAY_ICON="',
    'FOR %%I IN ("%PROJECT_ROOT%\\*.ico") DO IF NOT DEFINED DISPLAY_ICON SET "DISPLAY_ICON=%%~fI"',
)
OLD_ICON_REG = 'reg add "%ARP_KEY%" /v DisplayIcon /d "%DISPLAY_ICON%" /f >> "%LOG_FILE%" 2>&1'
NEW_ICON_REG = "IF DEFINED DISPLAY_ICON " + OLD_ICON_REG
BAT_PIN_LINE = re.compile(r'^([ \t]*SET "LC_TOOLS_PIN=)([^"]*)(".*)$', re.IGNORECASE)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def read(path: Path) -> str:
    """The file's text with its line endings as they are (bytes decoded as UTF-8)."""
    return path.read_bytes().decode("utf-8")


def write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def move_pin(path: Path, pattern: re.Pattern[str]) -> str:
    text = read(path)
    if "labconstrictor_tools" not in text:
        return "has no Tools bridge hook: nothing to move"
    lines = text.splitlines(keepends=True)
    hits = [(index, match) for index, line in enumerate(lines) if (match := pattern.match(line.rstrip("\r\n")))]
    if not hits:
        return "has no LC_TOOLS_PIN line (unpinned source or its own Tools source): left as it is"
    if len(hits) > 1:
        return f"has {len(hits)} LC_TOOLS_PIN lines: left as it is, set the pin by hand to {NEW_PIN}"
    index, match = hits[0]
    current = match.group(2)
    if current == NEW_PIN:
        return f"already pins Tools {NEW_PIN}: unchanged"
    if current != OLD_PIN:
        return f"has its own pin {current}: left as it is (the template pin is now {NEW_PIN})"
    ending = lines[index][len(lines[index].rstrip("\r\n")) :]
    lines[index] = f"{match.group(1)}{NEW_PIN}{match.group(3)}{ending}"
    write(path, "".join(lines))
    return f"updated: Tools pin {OLD_PIN} -> {NEW_PIN}"


def fix_apps_list(path: Path) -> str:
    text = read(path)
    if NEW_ICON_REG in text:
        return "already reads the Apps-list version and icon at install time: unchanged"
    match = OLD_ARP.search(text)
    if match is None or text.count(OLD_ICON_REG) != 1:
        return "has no template Apps-list block (changed by hand): left as it is"
    newline = "\r\n" if "\r\n" in text else "\n"
    block = newline.join(NEW_ARP_LINES).replace("{version}", match.group(1))
    text = text[: match.start()] + block + text[match.end() :]
    write(path, text.replace(OLD_ICON_REG, NEW_ICON_REG))
    return "updated: the Windows Apps list shows the installed version and the app icon"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Move the Tools pin to v0.3.0-alpha.3 and fix the Windows Apps-list entry (idempotent, surgical, never raises for a customised file)."""
    _ = context
    changed = 0
    steps = ((POST_INSTALL_SH, lambda p: move_pin(p, SH_PIN_LINE)), (POST_INSTALL_BAT, lambda p: move_pin(p, BAT_PIN_LINE)),
             (POST_INSTALL_BAT, fix_apps_list))
    for relative, step in steps:
        path = repo_root / relative
        if not path.is_file():
            log(f"{relative}: not present in this repository (nothing was created)")
            continue
        try:
            status = step(path)
        except (OSError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        changed += status.startswith("updated")
        log(f"{relative}: {status}")
    if changed:
        log(f"done: {changed} change(s). They reach users with the next release of the app. See {DOCS_URL}")
    else:
        log("done: no repository changes were required.")

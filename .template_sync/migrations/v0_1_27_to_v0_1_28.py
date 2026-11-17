"""Template 0.1.27 -> 0.1.28: the installers use LabConstrictor-Tools v0.3.0-alpha.2, which registers an app's host_requirements.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_28.md

Tools v0.3.0-alpha.2 (feature host_requirements_v1, LabConstrictor-Tools #155) copies the `host_requirements:` block of labconstrictor-app.yaml
into the registry entry, so napari, Fiji and QuPath can tell a person with an older plugin to update it. Older Tools ignore the block. What
this changes in an app repository (each step is independent, idempotent and never deletes anything):

1. `LC_TOOLS_PIN` in app/bash_bat_scripts/post_install.sh and post_install.bat moves from `v0.3.0-alpha.1` to `v0.3.0-alpha.2`, ONLY when the
   line still holds exactly the old template value. An app with its own pin, a script without a pin line or without the Tools hook stays as it is,
   with a note.
2. `.tools/docs/tools_bridge.md`: the tag name `v0.3.0-alpha.1` is replaced by `v0.3.0-alpha.2` (the guide names the pinned tag).

The pin reaches users with the app's next release.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_28.md"
TAG = "[migration 0.1.28]"
OLD_PIN = "v0.3.0-alpha.1"
NEW_PIN = "v0.3.0-alpha.2"
POST_INSTALL_SH = Path("app/bash_bat_scripts/post_install.sh")
POST_INSTALL_BAT = Path("app/bash_bat_scripts/post_install.bat")
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
SH_PIN_LINE = re.compile(r'^([ \t]*LC_TOOLS_PIN=")([^"]*)(".*)$')
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


def refresh_guide(path: Path) -> str:
    text = read(path)
    if OLD_PIN not in text:
        return "does not name the old tag: unchanged"
    write(path, text.replace(OLD_PIN, NEW_PIN))
    return f"updated: names {NEW_PIN}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Move the Tools pin to v0.3.0-alpha.2 and refresh the Tools guide (idempotent, surgical, never raises for a customised file)."""
    _ = context
    changed = 0
    steps = ((POST_INSTALL_SH, lambda p: move_pin(p, SH_PIN_LINE)), (POST_INSTALL_BAT, lambda p: move_pin(p, BAT_PIN_LINE)),
             (TOOLS_DOC, refresh_guide))
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
        log(f"done: {changed} file(s) changed. The pin reaches users with the next release of the app. See {DOCS_URL}")
    else:
        log("done: no repository changes were required.")

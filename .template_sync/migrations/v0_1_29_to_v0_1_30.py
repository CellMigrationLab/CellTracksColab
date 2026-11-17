"""Template 0.1.29 -> 0.1.30: the website's address is its GitHub Pages address, and two text corrections.

What this changes in an app repository. Every change replaces one exact piece of text and is skipped when that text is not there (a file
the app changed itself stays as it is, with a note). Each step is independent and idempotent; nothing is created or deleted.

1. The address https://labconstrictor.cellmig.org/ becomes https://cellmigrationlab.github.io/LabConstrictor-Site/ (the site stays at
   its GitHub Pages address) in the template's files: .tools/docs/app_manifest.md, .tools/docs/install.md, .tools/schema/README.md and
   .tools/schema/labconstrictor-app.schema.json (its `$id` and the line it tells editors to add).
2. In the app's own labconstrictor-app.yaml and docs/labconstrictor-project.yaml only the editor line
   (`# yaml-language-server: $schema=...`) gets the new address, so editors can load the schema again. Nothing else in them is touched.
3. .tools/docs/use_cases.md: the VLab4Mic logo is loaded from app/logo/logo.png (the copy under src/logo/ was removed).
4. .tools/docs/tools_bridge.md: people have installed apps on Windows and macOS from a clean system.

The changes reach users with the app's next release.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TAG = "[migration 0.1.30]"

OLD_URL = "https://labconstrictor.cellmig.org/"
NEW_URL = "https://cellmigrationlab.github.io/LabConstrictor-Site/"

MANIFEST_DOC = Path(".tools/docs/app_manifest.md")
INSTALL_DOC = Path(".tools/docs/install.md")
SCHEMA_README = Path(".tools/schema/README.md")
SCHEMA = Path(".tools/schema/labconstrictor-app.schema.json")
USE_CASES_DOC = Path(".tools/docs/use_cases.md")
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
APP_MANIFEST = Path("labconstrictor-app.yaml")
PROJECT_FILE = Path("docs/labconstrictor-project.yaml")

# Template files: every occurrence of the old address is replaced.
ADDRESS_FILES = (MANIFEST_DOC, INSTALL_DOC, SCHEMA_README, SCHEMA)
# The app's own files: only the editor line is rewritten.
EDITOR_LINE_FILES = (APP_MANIFEST, PROJECT_FILE)
EDITOR_LINE = re.compile(r"^(# yaml-language-server: \$schema=)" + re.escape(OLD_URL), re.MULTILINE)

# (old, new) pairs, written with LF line endings; a file with CRLF line endings gets them with CRLF.
EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    USE_CASES_DOC: (
        ("https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor-VLab4Mic/main/src/logo/logo.png",
         "https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor-VLab4Mic/main/app/logo/logo.png"),
    ),
    TOOLS_DOC: (
        ("machine, but it has not yet been tested by a person on Windows.\n",
         "machine, and people have installed apps on Windows and macOS from a clean system.\n"),
    ),
}
FILES = (*ADDRESS_FILES, *EDITOR_LINE_FILES, *EDITS)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def apply_edits(text_lf: str, edits: tuple[tuple[str, str], ...]) -> tuple[str, int, int]:
    """`text_lf` with each pair whose old text occurs exactly once replaced. Returns the text, the pairs applied and the pairs not found
    (neither the old text once nor the new text present)."""
    applied = missing = 0
    for old, new in edits:
        if text_lf.count(old) == 1:
            text_lf = text_lf.replace(old, new)
            applied += 1
        elif new not in text_lf:
            missing += 1
    return text_lf, applied, missing


def changed_text(relative: Path, text_lf: str) -> tuple[str, list[str]]:
    """The text of the file at `relative` after this migration, and what was done to it."""
    notes: list[str] = []
    if relative in ADDRESS_FILES:
        count = text_lf.count(OLD_URL)
        text_lf = text_lf.replace(OLD_URL, NEW_URL)
        if count:
            notes.append(f"{count} address(es) changed")
    elif relative in EDITOR_LINE_FILES:
        text_lf, count = EDITOR_LINE.subn(lambda match: match.group(1) + NEW_URL, text_lf)
        if count:
            notes.append("the editor line names the new address")
    else:
        text_lf, applied, missing = apply_edits(text_lf, EDITS[relative])
        if applied:
            notes.append(f"{applied} text correction(s)")
        if missing:
            notes.append(f"{missing} passage(s) differ from the template text and were left as they are")
    return text_lf, notes


def update(path: Path, relative: Path) -> str:
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    original = data.replace("\r\n", "\n")
    text, notes = changed_text(relative, original)
    if text == original:
        return "unchanged: " + ("; ".join(notes) if notes else "already up to date")
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return "updated: " + "; ".join(notes)


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply the 0.1.30 address change and text corrections (idempotent, surgical, never raises for a customised file)."""
    _ = context
    changed = 0
    for relative in FILES:
        path = repo_root / relative
        if not path.is_file():
            log(f"{relative}: not present in this repository (nothing was created)")
            continue
        try:
            status = update(path, relative)
        except (OSError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged"
        changed += status.startswith("updated")
        log(f"{relative}: {status}")
    if changed:
        log(f"done: {changed} file(s) changed. The changes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

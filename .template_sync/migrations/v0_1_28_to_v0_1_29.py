"""Template 0.1.28 -> 0.1.29: text corrections, and the notebook changelog no longer quotes commit trailers.

What this changes in an app repository. Every change replaces one exact piece of template text and is skipped when that text is not
there (a file the app changed itself stays as it is, with a note). Each step is independent and idempotent; nothing is created or deleted.

1. .tools/python/update_changelog.py: the notebook CHANGELOG.md quotes the subject line of the commit (`--pretty=%s`) instead of the whole
   message (`--pretty=%B`), so trailers such as Co-authored-by are no longer published with the app.
2. app/bash_bat_scripts/post_install.sh and post_install.bat: the installer log says the tools are registered "for Napari, Fiji and QuPath".
3. .tools/docs/tools_bridge.md: the guide names QuPath next to Napari and Fiji, links the plugins' current repositories
   (LabConstrictor-napari, LabConstrictor-Fiji, LabConstrictor-QuPath) and says what is known about Windows.
4. .tools/docs/app_manifest.md: the host_requirements example names plugin version 0.1.0-alpha.7.
5. .tools/docs/use_cases.md: the VLab4Mic logo and app badge point to CellMigrationLab/LabConstrictor-VLab4Mic.
6. .tools/docs/README.md: the tools guide entry names QuPath, and the per-version design notes (migration_0_1_*.md) are no longer listed
   as user documentation (the files stay; the migrations link to them).

The changes reach users with the app's next release.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TAG = "[migration 0.1.29]"

UPDATE_CHANGELOG = Path(".tools/python/update_changelog.py")
POST_INSTALL_SH = Path("app/bash_bat_scripts/post_install.sh")
POST_INSTALL_BAT = Path("app/bash_bat_scripts/post_install.bat")
TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
MANIFEST_DOC = Path(".tools/docs/app_manifest.md")
USE_CASES_DOC = Path(".tools/docs/use_cases.md")
DOCS_INDEX = Path(".tools/docs/README.md")

# (old, new) pairs, written with LF line endings; a file with CRLF line endings gets them with CRLF.
EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    UPDATE_CHANGELOG: (
        ('    """Get current commit hash and message"""\n',
         '    """Get current commit hash and message (the subject line only: the body and its trailers, such as\n'
         '    Co-authored-by, do not belong in the notebook\'s changelog)"""\n'),
        ("subprocess.check_output(['git', 'log', '-1', '--pretty=%B'])", "subprocess.check_output(['git', 'log', '-1', '--pretty=%s'])"),
    ),
    POST_INSTALL_SH: (
        (' for Napari and Fiji"\n', ' for Napari, Fiji and QuPath"\n'),
    ),
    POST_INSTALL_BAT: (
        (' for Napari and Fiji. >> "%LOG_FILE%"\n', ' for Napari, Fiji and QuPath. >> "%LOG_FILE%"\n'),
    ),
    TOOLS_DOC: (
        ("# Expose your app's tools to Napari and Fiji (optional, experimental)\n",
         "# Expose your app's tools to Napari, Fiji and QuPath (optional, experimental)\n"),
        ("as **tools**. After installation they appear in a generic **Napari** widget and a generic **Fiji** command, and work from the\n",
         "as **tools**. After installation they appear in a generic **Napari** widget, a generic **Fiji** command and a generic **QuPath** command,"
         " and work from the\n"),
        ("4. Install the Napari plugin (`napari-labconstrictor`) or the Fiji plugin (`labconstrictor-fiji`) once per machine. Neither is on PyPI or a"
         " Fiji update site yet: see the READMEs of\n"
         "   [napari-labconstrictor](https://github.com/CellMigrationLab/napari-labconstrictor) and"
         " [LabConstrictor-Fiji](https://github.com/CellMigrationLab/LabConstrictor-Fiji) for the install commands.\n",
         "4. Install the Napari plugin (`napari-labconstrictor`), the Fiji plugin (`labconstrictor-fiji`) or the QuPath extension once per machine."
         " None of them is on PyPI or an update site yet:\n"
         "   see the READMEs of [LabConstrictor-napari](https://github.com/CellMigrationLab/LabConstrictor-napari),\n"
         "   [LabConstrictor-Fiji](https://github.com/CellMigrationLab/LabConstrictor-Fiji) and"
         " [LabConstrictor-QuPath](https://github.com/CellMigrationLab/LabConstrictor-QuPath) for the install steps.\n"),
        ("The Windows hook (`post_install.bat`) has not yet been run on a real Windows computer; if you can help, follow the\n"
         "[human test protocol](https://github.com/CellMigrationLab/LabConstrictor-Tools/blob/main/docs/HUMAN_TEST_PROTOCOL.md).\n",
         "The Windows hook (`post_install.bat`) runs in the template's automated tests on a Windows\n"
         "machine, but it has not yet been tested by a person on Windows.\n"),
    ),
    MANIFEST_DOC: (
        ("    min_version: 0.1.0-alpha.6                  # a release tag of the plugin, with or without v\n",
         "    min_version: 0.1.0-alpha.7                  # a release tag of the plugin, with or without v\n"),
        ("  fiji:\n    min_version: 0.1.0-alpha.6\n", "  fiji:\n    min_version: 0.1.0-alpha.7\n"),
    ),
    USE_CASES_DOC: (
        ("https://raw.githubusercontent.com/HenriquesLab/LabConstrictor-VLab4Mic/main/src/logo/logo.png",
         "https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor-VLab4Mic/main/src/logo/logo.png"),
        ("-181717?logo=github)](https://github.com/HenriquesLab/LabConstrictor-VLab4Mic)",
         "-181717?logo=github)](https://github.com/CellMigrationLab/LabConstrictor-VLab4Mic)"),
    ),
    DOCS_INDEX: (
        ("- [Expose tools to Napari and Fiji (experimental)](tools_bridge.md)\n",
         "- [Expose tools to Napari, Fiji and QuPath (experimental)](tools_bridge.md)\n"),
    ),
}
DESIGN_NOTE_ENTRY = re.compile(r"^- \[Template 0\.1\.\d+: [^\n]*\]\(migration_0_1_\d+\.md\)[ \t]*\n", re.MULTILINE)


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


def update(path: Path, relative: Path) -> str:
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    original = data.replace("\r\n", "\n")
    text, applied, missing = apply_edits(original, EDITS[relative])
    notes = [f"{applied} text correction(s)"] if applied else []
    if relative == DOCS_INDEX:
        text, removed = DESIGN_NOTE_ENTRY.subn("", text)
        if removed:
            notes.append(f"{removed} design note entries removed from the list")
    if missing:
        notes.append(f"{missing} passage(s) differ from the template text and were left as they are")
    if text == original:
        return "unchanged: " + ("; ".join(notes) if notes else "already up to date")
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return "updated: " + "; ".join(notes)


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply the 0.1.29 text corrections (idempotent, surgical, never raises for a customised file)."""
    _ = context
    changed = 0
    for relative in EDITS:
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

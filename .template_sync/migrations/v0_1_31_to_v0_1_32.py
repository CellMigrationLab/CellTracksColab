"""Template 0.1.31 -> 0.1.32: the user docs link to the guide that owns a topic instead of repeating it.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_32.md

Each topic of the LabConstrictor documentation is written on one page, and every other page links to it. The plugins' install steps belong
to the plugin guides on the website, and how to write tools belongs to LabConstrictor-Tools. The template keeps what is its own: where an
app declares its tools and what the installer does with them.

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. .tools/docs/tools_bridge.md is replaced by the 0.1.32 guide when it is still the 0.1.31 template file. The guide no longer repeats the
   plugins' install steps, the list of form markers and the test commands of LabConstrictor-Tools, or the version of the Tools pin; it links
   to the owner guides instead. A file the app changed itself stays as it is (SKIPPED).
2. .tools/docs/install.md: "Where the app is listed" writes napari in lower case and links to the guide that says how to run the tools.
3. .tools/docs/README.md: the entry of the tools guide writes napari in lower case.

Steps 2 and 3 replace one exact piece of template text each and leave a file alone when that text is not there (SKIPPED). The line endings
of each file are kept. The changes reach users with the app's next release.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_32.md"
TAG = "[migration 0.1.32]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_31_to_v0_1_32"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

TOOLS_DOC = Path(".tools/docs/tools_bridge.md")
INSTALL_DOC = Path(".tools/docs/install.md")
DOCS_INDEX = Path(".tools/docs/README.md")

SITE = "https://cellmigrationlab.github.io/LabConstrictor-Site/"

# Files replaced whole, only when they are the 0.1.31 template file (compared with LF line endings).
WHOLE_FILES = (TOOLS_DOC,)

# (old, new) pairs, written with LF line endings; a file with CRLF line endings gets them with CRLF.
EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    INSTALL_DOC: (
        ("If the app ships tools for Napari, Fiji or QuPath, the installer registers them with LabConstrictor Tools",
         "If the app ships tools for napari, Fiji or QuPath, the installer registers them with LabConstrictor Tools"),
        ("with `hosts: [notebook, napari]` the app is listed in Napari, and not in Fiji or QuPath.",
         "with `hosts: [notebook, napari]` the app is listed in napari, and not in Fiji or QuPath."),
        ("so it is listed in every host; ...`).\n",
         "so it is listed in every host; ...`).\n"
         "\n"
         "To run the tools, install the plugin for your image software once: see"
         f" [Use apps inside napari, Fiji and QuPath]({SITE}guides/image-software/).\n"),
    ),
    DOCS_INDEX: (
        ("- [Expose tools to Napari, Fiji and QuPath (experimental)](tools_bridge.md)\n",
         "- [Expose tools to napari, Fiji and QuPath (experimental)](tools_bridge.md)\n"),
    ),
}
FILES = (*WHOLE_FILES, *EDITS)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def apply_edits(text_lf: str, edits: tuple[tuple[str, str], ...]) -> tuple[str, int, int]:
    """`text_lf` with each pair applied whose old text occurs exactly once and whose new text is not there yet. Returns the text, the
    pairs applied and the pairs not found (neither the old text once nor the new text present)."""
    applied = missing = 0
    for old, new in edits:
        if new in text_lf:
            continue
        if text_lf.count(old) == 1:
            text_lf = text_lf.replace(old, new)
            applied += 1
        else:
            missing += 1
    return text_lf, applied, missing


def replaced_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The 0.1.32 text of a whole-file step, and what was done. The text is unchanged when the file is not the 0.1.31 template file."""
    new = (RESOURCE_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if text_lf == new:
        return text_lf, "already up to date"
    previous = (PREVIOUS_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if text_lf != previous:
        return text_lf, f"SKIPPED: customised (not the 0.1.31 template file), so it was left unchanged; compare it with the template's file. See {DOCS_URL}"
    return new, "replaced by the 0.1.32 guide"


def edited_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The text after the exact-text edits of `relative`, and what was done."""
    edits = EDITS[relative]
    text, applied, missing = apply_edits(text_lf, edits)
    if not missing:
        return text, f"{applied} text change(s)" if applied else "already up to date"
    if missing == len(edits):
        return text, f"SKIPPED: customised (the 0.1.31 text was not found), so it was left unchanged. See {DOCS_URL}"
    skipped = f"SKIPPED {missing} passage(s) that differ from the template text (left as they are). See {DOCS_URL}"
    return text, f"{applied} text change(s); {skipped}" if applied else f"the other changes are there; {skipped}"


def update(path: Path, relative: Path) -> str:
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    original = data.replace("\r\n", "\n")
    text, note = replaced_file(relative, original) if relative in WHOLE_FILES else edited_file(relative, original)
    if text == original:
        return note if note.startswith("SKIPPED") else f"unchanged: {note}"
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return f"updated: {note}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply the 0.1.32 documentation changes (idempotent, surgical, never raises for a customised file)."""
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
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        changed += status.startswith("updated")
        log(f"{relative}: {status}")
    if changed:
        log(f"done: {changed} file(s) changed. The changes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

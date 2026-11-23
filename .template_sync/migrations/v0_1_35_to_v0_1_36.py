"""Template 0.1.35 -> 0.1.36: the napari, Fiji and QuPath plugins are no longer called experimental, and the page of examples is gone.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_36.md

The plugins that run an app's tools in napari, Fiji and QuPath are now an implemented way to use an app, not an experiment. The "Examples"
section of the guides had one page, use_cases.md, which since 0.1.35 only pointed to the App Centre; the App Centre lists the apps, so the
page and the links to it go.

What this changes in an app repository (every step is independent and idempotent):

1. .tools/docs/use_cases.md is removed when it is still the 0.1.35 template file. A file the app changed itself stays as it is (SKIPPED): it
   is the app's own page. This is the only file the migration deletes.
2. .tools/docs/tools_bridge.md: "experimental" is dropped from the title.
3. .tools/docs/README.md: "experimental" is dropped from the entry of the tools guide, and the "Examples" section (use_cases.md) goes.
4. README.md: "experimental" is dropped from the line of the tools guide, and the "See what others have built" line (use_cases.md) goes.

Steps 2 to 4 replace exact pieces of template text and leave a passage alone when its text is not there (SKIPPED). The line endings of each
file are kept. The changes reach users with the app's next release.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_36.md"
TAG = "[migration 0.1.36]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_35_to_v0_1_36"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

DOCS = Path(".tools/docs")
USE_CASES_DOC = DOCS / "use_cases.md"
TOOLS_DOC = DOCS / "tools_bridge.md"
DOCS_INDEX = DOCS / "README.md"
ROOT_README = Path("README.md")

# Files removed, only when they are the 0.1.35 template file (compared with LF line endings).
REMOVED_FILES = (USE_CASES_DOC,)

# (old, new) pairs, written with LF line endings; a file with CRLF line endings gets them with CRLF. Each new text is distinct from the
# old one and not contained in it, so a pair that removes a line keeps its neighbours in both texts.
EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    TOOLS_DOC: (
        ("# Expose your app's tools to napari, Fiji and QuPath (optional, experimental)\n",
         "# Expose your app's tools to napari, Fiji and QuPath (optional)\n"),
    ),
    DOCS_INDEX: (
        ("- [Expose tools to napari, Fiji and QuPath (experimental)](tools_bridge.md)\n",
         "- [Expose tools to napari, Fiji and QuPath](tools_bridge.md)\n"),
        ("- [Hide code cells in notebooks](code_hiding.md)\n\n## Examples\n\n- [LabConstrictor use cases](use_cases.md)\n\n## GitHub management\n",
         "- [Hide code cells in notebooks](code_hiding.md)\n\n## GitHub management\n"),
    ),
    ROOT_README: (
        ("- [Offer your app's functions in napari, Fiji and QuPath](.tools/docs/tools_bridge.md) (experimental)\n",
         "- [Offer your app's functions in napari, Fiji and QuPath](.tools/docs/tools_bridge.md)\n"),
        ("(.tools/docs/notebook_portability.md)\n- [See what others have built](.tools/docs/use_cases.md)\n- [All guides]",
         "(.tools/docs/notebook_portability.md)\n- [All guides]"),
    ),
}
FILES = (*REMOVED_FILES, *EDITS)


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


def edited_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The text after the exact-text edits of `relative`, and what was done."""
    edits = EDITS[relative]
    text, applied, missing = apply_edits(text_lf, edits)
    if not missing:
        return text, f"{applied} text change(s)" if applied else "already up to date"
    if missing == len(edits):
        return text, f"SKIPPED: customised (the 0.1.35 text was not found), so it was left unchanged. See {DOCS_URL}"
    skipped = f"SKIPPED {missing} passage(s) that differ from the template text (left as they are). See {DOCS_URL}"
    return text, f"{applied} text change(s); {skipped}" if applied else f"the other changes are there; {skipped}"


def remove(path: Path, relative: Path) -> str:
    """Remove `path` when it is the 0.1.35 template file of `relative`, and say what was done."""
    text_lf = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    previous = (PREVIOUS_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if text_lf != previous:
        return f"SKIPPED: customised (not the 0.1.35 template file), so it was kept as the app's own page; the template no longer has it. See {DOCS_URL}"
    path.unlink()
    return "updated: removed (it was the template file; the App Centre lists the apps)"


def update(path: Path, relative: Path) -> str:
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    original = data.replace("\r\n", "\n")
    text, note = edited_file(relative, original)
    if text == original:
        return note if note.startswith("SKIPPED") else f"unchanged: {note}"
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return f"updated: {note}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply the 0.1.36 documentation changes (idempotent, surgical, never raises for a customised file, removes only an unchanged file)."""
    _ = context
    changed = 0
    for relative in FILES:
        path = repo_root / relative
        if not path.is_file():
            log(f"{relative}: not present in this repository (nothing was created)")
            continue
        try:
            status = remove(path, relative) if relative in REMOVED_FILES else update(path, relative)
        except (OSError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        changed += status.startswith("updated")
        log(f"{relative}: {status}")
    if changed:
        log(f"done: {changed} file(s) changed. The changes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

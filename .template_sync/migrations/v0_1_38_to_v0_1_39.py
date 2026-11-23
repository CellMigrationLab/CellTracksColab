"""Template 0.1.38 -> 0.1.39: the installer intros say that a release builds only the systems the app lists.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_39.md

before_getting_started.md and executable_creation.md said that a release builds installers for Windows, macOS and Linux. It builds only
the systems listed under `distribution.platforms` in labconstrictor-app.yaml (all of them when the list is missing); both intros now say so
and link the manifest guide.

What this changes in an app repository: each file in FILES is replaced by the 0.1.39 page when it is still the 0.1.38 template file
(compared with LF line endings; the file keeps its own line endings). A file the app changed itself stays as it is (SKIPPED, with a note of
what to change by hand). A missing file is not created, and nothing is deleted. The changes reach users with the app's next release.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_39.md"
TAG = "[migration 0.1.39]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_38_to_v0_1_39"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

DOCS = Path(".tools/docs")

# Files replaced whole, only when they are the 0.1.38 template file (compared with LF line endings).
FILES = (DOCS / "before_getting_started.md", DOCS / "executable_creation.md")

# What to change by hand in a customised file (added to the SKIPPED note).
BY_HAND = (
    "where the page says a release builds installers for Windows, macOS and Linux, add that it builds only the systems listed under "
    "distribution.platforms in labconstrictor-app.yaml (link app_manifest.md#distribution-how-people-get-the-app)"
)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def lf_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def replaced_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The 0.1.39 text of `relative` for the app's `text_lf`, and what was done. The text is unchanged when it is not the 0.1.38 template file."""
    new = lf_text(RESOURCE_ROOT / relative)
    previous = lf_text(PREVIOUS_ROOT / relative)
    if text_lf == new:
        return text_lf, "already up to date"
    if text_lf == previous:
        return new, "replaced by the 0.1.39 page"
    return text_lf, (
        "SKIPPED: customised (not the 0.1.38 template file), so it was left unchanged; compare it with the template's page. "
        f"By hand: {BY_HAND}. See {DOCS_URL}"
    )


def update(path: Path, relative: Path) -> str:
    data = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in data else "\n"
    original = data.replace("\r\n", "\n")
    text, note = replaced_file(relative, original)
    if text == original:
        return note if note.startswith("SKIPPED") else f"unchanged: {note}"
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return f"updated: {note}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply the 0.1.39 documentation changes (idempotent, never raises for a customised file, creates and deletes nothing)."""
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

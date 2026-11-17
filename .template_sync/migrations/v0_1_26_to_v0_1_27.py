"""Template 0.1.26 -> 0.1.27: an app can say which napari, Fiji or QuPath plugin version its tools need (host_requirements).

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_27.md
Proposal: https://github.com/CellMigrationLab/LabConstrictor-Tools/issues/155

The App Centre schema gained an optional field, host_requirements. The template's copy of the schema rejects fields it does not
know, so without this migration an app that adds the field fails its "Check App Centre manifest" workflow. What this changes in an
app repository (each step is independent, idempotent and never deletes anything):

1. .tools/schema/labconstrictor-app.schema.json: replaced by the template's copy (the template owns it; it must equal the App Centre's
   schema). A missing schema is not created: the app has no manifest check to update.
2. .tools/docs/app_manifest.md: the section "Plugin versions your tools need (optional)" goes before "## The automatic check". A page
   that already has the section is unchanged; a page without that heading gets it before its Previous / Home / Next links, or at
   its end. Line endings are kept.

labconstrictor-app.yaml belongs to the app and is never created or changed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_27.md"
TAG = "[migration 0.1.27]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_26_to_v0_1_27"

SCHEMA = Path(".tools/schema/labconstrictor-app.schema.json")
MANIFEST_PAGE = Path(".tools/docs/app_manifest.md")
SECTION = Path(".tools/docs/app_manifest_host_requirements.md")

SECTION_HEADING = "## Plugin versions your tools need (optional)"
ANCHOR = "## The automatic check"
FOOTER = re.compile(r'^---[ \t]*\n(?:[ \t]*\n)*<div align="center">.*?</div>[ \t]*(?:\n|$)', re.MULTILINE | re.DOTALL)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def resource_bytes(relative: Path) -> bytes:
    return (RESOURCE_ROOT / relative).read_bytes()


def with_section(text_lf: str, section: str) -> str:
    """The page with the section before "## The automatic check", else before its links, else at its end."""
    lines = text_lf.split("\n")
    for index, line in enumerate(lines):
        if line.rstrip() == ANCHOR:
            before = "\n".join(lines[:index]).rstrip("\n")
            return before + "\n\n" + section.rstrip("\n") + "\n\n" + "\n".join(lines[index:])
    footers = list(FOOTER.finditer(text_lf))
    if footers:
        start = footers[-1].start()
        return text_lf[:start].rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n\n" + text_lf[start:]
    return text_lf.rstrip("\n") + "\n\n" + section


def update_schema(repo_root: Path) -> str:
    path = repo_root / SCHEMA
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    data = resource_bytes(SCHEMA)
    if path.read_bytes() == data:
        return "unchanged: already the template's copy"
    path.write_bytes(data)
    return "updated: the template's copy (accepts host_requirements)"


def update_page(repo_root: Path) -> str:
    path = repo_root / MANIFEST_PAGE
    if not path.is_file():
        return "not present in this repository (nothing was created)"
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text_lf = raw.replace("\r\n", "\n")
    if SECTION_HEADING in (line.rstrip() for line in text_lf.split("\n")):
        return "unchanged: already has the section on plugin versions"
    section = resource_bytes(SECTION).decode("utf-8").replace("\r\n", "\n")
    path.write_bytes(with_section(text_lf, section).replace("\n", newline).encode("utf-8"))
    return "updated: added the section on plugin versions"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Refresh the schema copy and document host_requirements (idempotent, never deletes)."""
    _ = context
    changed = 0
    for name, step in ((SCHEMA, update_schema), (MANIFEST_PAGE, update_page)):
        try:
            status = step(repo_root)
        except (OSError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        changed += status.startswith("updated")
        log(f"{name}: {status}")
    if changed:
        log(f"done: {changed} file(s) changed. See {DOCS_URL}")
    else:
        log("done: no repository changes were required.")

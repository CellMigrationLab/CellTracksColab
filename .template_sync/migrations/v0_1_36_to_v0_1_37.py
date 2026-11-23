"""Template 0.1.36 -> 0.1.37: the user guides are rewritten for the website's three tracks.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_37.md

The guides were checked against the template, the repository form and the installer, and rewritten to be shorter and plainer. The guides a
first-time reader follows to the end (install, initialise the repository, generate requirements, upload a notebook, create the installers)
now say what you need, how to check that it worked and what to do if it does not.

What this changes in an app repository: each guide in FILES is replaced by the 0.1.37 page when it is still the 0.1.36 template file
(compared with LF line endings; the file keeps its own line endings). A guide the app changed itself stays as it is (SKIPPED, with a note).
A missing guide is not created, and nothing is deleted. external_code_upload.md is also recognised with the app's Python package name in
place of PYTHON_PROJ_NAME (the repository form writes it there), and the new page gets the same name. The changes reach users with the
app's next release.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_37.md"
TAG = "[migration 0.1.37]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_36_to_v0_1_37"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

DOCS = Path(".tools/docs")
EXTERNAL_CODE_DOC = DOCS / "external_code_upload.md"

# Files replaced whole, only when they are the 0.1.36 template file (compared with LF line endings).
FILES = tuple(
    DOCS / name
    for name in (
        "accept_pull_request.md",
        "app_manifest.md",
        "before_getting_started.md",
        "create_repository.md",
        "executable_creation.md",
        "external_code_upload.md",
        "initialise_repository.md",
        "install.md",
        "installer_images.md",
        "notebook_requirements.md",
        "notebook_upload.md",
        "notebook_usage.md",
        "template_synchronization.md",
        "tools_bridge.md",
        "troubleshooting.md",
        "troubleshooting_installed_app.md",
        "workflow_status.md",
    )
)

# The placeholder the repository form replaces in a guide, by guide: the form writes the app's Python package name in place of it.
RENDERED = {EXTERNAL_CODE_DOC: "PYTHON_PROJ_NAME"}
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


def log(message: str) -> None:
    print(f"{TAG} {message}")


def lf_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def filled_in_name(template_lf: str, text_lf: str, placeholder: str) -> str | None:
    """The name written in `text_lf` everywhere `template_lf` has `placeholder` (the same name each time), or None when the texts differ
    elsewhere."""
    if placeholder not in template_lf:
        return None
    parts = template_lf.split(placeholder)
    pattern = re.escape(parts[0]) + f"(?P<name>{IDENTIFIER})" + "".join(re.escape(part) + "(?P=name)" for part in parts[1:-1])
    pattern += re.escape(parts[-1])
    match = re.fullmatch(pattern, text_lf, flags=re.DOTALL)
    return match.group("name") if match else None


def replaced_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The 0.1.37 text of `relative` for the app's `text_lf`, and what was done. The text is unchanged when it is not the 0.1.36 template file."""
    new = lf_text(RESOURCE_ROOT / relative)
    previous = lf_text(PREVIOUS_ROOT / relative)
    if text_lf == new:
        return text_lf, "already up to date"
    if text_lf == previous:
        return new, "replaced by the 0.1.37 page"
    placeholder = RENDERED.get(relative)
    if placeholder:
        name = filled_in_name(new, text_lf, placeholder)
        if name:
            return text_lf, "already up to date"
        name = filled_in_name(previous, text_lf, placeholder)
        if name:
            return new.replace(placeholder, name), f"replaced by the 0.1.37 page (with the package name {name})"
    return text_lf, f"SKIPPED: customised (not the 0.1.36 template file), so it was left unchanged; compare it with the template's page. See {DOCS_URL}"


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
    """Apply the 0.1.37 documentation changes (idempotent, never raises for a customised file, creates and deletes nothing)."""
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

"""Template 0.1.34 -> 0.1.35: the user docs say how to uninstall an app, link to the website's guides, and use its heading style.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_35.md

The website (LabConstrictor-Site) publishes the template's .tools/docs as its guides. A review of the website on 2026-11-17 found errors in
them: a link to an uninstall guide that does not exist, links to a GitHub file where the website has the guide, a list of apps that drifts
from the App Centre, and headings that break the website's style. This migration brings the same fixes to an app repository.

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. .tools/docs/use_cases.md is replaced by the 0.1.35 page, which links to the App Centre instead of keeping its own list of apps, when it is
   still the 0.1.34 template file. A file the app changed itself stays as it is (SKIPPED).
2. .tools/docs/install.md gets an "Uninstall an app" section; the update steps link to it.
3. .tools/docs/troubleshooting_installed_app.md links to that section instead of download_executable.md.
4. .tools/docs/tools_bridge.md links to the website's "Writing tools" guide instead of the file on GitHub.
5. .tools/docs/notebook_usage.md names the applications menu on Linux (not the Start Menu).
6. .tools/docs/notebook_upload.md marks the version-check cell as Python.
7. Headings in sentence case, without emoji or trailing colons: code_hiding.md, before_getting_started.md, executable_creation.md,
   workflow_status.md, installer_images.md.

Steps 2 to 7 replace exact pieces of template text and leave a passage alone when its text is not there (SKIPPED). The line endings of each
file are kept. The changes reach users with the app's next release.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_35.md"
TAG = "[migration 0.1.35]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_34_to_v0_1_35"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

DOCS = Path(".tools/docs")
USE_CASES_DOC = DOCS / "use_cases.md"
INSTALL_DOC = DOCS / "install.md"
TROUBLESHOOTING_DOC = DOCS / "troubleshooting_installed_app.md"
TOOLS_DOC = DOCS / "tools_bridge.md"
USAGE_DOC = DOCS / "notebook_usage.md"
UPLOAD_DOC = DOCS / "notebook_upload.md"

SITE = "https://cellmigrationlab.github.io/LabConstrictor-Site/"
AUTHORING_ON_GITHUB = "https://github.com/CellMigrationLab/LabConstrictor-Tools/blob/main/docs/AUTHORING.md"
WRITING_TOOLS = f"{SITE}guides/writing-tools/"

UNINSTALL_SECTION = (
    "## Uninstall an app\n"
    "\n"
    "Uninstalling deletes the app's install folder and everything in it: move anything you saved there to another folder first.\n"
    "\n"
    "- **Windows.** Open Settings > Apps > Installed apps, find the app, click the three dots next to it and choose Uninstall. You can also"
    " double-click `Uninstall-<AppName>.exe` in the app's install folder. If the app is still listed afterwards, see"
    " [Troubleshooting an installed app](troubleshooting_installed_app.md).\n"
    "- **macOS and Linux.** Open a terminal and run the `uninstall.sh` script of the app's install folder (the folder you chose when you"
    " installed it), for example `bash ~/labconstrictor/MyApp/uninstall.sh`. It removes the app's shortcut (from the Applications folder of"
    " your home folder on macOS, from your applications menu on Linux), takes the app out of the tools list of napari, Fiji and QuPath, and"
    " deletes the install folder.\n"
    "\n"
)


def heading(old: str, new: str) -> tuple[str, str]:
    """A heading line, matched as a whole line."""
    return (f"\n{old}\n", f"\n{new}\n")


# Files replaced whole, only when they are the 0.1.34 template file (compared with LF line endings).
WHOLE_FILES = (USE_CASES_DOC,)

# (old, new) pairs, written with LF line endings; a file with CRLF line endings gets them with CRLF.
EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    INSTALL_DOC: (
        ("update an app by uninstalling the old version first and then installing the new one",
         "update an app by [uninstalling the old version](#uninstall-an-app) first and then installing the new one"),
        ("otherwise delete `<prefix>/pkgs` or uninstall the app and install it again.\n\n## Where the app is listed\n",
         "otherwise delete `<prefix>/pkgs` or uninstall the app and install it again.\n\n" + UNINSTALL_SECTION + "## Where the app is listed\n"),
    ),
    TROUBLESHOOTING_DOC: (
        ("as explained in the [uninstalling guide](./download_executable.md),",
         "as explained in [Uninstall an app](install.md#uninstall-an-app),"),
    ),
    TOOLS_DOC: (
        (f"- [Writing tools]({AUTHORING_ON_GITHUB}), in LabConstrictor-Tools: the inputs\n",
         f"- [Writing tools]({WRITING_TOOLS}): the inputs\n"),
        (f"Test your tools before you publish a release, as [Writing tools]({AUTHORING_ON_GITHUB})\n",
         f"Test your tools before you publish a release, as [Writing tools]({WRITING_TOOLS})\n"),
    ),
    USAGE_DOC: (
        ("go to the Start Menu on Windows or Linux, or the Applications folder on macOS.",
         "go to the Start Menu (Windows), the Applications folder (macOS) or your applications menu (Linux)."),
    ),
    UPLOAD_DOC: (
        ("so users can see there is a new version.\n\n```\n# @markdown Run this cell to check this notebook's version.\n",
         "so users can see there is a new version.\n\n```python\n# @markdown Run this cell to check this notebook's version.\n"),
    ),
    DOCS / "code_hiding.md": (
        heading("## How is this for?", "## Who is this for?"),
    ),
    DOCS / "before_getting_started.md": (
        heading("## 🎯 LabConstrictor in a Nutshell", "## LabConstrictor in a nutshell"),
        heading("### ⚙️ Easy Configuration", "### Easy configuration"),
        heading("### 🖥️ Cross-Platform Installers", "### Cross-platform installers"),
        heading("### 📦 Environment Management", "### Environment management"),
        heading("### 📈 Version Control", "### Version control"),
        heading("### ✨ User-Friendly Experience", "### User-friendly experience"),
        heading("### 📱 Offline Usage", "### Offline usage"),
        heading("## ✅ LabConstrictor Requirements", "## LabConstrictor requirements"),
    ),
    DOCS / "executable_creation.md": (
        heading("## ⚠️ Please check before proceeding:", "## Please check before proceeding"),
    ),
    DOCS / "workflow_status.md": (
        heading("## 1. Monitor the Progress", "## 1. Monitor the progress"),
        heading("## 2. How to Debug a Failure", "## 2. How to debug a failure"),
        heading("### 💡 Best Practice for Large Updates", "### Best practice for large updates"),
    ),
    DOCS / "installer_images.md": (
        heading("## Logo:", "## Logo"),
        heading("## Banner:", "## Banner"),
        heading("## Header:", "## Header"),
        heading("## Default Installer", "## Default installer"),
        heading("## Customised Installer", "## Customised installer"),
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
    """The 0.1.35 text of a whole-file step, and what was done. The text is unchanged when the file is not the 0.1.34 template file."""
    new = (RESOURCE_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if text_lf == new:
        return text_lf, "already up to date"
    previous = (PREVIOUS_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if text_lf != previous:
        return text_lf, f"SKIPPED: customised (not the 0.1.34 template file), so it was left unchanged; compare it with the template's file. See {DOCS_URL}"
    return new, "replaced by the 0.1.35 page"


def edited_file(relative: Path, text_lf: str) -> tuple[str, str]:
    """The text after the exact-text edits of `relative`, and what was done."""
    edits = EDITS[relative]
    text, applied, missing = apply_edits(text_lf, edits)
    if not missing:
        return text, f"{applied} text change(s)" if applied else "already up to date"
    if missing == len(edits):
        return text, f"SKIPPED: customised (the 0.1.34 text was not found), so it was left unchanged. See {DOCS_URL}"
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
    """Apply the 0.1.35 documentation changes (idempotent, surgical, never raises for a customised file)."""
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

"""Template 0.1.25 -> 0.1.26: the app's own documentation explains PyPI publishing (added in 0.1.25).

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_26.md

The 0.1.25 migration added publish_pypi.yml and pypi_package.py but left the app's documentation as it was, so the one-time setup was
described only in the template's documentation. What this changes in an app repository (each step is independent, idempotent and never
deletes anything):

1. .tools/docs/external_code_upload.md: the section "4. Publish the code on PyPI (optional)" of the template. A page with a section 4
   of its own is left as it is.
2. .tools/docs/troubleshooting.md: the entry "Publishing to PyPI fails" of the template.

Each text goes at the end of the page's content, before its Previous / Home / Next links, which stay the last thing on the page (they are
moved there if text followed them). A page without these links gets the text at its end.

Both texts are the template's, byte for byte (.template_sync/resources/v0_1_25_to_v0_1_26/.tools/docs/): they have no placeholder,
and they link to the design note of 0.1.25 by its address, so every link works in an app. The line endings of each file are kept.
"""
from __future__ import annotations

import importlib.util
import re
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_26.md"
TAG = "[migration 0.1.26]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_25_to_v0_1_26"

EXTERNAL_CODE_PAGE = Path(".tools/docs/external_code_upload.md")
TROUBLESHOOTING_PAGE = Path(".tools/docs/troubleshooting.md")
PYPI_SECTION = Path(".tools/docs/external_code_upload_pypi.md")
PYPI_TROUBLESHOOTING = Path(".tools/docs/troubleshooting_pypi.md")

SECTION_HEADING = "## 4. Publish the code on PyPI (optional)"
TROUBLESHOOTING_HEADING = "## Publishing to PyPI fails"
# A section 4 of the app's own (the template's page has sections 1 to 3 before 0.1.26).
OWN_SECTION_4 = re.compile(r"^##[ \t]+4\.", re.MULTILINE)
# The Previous / Home / Next links at the bottom of a documentation page: a "---" line, then <div align="center"> ... </div>.
FOOTER = re.compile(r'^---[ \t]*\n(?:[ \t]*\n)*<div align="center">.*?</div>[ \t]*(?:\n|$)', re.MULTILINE | re.DOTALL)


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load(name: str, alias: str) -> ModuleType:
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resource(relative: Path) -> str:
    return (RESOURCE_ROOT / relative).read_text(encoding="utf-8")


def has_line(text_lf: str, line: str) -> bool:
    return line in (candidate.rstrip() for candidate in text_lf.split("\n"))


def add_section(text_lf: str, section: str) -> str:
    """The page with the section at the end of its content, then its Previous / Home / Next links (always last), if it has them."""
    footers = list(FOOTER.finditer(text_lf))
    if not footers:
        return text_lf.rstrip("\n") + "\n\n" + section
    footer = footers[-1]
    before, after = text_lf[: footer.start()].rstrip("\n"), text_lf[footer.end():].strip("\n")
    content = f"{before}\n\n{after}" if after else before
    return content + "\n\n" + section + "\n" + footer.group(0).rstrip("\n") + "\n"


def _update(helpers: Any, path: Path, change: Callable[[str], str]) -> None:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    helpers.write_text(path, change(text.replace("\r\n", "\n")).replace("\n", newline))


def _external_code_page(helpers: Any, path: Path) -> str:
    lf = helpers.read_text(path).replace("\r\n", "\n")
    if has_line(lf, SECTION_HEADING):
        return "unchanged: already explains PyPI publishing"
    if OWN_SECTION_4.search(lf):
        return (
            "SKIPPED: the page has a section 4 of its own, left as it is; the PyPI section is in the template's page "
            f"(.tools/docs/external_code_upload.md), see {DOCS_URL}"
        )
    section = resource(PYPI_SECTION)
    _update(helpers, path, lambda text: add_section(text, section))
    return "updated: added the section 4. Publish the code on PyPI (optional)"


def _troubleshooting_page(helpers: Any, path: Path) -> str:
    lf = helpers.read_text(path).replace("\r\n", "\n")
    if has_line(lf, TROUBLESHOOTING_HEADING):
        return "unchanged: already has the entry Publishing to PyPI fails"
    entry = resource(PYPI_TROUBLESHOOTING)
    _update(helpers, path, lambda text: add_section(text, entry))
    return "updated: added the entry Publishing to PyPI fails"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """The PyPI section of external_code_upload.md and the PyPI entry of troubleshooting.md (idempotent, surgical)."""
    _ = context
    helpers = _load("v0_1_9_to_v0_1_10.py", "template_sync_helpers_0_1_26")
    changed = 0

    def report(name: Path, step: Callable[[Path], str]) -> None:
        nonlocal changed
        path = repo_root / name
        if not path.is_file():
            log(f"{name}: not present in this repository (nothing was created)")
            return
        try:
            status = step(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    report(EXTERNAL_CODE_PAGE, partial(_external_code_page, helpers))
    report(TROUBLESHOOTING_PAGE, partial(_troubleshooting_page, helpers))

    if changed:
        log(f"done: {changed} file(s) changed (documentation only). See {DOCS_URL}")
    else:
        log("done: no repository changes were required.")

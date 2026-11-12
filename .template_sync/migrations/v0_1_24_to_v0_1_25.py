"""Template 0.1.24 -> 0.1.25: publish the app's Python code (src/) on PyPI at each release, off until the developer turns it on.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_25.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. .github/workflows/publish_pypi.yml (new). At a published release it builds the package of src/ and uploads it to PyPI with trusted
   publishing (no secret). Both jobs are skipped unless the repository variable PYPI_PUBLISH is "true" (PyPI) or "testpypi" (TestPyPI),
   so nothing changes for an app that does not turn it on. It is NOT added to a repository that already uploads to PyPI with a
   workflow of its own (NucleiSky): the package would be uploaded twice. The log says what to do instead.
2. .tools/python/pypi_package.py (new): the checks and the build the workflow runs (version not on the index yet, a package in src/,
   the release version written into setup.py, name, version and dependencies of the built files).
3. setup.py, when it is still the template's (any name, version and Python series): the package needs what requirements.txt lists,
   README.md is its description on PyPI, and the source repository is linked. MANIFEST.in (new) puts requirements.txt in the source
   distribution. A setup.py changed by hand is left as it is (it still builds; the workflow uses it as it is).
4. .tools/python/bump_version.py: the release also writes its version into setup.py (it never did).

Rules this migration keeps: surgical (only the known blocks, the known setup.py, new files), idempotent (a second run reports no changes),
never raises for a customised file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import re
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_25.md"
TAG = "[migration 0.1.25]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_24_to_v0_1_25"

WORKFLOW = Path(".github/workflows/publish_pypi.yml")
PACKAGE_SCRIPT = Path(".tools/python/pypi_package.py")
SETUP = Path("setup.py")
MANIFEST_IN = Path("MANIFEST.in")
BUMP_VERSION = Path(".tools/python/bump_version.py")
NEW_FILES = (WORKFLOW, PACKAGE_SCRIPT, MANIFEST_IN)

# Text in a workflow that uploads to PyPI (as in .tools/python/pypi_package.py).
PUBLISH_MARKERS = ("pypa/gh-action-pypi-publish", "twine upload")

# --------------------------------------------------------------------------------------------------------------------------------
# setup.py of the template up to 0.1.24; the form (and merge_requirements.py) fill in the three placeholders.
# --------------------------------------------------------------------------------------------------------------------------------
OLD_SETUP = """from setuptools import setup, find_packages

setup(
    name="PYTHON_PROJ_NAME",
    version="VERSION_NUMBER",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=PYTHON_VERSION",
    install_requires=[
    ],
)
"""
SETUP_FIELDS = (("PYTHON_PROJ_NAME", "name"), ("VERSION_NUMBER", "version"), ("PYTHON_VERSION", "python"))
REQUIRED_LINE = "include requirements.txt"

# --------------------------------------------------------------------------------------------------------------------------------
# bump_version.py: (0.1.24 text, 0.1.25 text), each once.
# --------------------------------------------------------------------------------------------------------------------------------
BUMP_BLOCKS: tuple[tuple[str, str], ...] = (
    (
        'POST_INSTALL = ROOT / "app" / "bash_bat_scripts" / "post_install.bat"\n',
        'POST_INSTALL = ROOT / "app" / "bash_bat_scripts" / "post_install.bat"\n'
        "# version=\"1.2.3\" in setup.py: the version of the app's Python package (src/)\n"
        'SETUP = ROOT / "setup.py"\n'
        'SETUP_VERSION_RE = re.compile(r"^(?P<head>[ \\t]*version\\s*=\\s*)(?P<quote>[\\"\'])\\d+\\.\\d+\\.\\d+(?P=quote)", re.MULTILINE)\n',
    ),
    (
        '    print(f"Updated post_install.bat PKG_VERSION to {new_version}")\n',
        '''    print(f"Updated post_install.bat PKG_VERSION to {new_version}")


def bump_setup_py(new_version: str) -> None:
    """The version of the app's Python package in setup.py, when it is written there as X.Y.Z.

    The package in the installer and the one publish_pypi.yml publishes on PyPI then carry the release version. The line
    endings of the file are kept.
    """
    if not SETUP.exists():
        return
    with SETUP.open(encoding="utf-8", newline="") as handle:
        text = handle.read()
    updated = SETUP_VERSION_RE.sub(lambda m: f"{m.group('head')}{m.group('quote')}{new_version}{m.group('quote')}", text, count=1)
    if updated != text:
        with SETUP.open("w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        print(f"Updated setup.py version to {new_version}")
''',
    ),
    (
        "    # Also bump PKG_VERSION in post_install.bat\n    bump_post_install_bat(args.new_version)\n",
        "    # Also bump PKG_VERSION in post_install.bat\n    bump_post_install_bat(args.new_version)\n\n"
        "    # The version of the app's Python package (setup.py), also the version publish_pypi.yml publishes on PyPI\n"
        "    bump_setup_py(args.new_version)\n",
    ),
)


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


def apply_blocks(text_lf: str, blocks: tuple[tuple[str, str], ...]) -> str | None:
    """The text with every block changed, or None when a 0.1.24 block is not there exactly once (a customised file)."""
    for old, new in blocks:
        if text_lf.count(old) != 1:
            return None
        text_lf = text_lf.replace(old, new, 1)
    return text_lf


def setup_pattern(template: str) -> re.Pattern[str]:
    """A template setup.py as a pattern: each placeholder is the value the form (or merge_requirements.py) wrote, or the placeholder."""
    pattern = re.escape(template.rstrip())
    for placeholder, group in SETUP_FIELDS:
        pattern = pattern.replace(re.escape(placeholder), f'(?P<{group}>[^"\\n]*)', 1)
    return re.compile(pattern)


def setup_values(text_lf: str, template: str) -> dict[str, str] | None:
    """The name, version and Python series of a setup.py that is the given template text with its placeholders filled in, else None."""
    match = setup_pattern(template).fullmatch(text_lf.rstrip())
    return match.groupdict() if match else None


def render_setup(template: str, values: dict[str, str]) -> str:
    for placeholder, group in SETUP_FIELDS:
        template = template.replace(placeholder, values[group], 1)
    return template


def other_publishers(repo_root: Path) -> list[str]:
    """The workflows of the repository, other than publish_pypi.yml, that upload to PyPI."""
    folder = repo_root / WORKFLOW.parent
    found = []
    for path in sorted(folder.glob("*.y*ml")) if folder.is_dir() else []:
        if path.name != WORKFLOW.name and any(marker in path.read_text(encoding="utf-8", errors="replace") for marker in PUBLISH_MARKERS):
            found.append(path.relative_to(repo_root).as_posix())
    return found


def _create(helpers: Any, repo_root: Path, relative: Path, what: str) -> str:
    """Create a new file from the resource (LF); a different file of the repository at that place is kept."""
    path = repo_root / relative
    new = resource(relative)
    if path.is_file():
        if helpers.read_text(path).replace("\r\n", "\n") == new:
            return "unchanged: already the 0.1.25 template file"
        return f"SKIPPED: the repository has its own {relative.name}, left as it is; {what}"
    if not path.parent.is_dir():
        return f"SKIPPED: the repository has no {relative.parent.as_posix()} folder (nothing was created)"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(new)
    return "updated: created"


def _create_workflow(helpers: Any, repo_root: Path) -> str:
    others = other_publishers(repo_root)
    if others and not (repo_root / WORKFLOW).is_file():
        return (
            f"SKIPPED: {', '.join(others)} already uploads to PyPI, so the template's publish_pypi.yml was not added (the package would be "
            "uploaded twice). Nothing to do to keep publishing that way. To publish with the template's workflow instead (at each release, "
            f"with the release version): delete your workflow, add publish_pypi.yml by hand and register it on pypi.org, see {DOCS_URL}"
        )
    return _create(
        helpers, repo_root, WORKFLOW, "to publish the package at each release, compare it with the template's publish_pypi.yml, see " + DOCS_URL
    )


def _update_setup(helpers: Any, repo_root: Path) -> str:
    path = repo_root / SETUP
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lf = text.replace("\r\n", "\n")
    new_template = resource(SETUP)
    if setup_values(lf, new_template) is not None:
        return "unchanged: already reads its dependencies from requirements.txt and its description from README.md"
    values = setup_values(lf, OLD_SETUP)
    if values is None:
        return (
            "SKIPPED: not the template's setup.py (changed by hand, or the package is described in pyproject.toml), left as it is. It is "
            f"published as it is if you turn on PyPI publishing: check its dependencies and description, see {DOCS_URL}"
        )
    helpers.write_text(path, render_setup(new_template, values).replace("\n", newline))
    return f"updated: {values['name']} now needs what requirements.txt lists and has README.md as its description"


def _create_manifest_in(helpers: Any, repo_root: Path) -> str:
    path = repo_root / MANIFEST_IN
    setup = repo_root / SETUP
    reads_requirements = setup.is_file() and setup_values(helpers.read_text(setup).replace("\r\n", "\n"), resource(SETUP)) is not None
    if path.is_file():
        if REQUIRED_LINE in helpers.read_text(path).replace("\r\n", "\n").splitlines():
            return "unchanged: already puts requirements.txt in the source distribution"
        if reads_requirements:
            return f"SKIPPED: the repository has its own MANIFEST.in, left as it is; add the line `{REQUIRED_LINE}` to it, see {DOCS_URL}"
        return "unchanged: the repository's own MANIFEST.in"
    if not reads_requirements:
        return "unchanged: not needed (setup.py is not the template's, so it does not read requirements.txt)"
    return _create(helpers, repo_root, MANIFEST_IN, "")


def _update_bump_version(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lf = text.replace("\r\n", "\n")
    if all(new in lf for _, new in BUMP_BLOCKS):
        return "unchanged: already writes the release version into setup.py"
    new_lf = apply_blocks(lf, BUMP_BLOCKS)
    if new_lf is None:
        return (
            "SKIPPED: customised (a 0.1.24 block was not found), so it was left unchanged; the release version does not reach setup.py "
            f"(publish_pypi.yml writes it in its own copy anyway), see {DOCS_URL}"
        )
    helpers.write_text(path, new_lf.replace("\n", newline))
    return "updated: the release also writes its version into setup.py"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Opt-in PyPI publishing of the package in src/: a new workflow and script, setup.py and bump_version.py (idempotent, surgical)."""
    _ = context
    helpers = _load("v0_1_9_to_v0_1_10.py", "template_sync_helpers_0_1_25")
    changed = 0

    def report(name: Path, step: Callable[[], str]) -> None:
        nonlocal changed
        if name not in NEW_FILES and not (repo_root / name).is_file():
            log(f"{name}: not present in this repository (nothing was created)")
            return
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    report(PACKAGE_SCRIPT, partial(_create, helpers, repo_root, PACKAGE_SCRIPT, "publish_pypi.yml runs the template's, compare them"))
    report(WORKFLOW, partial(_create_workflow, helpers, repo_root))
    report(SETUP, partial(_update_setup, helpers, repo_root))
    report(MANIFEST_IN, partial(_create_manifest_in, helpers, repo_root))
    report(BUMP_VERSION, partial(_update_bump_version, helpers, repo_root / BUMP_VERSION))

    if changed:
        log(
            f"done: {changed} file(s) changed. Nothing is published until you turn it on (the repository variable PYPI_PUBLISH and a trusted "
            f"publisher on pypi.org), see {DOCS_URL}"
        )
    else:
        log("done: no repository changes were required.")

"""Template 0.1.23 -> 0.1.24: download links for "v" tags, installers only for the supported systems, the app's own Jupyter kernel.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_24.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. #41, a release tagged v1.0.0 wrote download links to /releases/download/1.0.0/, which do not exist. The release workflow now exports the
   tag as typed (RELEASE_TAG), and .tools/python/bump_version.py points the links to it; the installer names keep the bare version.
   bump_version.py reads the tag from the environment, so a workflow without the new line, or an older bump_version.py, still works.
2. #42, every release built the four installers, also for systems an app cannot run on (TensorFlow 2.17+ on Intel Macs). The new
   .tools/python/release_platforms.py reads `distribution.platforms` of labconstrictor-app.yaml: the release workflow builds only those
   systems (a new `platforms` job writes the build matrix), and the download page has no button for the others. Without the key,
   every system is built, as before. A build matrix that is not the template's (an app that removed a system by hand) is left as it is.
3. #43, a user-level "python3" kernelspec (`ipykernel install --user`) won over the app's own kernel, and the notebook ran in another
   Python. app/python_scripts/launch_jupyter.py sets JUPYTER_PREFER_ENV_PATH=1 before JupyterLab starts.

Rules this migration keeps: surgical (only the known blocks, one new file), idempotent (a second run reports no changes), never raises
for a customised file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import re
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_24.md"
TAG = "[migration 0.1.24]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_23_to_v0_1_24"

BUMP_VERSION = Path(".tools/python/bump_version.py")
PLATFORMS_SCRIPT = Path(".tools/python/release_platforms.py")
RELEASE_WORKFLOW = Path(".github/workflows/release_constructor.yml")
LAUNCHER = Path("app/python_scripts/launch_jupyter.py")

# --------------------------------------------------------------------------------------------------------------------------------
# bump_version.py (#41 the tag in the links, #42 no button for a system that is not built): (0.1.23 text, 0.1.24 text), each once.
# --------------------------------------------------------------------------------------------------------------------------------
BUMP_BLOCKS: tuple[tuple[str, str], ...] = (
    ("import argparse\nimport pathlib\n", "import argparse\nimport os\nimport pathlib\n"),
    (
        '''def replace_version_placeholder(text: str, new_version: str) -> str:
    """Render the release version in download documentation.

    Handles both pristine templates containing ``VERSION_NUMBER`` and
    initialized repositories where that placeholder was already rendered to an
    older explicit version such as ``0.0.0``.
    """
    explicit_versions = set(
        re.findall(r"/releases/download/(\\d+\\.\\d+\\.\\d+)/", text)
    )

    text = text.replace("VERSION_NUMBER", new_version)
    for previous_version in explicit_versions:
        text = text.replace(previous_version, new_version)
    return text
''',
        '''# The folder of a release in a download link: /releases/download/<tag>/ (the tag is 1.2.3 or v1.2.3).
DOWNLOAD_PATH_RE = re.compile(r"/releases/download/v?\\d+\\.\\d+\\.\\d+/")
# A button of the download page: <a href=".../releases/download/<tag>/<App>-<version>-<System>.<ext>"> <img ...> </a>
DOWNLOAD_BUTTON_RE = re.compile(r'<a href="[^"]*/releases/download/[^"]*">\\s*<img[^>]*>\\s*</a>')
NOT_BUILT_NOTE = "> No installer is built for {label}: this app does not support it."


def replace_version_placeholder(text: str, new_version: str, tag: str | None = None) -> str:
    """Render the release version in download documentation.

    Handles both pristine templates containing ``VERSION_NUMBER`` and
    initialized repositories where that placeholder was already rendered to an
    older explicit version such as ``0.0.0``. The links point to the release
    tag as it was typed (``v1.2.3`` or ``1.2.3``); the installer names carry
    the version without ``v`` (CellMigrationLab/LabConstrictor#41).
    """
    explicit_versions = set(
        re.findall(r"/releases/download/v?(\\d+\\.\\d+\\.\\d+)/", text)
    )

    text = text.replace("VERSION_NUMBER", new_version)
    for previous_version in explicit_versions:
        text = text.replace(previous_version, new_version)
    return DOWNLOAD_PATH_RE.sub(f"/releases/download/{tag or new_version}/", text)


def drop_unbuilt_buttons(text: str, not_built: list[tuple[tuple[str, ...], str]]) -> str:
    """Replace the download button of each system the release does not build with a note (CellMigrationLab/LabConstrictor#42).

    ``not_built`` holds (installer name endings, label) pairs, from release_platforms.py. The "or" between
    two buttons goes when one of them became a note.
    """
    if not not_built:
        return text

    def _button(match: re.Match) -> str:
        link = match.group(0).split('"', 2)[1]
        for endings, label in not_built:
            if link.endswith(endings):
                return NOT_BUILT_NOTE.format(label=label)
        return match.group(0)

    lines = DOWNLOAD_BUTTON_RE.sub(_button, text).split("\\n")
    note_start = NOT_BUILT_NOTE.split("{", 1)[0]
    kept = []
    for index, line in enumerate(lines):
        if line.strip() == "or":
            before = next((other for other in reversed(lines[:index]) if other.strip()), "")
            after = next((other for other in lines[index + 1 :] if other.strip()), "")
            if before.startswith(note_start) or after.startswith(note_start):
                continue
        kept.append(line)
    return "\\n".join(kept)


def not_built_systems() -> list[tuple[tuple[str, ...], str]]:
    """(installer name endings, label) of each system left out of distribution.platforms in labconstrictor-app.yaml."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import release_platforms

    return [(build.files, build.label) for build in release_platforms.skipped_builds(ROOT)]
''',
    ),
    (
        "def bump_version_in_download_executable_md(new_version: str) -> None:\n",
        "def bump_version_in_download_executable_md(\n"
        "    new_version: str, tag: str | None = None, not_built: list[tuple[tuple[str, ...], str]] | None = None\n"
        ") -> None:\n",
    ),
    (
        "            updated_text = replace_version_placeholder(text, new_version)\n",
        "            updated_text = drop_unbuilt_buttons(replace_version_placeholder(text, new_version, tag), not_built or [])\n",
    ),
    (
        "    args = parser.parse_args()\n",
        """    parser.add_argument(
        "--tag",
        default=os.environ.get("RELEASE_TAG") or None,
        help="The release tag the download links point to: 0.0.4 or v0.0.4 (default: $RELEASE_TAG, else the version)",
    )
    args = parser.parse_args()
""",
    ),
    (
        '        sys.exit("Version must look like X.Y.Z")\n',
        """        sys.exit("Version must look like X.Y.Z")
    if args.tag and args.tag not in (args.new_version, f"v{args.new_version}"):
        sys.exit(f"The release tag {args.tag} does not match the version {args.new_version} (expected {args.new_version} or v{args.new_version})")
""",
    ),
    (
        "    # Update download_executable.md\n    bump_version_in_download_executable_md(args.new_version)\n",
        "    # Update download_executable.md: the links point to the release tag, and a system that is not built has no button\n"
        "    bump_version_in_download_executable_md(args.new_version, args.tag, not_built_systems())\n",
    ),
)

# --------------------------------------------------------------------------------------------------------------------------------
# release_constructor.yml
# --------------------------------------------------------------------------------------------------------------------------------
# #41: both "Determine release version" steps strip the "v" of the tag; the tag as typed is exported first.
STRIP_LINE = re.compile(r'^(?P<indent>[ ]+)RAW="\$\{RAW#v\}"\n', re.MULTILINE)
TAG_LINES = (
    "{indent}# The tag as typed (v1.0.0 or 1.0.0): bump_version.py points the download links to it.\n"
    '{indent}echo "RELEASE_TAG=$RAW" >> "$GITHUB_ENV"\n'
)
# #42: the build matrix of the template, replaced by the one the platforms job writes; the job goes right before the build job.
MATRIX_OLD = """    needs: bump-version
    strategy:
      fail-fast: false
      matrix:
        include:
          - os_name: linux
            runner: ubuntu-latest
          - os_name: macos-arm64
            runner: macos-15
          - os_name: macos-intel
            runner: macos-15-intel
          - os_name: windows
            runner: windows-latest
"""
MATRIX_NEW = """    needs: [bump-version, platforms]
    strategy:
      fail-fast: false
      # One build per system of distribution.platforms in labconstrictor-app.yaml (every system without it), from the platforms job.
      matrix: ${{ fromJSON(needs.platforms.outputs.matrix) }}
"""
BUILD_JOB = "  build-installers:\n"
PLATFORMS_JOB = """  platforms:
    name: Installers to build
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.matrix.outputs.matrix }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4

      - name: Set up Python
        uses: actions/setup-python@7f4fc3e22c37d6ff65e88745f38bd3157c663f7c # v4
        with:
          python-version: '3.11'

      - name: Install requirements
        run: pip install -r .tools/requirements.txt

      - name: Read distribution.platforms of labconstrictor-app.yaml
        id: matrix
        run: python .tools/python/release_platforms.py --github-output "$GITHUB_OUTPUT"

"""

# --------------------------------------------------------------------------------------------------------------------------------
# launch_jupyter.py (#43): the variable is set right before JupyterLab is imported, in run_jupyter().
# --------------------------------------------------------------------------------------------------------------------------------
LAUNCHER_ANCHOR = """    # Import only after configure_tls() has completed. Tornado creates SSL
    # contexts during import on some versions.
    from jupyterlab.labapp import main
"""
LAUNCHER_LINES = """    # The app's own kernel must win over a user-level "python3" kernelspec (`ipykernel install --user`), which Jupyter
    # otherwise prefers: the notebook would run in another Python and fail with "No module named <app>". Set, not
    # setdefault: a JUPYTER_PREFER_ENV_PATH=0 of the user would bring the problem back.
    os.environ["JUPYTER_PREFER_ENV_PATH"] = "1"

"""
LAUNCHER_BLOCK = (LAUNCHER_ANCHOR, LAUNCHER_LINES + LAUNCHER_ANCHOR)


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


def apply_blocks(text_lf: str, blocks: tuple[tuple[str, str], ...]) -> str | None:
    """The text with every block changed, or None when a 0.1.23 block is not there exactly once (a customised file)."""
    for old, new in blocks:
        if text_lf.count(old) != 1:
            return None
        text_lf = text_lf.replace(old, new, 1)
    return text_lf


def add_tag_lines(text_lf: str) -> tuple[str, int]:
    """#41: export RELEASE_TAG before each `RAW="${RAW#v}"`; returns the text and how many were added."""
    return STRIP_LINE.subn(lambda m: TAG_LINES.format(indent=m.group("indent")) + m.group(0), text_lf)


def build_from_platforms(text_lf: str) -> str | None:
    """#42: the platforms job and its matrix, or None when the build job or its matrix is not the template's."""
    if text_lf.count(MATRIX_OLD) != 1 or text_lf.count(BUILD_JOB) != 1:
        return None
    return text_lf.replace(BUILD_JOB, PLATFORMS_JOB + BUILD_JOB, 1).replace(MATRIX_OLD, MATRIX_NEW, 1)


def _blocks_step(helpers: Any, path: Path, blocks: tuple[tuple[str, str], ...], what: str) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lf = text.replace("\r\n", "\n")
    if all(new in lf for _, new in blocks):
        return f"unchanged: already {what}"
    new_lf = apply_blocks(lf, blocks)
    if new_lf is None:
        return f"SKIPPED: customised (a 0.1.23 block was not found), so it was left unchanged; apply the 0.1.24 change by hand, see {DOCS_URL}"
    helpers.write_text(path, new_lf.replace("\n", newline))
    return f"updated: {what}"


def _create_platforms_script(helpers: Any, repo_root: Path) -> str:
    path = repo_root / PLATFORMS_SCRIPT
    new = (RESOURCE_ROOT / PLATFORMS_SCRIPT).read_text(encoding="utf-8")
    if path.is_file():
        if helpers.read_text(path).replace("\r\n", "\n") == new:
            return "already the 0.1.24 template file: unchanged"
        return "SKIPPED: the repository has its own release_platforms.py, left as it is"
    if not path.parent.is_dir():
        return "SKIPPED: the repository has no .tools/python folder (nothing was created)"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(new)
    return "updated: created"


def _update_workflow(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lf = text.replace("\r\n", "\n")
    done, notes = [], []
    if "RELEASE_TAG=" in lf:
        notes.append("already exports RELEASE_TAG")
    else:
        lf, count = add_tag_lines(lf)
        if count:
            done.append(f"exports RELEASE_TAG in {count} job(s)")
        else:
            notes.append(f'SKIPPED the release tag: no RAW="${{RAW#v}}" line (customised); the download links keep the bare version, see {DOCS_URL}')
    if MATRIX_NEW in lf:
        notes.append("already builds the systems of distribution.platforms")
    else:
        migrated = build_from_platforms(lf)
        if migrated is None:
            notes.append(
                "SKIPPED the build matrix: it is not the template's four builds (customised), so it was left as it is; to build the systems of "
                f"distribution.platforms instead, see {DOCS_URL}"
            )
        else:
            lf = migrated
            done.append("builds the systems of distribution.platforms (new platforms job)")
    if done:
        helpers.write_text(path, lf.replace("\n", newline))
        return "updated: " + "; ".join(done + notes)
    if any(note.startswith("SKIPPED") for note in notes):
        return "; ".join(notes)
    return "unchanged: " + "; ".join(notes)


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Release tag in the download links, installers for the systems of distribution.platforms, the app's own kernel (idempotent, surgical)."""
    _ = context
    helpers = _load("v0_1_9_to_v0_1_10.py", "template_sync_helpers_0_1_24")
    changed = 0

    def report(name: Path, step: Callable[[], str]) -> None:
        nonlocal changed
        if not (repo_root / name).is_file() and name != PLATFORMS_SCRIPT:
            log(f"{name}: not present in this repository (nothing was created)")
            return
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    report(PLATFORMS_SCRIPT, partial(_create_platforms_script, helpers, repo_root))
    report(BUMP_VERSION, partial(_blocks_step, helpers, repo_root / BUMP_VERSION, BUMP_BLOCKS, "points the download links to the release tag"))
    report(RELEASE_WORKFLOW, partial(_update_workflow, helpers, repo_root / RELEASE_WORKFLOW))
    report(LAUNCHER, partial(_blocks_step, helpers, repo_root / LAUNCHER, (LAUNCHER_BLOCK,), "prefers the app's own Jupyter kernel"))

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the fixes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

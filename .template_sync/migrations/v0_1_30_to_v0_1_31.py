"""Template 0.1.30 -> 0.1.31: no silent change of a notebook's package versions (#48), no lost work when a notebook is updated (#49).

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_31.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. #48, the requirements merge replaced one notebook's pin with another notebook's higher one (`numpy==1.26.4` and `numpy==2.0.0` gave
   `numpy==2.0.0`), and wrote every requirement as `==` (`numpy<2.0` became `numpy==2.0`), without a word.
   * .tools/python/merge_requirements.py is replaced by the 0.1.31 file when it is still the 0.1.30 template file: requirements are kept
     as written and combined, and requirements that disagree stop the merge, naming every notebook, unless requirements_overrides.yaml
     decides, with the reason. `--check` only looks for conflicts.
   * .tools/requirements.txt installs `packaging` (the merge compares versions as pip does).
   * requirements_overrides.yaml is created when the notebooks already pin a package to different versions: it records the version the
     merge chose until now (the newest), so the app installs what it installed before, but now as a visible decision to check.
   * .github/workflows/update_on_notebook_change.yml runs the merge again when requirements_overrides.yaml changes.
   * .github/workflows/release_constructor.yml runs `merge_requirements.py --check` before a release, after check_placeholders.py.
2. #49, the Update button of the Welcome notebook wrote the new notebook over the user's copy, with their changes and outputs, without a
   word. .tools/templates/Welcome_template.ipynb (the Welcome notebook of the next release is made from it): the button asks first, keeps
   the current copy as <notebook>.backup-<date>.ipynb, writes the new notebook atomically, and records the new version only after that.

Rules this migration keeps: surgical (whole files only when they are the 0.1.30 template files, otherwise known blocks), idempotent (a
second run reports no changes), never raises for a customised file (the sync pull request must still be created), the line endings of
each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import json
import re
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_31.md"
TAG = "[migration 0.1.31]"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_30_to_v0_1_31"
PREVIOUS_ROOT = RESOURCE_ROOT / "previous"

MERGE = Path(".tools/python/merge_requirements.py")
TOOLS_REQUIREMENTS = Path(".tools/requirements.txt")
OVERRIDES = Path("requirements_overrides.yaml")
NOTEBOOKS = Path("notebooks")
NOTEBOOK_WORKFLOW = Path(".github/workflows/update_on_notebook_change.yml")
RELEASE_WORKFLOW = Path(".github/workflows/release_constructor.yml")
WELCOME_TEMPLATE = Path(".tools/templates/Welcome_template.ipynb")

PACKAGING_PIN = "packaging==25.0"
PACKAGING_LINES = "# merge_requirements.py (versions compared as pip compares them)\n" + PACKAGING_PIN + "\n"
PACKAGING_LINE = re.compile(r"^\s*packaging\s*([=<>!~;\[]|$)", re.IGNORECASE)

# --------------------------------------------------------------------------------------------------------------------------------
# update_on_notebook_change.yml: a change of requirements_overrides.yaml runs the merge again. (0.1.30 text, 0.1.31 text), each once.
# --------------------------------------------------------------------------------------------------------------------------------
NOTEBOOK_WORKFLOW_BLOCKS: tuple[tuple[str, str], ...] = (
    ("      - 'notebooks/**/requirements.yaml'\n", "      - 'notebooks/**/requirements.yaml'\n      - 'requirements_overrides.yaml'\n"),
    ("            notebooks/**/requirements.yaml\n", "            notebooks/**/requirements.yaml\n            requirements_overrides.yaml\n"),
    (
        "        if: github.event_name == 'workflow_dispatch' || contains(steps.changed-files.outputs.all_changed_files, 'requirements.yaml')\n",
        "        if: github.event_name == 'workflow_dispatch' || contains(steps.changed-files.outputs.all_changed_files, 'requirements.yaml')"
        " || contains(steps.changed-files.outputs.all_changed_files, 'requirements_overrides.yaml')\n",
    ),
)

# --------------------------------------------------------------------------------------------------------------------------------
# release_constructor.yml: `merge_requirements.py --check` right after check_placeholders.py in the bump-version job (the first one).
# --------------------------------------------------------------------------------------------------------------------------------
PLACEHOLDER_STEP = re.compile(
    r"(?P<indent>[ ]+)bash \.tools/bash/run_with_context\.sh \\\n"
    r"(?P=indent)  (?P<log>\S+) \\\n"
    r"(?P=indent)  \"check_placeholders\.py\" \\\n"
    r"(?P=indent)  \.tools/python/check_placeholders\.py \\\n"
    r"(?P=indent)  python \.tools/python/check_placeholders\.py\n"
)
REQUIREMENTS_CHECK_STEP = (
    "{indent}bash .tools/bash/run_with_context.sh \\\n"
    "{indent}  {log} \\\n"
    '{indent}  "merge_requirements.py --check" \\\n'
    "{indent}  .tools/python/merge_requirements.py \\\n"
    "{indent}  python .tools/python/merge_requirements.py --source-dir notebooks --output requirements.txt --check\n"
)

# --------------------------------------------------------------------------------------------------------------------------------
# Welcome_template.ipynb (#49), in the code cell that defines the notebook table. Blocks are compared line by line without trailing
# spaces (the template's lines carry some); the new text replaces the lines of the old block.
# --------------------------------------------------------------------------------------------------------------------------------
HELPER_ANCHOR = "def apply_grid_layout(grid):\n"
HELPER = '''def replace_notebook_with_backup(notebook_path: Path, new_content: str):
    """Replace a notebook by its update, keeping the current file (with the user's changes and outputs) as a backup.

    The new content must be a notebook. It is written to a temporary file next to the notebook and then moved into place, so a failed
    download or write never leaves a broken notebook. Returns the path of the backup, or None when there was no notebook yet.
    """
    import os
    from datetime import datetime

    if not isinstance(json.loads(new_content), dict):
        raise ValueError("the downloaded file is not a notebook")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if notebook_path.exists():
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = notebook_path.with_name(f"{notebook_path.stem}.backup-{stamp}{notebook_path.suffix}")
        number = 1
        while backup_path.exists():
            number += 1
            backup_path = notebook_path.with_name(f"{notebook_path.stem}.backup-{stamp}-{number}{notebook_path.suffix}")
        shutil.copy2(notebook_path, backup_path)
    temporary_path = notebook_path.with_name(f".{notebook_path.name}.update")
    try:
        temporary_path.write_text(new_content, encoding="utf-8")
        os.replace(temporary_path, notebook_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path


'''
HANDLER_OLD = '''                        def button_update(main_folder, subfolder, row_idx):
                            def show(button):
                                button.disabled = True
                                button.description = "Updating..."

                                # Download the notebook
                                notebook_url = f"https://api.github.com/repos/{github_owner}/{github_repo_name}/contents/notebooks/{main_folder}/{subfolder}.ipynb?ref={github_branch}"
                                if private_repository:
                                    if github_token.value:
                                        notebook_response = requests.get(notebook_url, headers={"Authorization": f"token {github_token.value}"})
                                    else:
                                        grip_output.value = "❌ Please check your Presonal Access Token"
                                else:
                                    notebook_response = requests.get(notebook_url)

                                if notebook_response.status_code == 200:
                                    try:
                                        # Decode the notebook content
                                        notebook_content = base64.b64decode(
                                            notebook_response.json()['content']
                                        ).decode('utf-8')

                                        # Save to local path
                                        local_notebook_path = Path(".") / main_folder / f"{subfolder}.ipynb"
                                        local_notebook_path.parent.mkdir(parents=True, exist_ok=True)
                                        local_notebook_path.write_text(notebook_content, encoding='utf-8')

                                        # Update the local version YAML
                                        local_latest_versions[main_folder][subfolder] = online_latest_versions[main_folder][subfolder]
                                        local_latest_version_path.write_text(yaml.dump(local_latest_versions, default_flow_style=False))

                                        grip_output.value = f"✅ Updated {subfolder} to version {online_latest_versions[main_folder][subfolder]}"

                                        # Update the grid to show up-to-date
                                        grid[row_idx, 3] = table_cell(online_latest_versions[main_folder][subfolder])
                                        grid[row_idx, 4] = table_cell("✅ Up-to-date")
                                        grid[row_idx, 5] = table_cell("-")
                                        apply_grid_layout(grid)

                                    except Exception as e:
                                        grip_output.value = f"❌ Error saving notebook: {e}"
                                        button.disabled = False
                                        button.description = "Update"
                                else:
                                    grip_output.value = f"❌ Failed to download notebook (HTTP {notebook_response.status_code})"
                                    display(notebook_response.text[:300])
                                    button.disabled = False
                                    button.description = "Update"

                            return show
'''
HANDLER_NEW = '''                        def button_update(main_folder, subfolder, row_idx):
                            # Updating replaces the user's copy of the notebook: ask first, keep that copy as a backup, and change the
                            # recorded version only once the new notebook is in place (CellMigrationLab/LabConstrictor#49).
                            local_notebook_path = Path(".") / main_folder / f"{subfolder}.ipynb"
                            new_version = online_latest_versions[main_folder][subfolder]

                            def reset(button):
                                button.disabled = False
                                button.description = "Update"

                            def do_update(button):
                                button.description = "Updating..."
                                notebook_url = f"https://api.github.com/repos/{github_owner}/{github_repo_name}/contents/notebooks/{main_folder}/{subfolder}.ipynb?ref={github_branch}"
                                headers = {}
                                if private_repository:
                                    if not github_token.value:
                                        grip_output.value = "❌ Please check your Personal Access Token. Your notebook was not changed."
                                        reset(button)
                                        return
                                    headers = {"Authorization": f"token {github_token.value}"}
                                try:
                                    notebook_response = requests.get(notebook_url, headers=headers, timeout=60)
                                except requests.RequestException as e:
                                    grip_output.value = f"❌ Failed to download notebook: {e}. Your notebook was not changed."
                                    reset(button)
                                    return
                                if notebook_response.status_code != 200:
                                    grip_output.value = f"❌ Failed to download notebook (HTTP {notebook_response.status_code}). Your notebook was not changed."
                                    display(notebook_response.text[:300])
                                    reset(button)
                                    return
                                try:
                                    notebook_content = base64.b64decode(notebook_response.json()['content']).decode('utf-8')
                                    backup_path = replace_notebook_with_backup(local_notebook_path, notebook_content)
                                except Exception as e:
                                    grip_output.value = f"❌ Error saving notebook: {e}. Your notebook was not changed."
                                    reset(button)
                                    return

                                # The new notebook is in place: only now record its version
                                local_latest_versions[main_folder][subfolder] = new_version
                                local_latest_version_path.write_text(yaml.dump(local_latest_versions, default_flow_style=False))
                                saved = f" Your previous copy is saved as {backup_path.name}, next to it." if backup_path else ""
                                grip_output.value = f"✅ Updated {subfolder} to version {new_version}.{saved}"

                                # Update the grid to show up-to-date
                                grid[row_idx, 3] = table_cell(new_version)
                                grid[row_idx, 4] = table_cell("✅ Up-to-date")
                                grid[row_idx, 5] = table_cell("-")
                                apply_grid_layout(grid)

                            def show(button):
                                button.disabled = True
                                button.description = "Confirm below"
                                question = widgets.HTML(
                                    f"<b>Update {subfolder} to version {new_version}?</b><br>"
                                    f"The new version replaces <code>{local_notebook_path.as_posix()}</code>. Your current copy, with your changes "
                                    f"and outputs, is first saved as <code>{subfolder}.backup-&lt;date&gt;.ipynb</code> in the same folder.<br>"
                                    "If the notebook is open in a tab, close that tab first: otherwise JupyterLab can save the old copy over the update."
                                )
                                confirm = widgets.Button(description="Back up and update", button_style="warning", layout=widgets.Layout(width="auto"))
                                cancel = widgets.Button(description="Cancel")
                                dialog = widgets.VBox([question, widgets.HBox([confirm, cancel])])

                                def on_confirm(_):
                                    dialog.close()
                                    do_update(button)

                                def on_cancel(_):
                                    dialog.close()
                                    grip_output.value = f"Update of {subfolder} cancelled: nothing was changed."
                                    reset(button)

                                confirm.on_click(_widget_callback_output_cell_1.capture(clear_output=True, wait=True)(on_confirm))
                                cancel.on_click(_widget_callback_output_cell_1.capture(clear_output=True, wait=True)(on_cancel))
                                display(dialog)

                            return show
'''
WELCOME_DONE = "def replace_notebook_with_backup("


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


def _lf(text: str) -> tuple[str, str]:
    return text.replace("\r\n", "\n"), ("\r\n" if "\r\n" in text else "\n")


def apply_blocks(text_lf: str, blocks: tuple[tuple[str, str], ...]) -> str | None:
    """The text with every block changed, or None when a 0.1.30 block is not there exactly once (a customised file)."""
    for old, new in blocks:
        if text_lf.count(old) != 1:
            return None
        text_lf = text_lf.replace(old, new, 1)
    return text_lf


# --------------------------------------------------------------------------------------------------------------------------------
# #48
# --------------------------------------------------------------------------------------------------------------------------------
def _replace_merge(helpers: Any, repo_root: Path) -> str:
    path = repo_root / MERGE
    current, newline = _lf(helpers.read_text(path))
    new = (RESOURCE_ROOT / MERGE).read_text(encoding="utf-8")
    if current == new:
        return "already the 0.1.31 template file: unchanged"
    if current != (PREVIOUS_ROOT / MERGE).read_text(encoding="utf-8"):
        return f"SKIPPED: customised (not the 0.1.30 template file), so it was left unchanged; apply the 0.1.31 change by hand, see {DOCS_URL}"
    helpers.write_text(path, new.replace("\n", newline))
    return "updated: replaced by the 0.1.31 template file (requirements that disagree stop the merge)"


def _add_packaging(helpers: Any, repo_root: Path) -> str:
    path = repo_root / TOOLS_REQUIREMENTS
    current = helpers.read_text(path)
    if any(PACKAGING_LINE.match(line) for line in current.splitlines()):
        return "already installs packaging: unchanged"
    newline = "\r\n" if "\r\n" in current else "\n"
    text = current if not current or current.endswith(("\n", "\r")) else current + newline
    helpers.write_text(path, text + PACKAGING_LINES.replace("\n", newline))
    return f"updated: added {PACKAGING_PIN}"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_key(version: str) -> tuple[tuple[int, Any], ...]:
    """The order the merge used until 0.1.30 for plain versions (numbers compared as numbers)."""
    parts = re.split(r"[.\-+_]|(?=[a-zA-Z])", version)
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts if p)


def notebook_dependencies(text: str) -> list[str]:
    """The `dependencies` list of a requirements.yaml (PyYAML when it is installed, else the simple list forms the template uses)."""
    try:
        import yaml
    except ImportError:
        pass
    else:
        data = yaml.safe_load(text) or {}
        found = data.get("dependencies") if isinstance(data, dict) else None
        return [str(item) for item in found or []]
    items: list[str] = []
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        key = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if key:
            inside = key.group(1) == "dependencies"
            value = key.group(2).strip()
            if inside and value.startswith("[") and value.endswith("]"):
                items += [part.strip().strip("'\"") for part in value[1:-1].split(",") if part.strip()]
                inside = False
            continue
        item = re.match(r"^\s*-\s+(.*)$", line)
        if inside and item:
            items.append(item.group(1).strip().strip("'\""))
    return items


EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(\[[^\]]*\])?\s*==\s*(?P<version>[A-Za-z0-9_.+!-]+)$")


def pin_conflicts(repo_root: Path, helpers: Any) -> dict[str, list[tuple[str, str]]]:
    """Packages that the notebooks' `dependencies` pin (==, without a marker) to different versions: {name: [(file, version)]}."""
    pins: dict[str, list[tuple[str, str]]] = {}
    root = repo_root / NOTEBOOKS
    files = sorted({*root.rglob("requirements.yaml"), *root.rglob("requirements.yml")}) if root.is_dir() else []
    for path in files:
        for dependency in notebook_dependencies(helpers.read_text(path)):
            match = EXACT_PIN.match(dependency.split("#", 1)[0].strip())
            if match:
                pins.setdefault(_canonical(match.group("name")), []).append((path.relative_to(repo_root).as_posix(), match.group("version")))
    return {name: found for name, found in sorted(pins.items()) if len({version for _, version in found}) > 1}


def overrides_text(conflicts: dict[str, list[tuple[str, str]]]) -> str:
    def quoted(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        "# Decisions for packages that the notebooks pin to different versions (template 0.1.31, CellMigrationLab/LabConstrictor#48).",
        "# The requirements merge (.tools/python/merge_requirements.py) stops when notebooks disagree and no entry here decides. Each entry",
        "# gives the requirement every notebook gets (use) and why (reason); both are written at the top of the generated requirements files.",
        f"# See {DOCS_URL}",
        "overrides:",
    ]
    for name, found in conflicts.items():
        newest = max((version for _, version in found), key=_version_key)
        asked = ", ".join(f"{file} =={version}" for file, version in found)
        reason = (
            "Recorded by the template 0.1.31 migration: the version the requirements merge chose until then (the newest pin), so the app "
            f"installs what it installed before. The notebooks ask for: {asked}. Check every notebook with this version, then write here why it is right."
        )
        lines += [f"  {name}:", f"    use: {quoted('==' + newest)}", f"    reason: {quoted(reason)}"]
    return "\n".join(lines) + "\n"


def _seed_overrides(helpers: Any, repo_root: Path) -> str:
    conflicts = pin_conflicts(repo_root, helpers)
    path = repo_root / OVERRIDES
    if path.is_file():
        return "already exists: unchanged (the merge says which conflicts it does not decide yet)"
    if not conflicts:
        return "not needed: no package is pinned to different versions by the notebooks (nothing was created)"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(overrides_text(conflicts))
    return f"updated: created; it records the version used until now for {', '.join(conflicts)}. Check each one (see {DOCS_URL})"


def add_requirements_check(text_lf: str) -> str | None:
    """The release workflow with `merge_requirements.py --check` after the first check_placeholders.py step; None without that step."""
    match = PLACEHOLDER_STEP.search(text_lf)
    if match is None:
        return None
    step = REQUIREMENTS_CHECK_STEP.format(indent=match.group("indent"), log=match.group("log"))
    return text_lf[: match.end()] + step + text_lf[match.end() :]


def _update_release_workflow(helpers: Any, repo_root: Path, path: Path) -> str:
    current, newline = _lf(helpers.read_text(path))
    if "merge_requirements.py --source-dir notebooks --output requirements.txt --check" in current:
        return "unchanged: already checks the requirements"
    merge = repo_root / MERGE
    if not merge.is_file() or '"--check"' not in helpers.read_text(merge):
        return f"SKIPPED: the repository's merge_requirements.py has no --check (it was not replaced), so the release does not run it. See {DOCS_URL}"
    new = add_requirements_check(current)
    if new is None:
        return f"SKIPPED: the check_placeholders.py command was not found (customised workflow); add the requirements check by hand. See {DOCS_URL}"
    helpers.write_text(path, new.replace("\n", newline))
    return "updated: the release stops when the notebooks' requirements conflict (merge_requirements.py --check)"


def _blocks_step(helpers: Any, path: Path, blocks: tuple[tuple[str, str], ...], what: str, skipped_hint: str) -> str:
    current, newline = _lf(helpers.read_text(path))
    if all(new in current for _, new in blocks):
        return f"unchanged: already {what}"
    new = apply_blocks(current, blocks)
    if new is None:
        return f"SKIPPED: customised (a 0.1.30 block was not found), so it was left unchanged; {skipped_hint} See {DOCS_URL}"
    helpers.write_text(path, new.replace("\n", newline))
    return f"updated: {what}"


# --------------------------------------------------------------------------------------------------------------------------------
# #49
# --------------------------------------------------------------------------------------------------------------------------------
def _replace_lines(source: str, old: str, new: str) -> str | None:
    """`source` with the lines of `old` (compared without trailing spaces) replaced by `new`; None when they are not there exactly once."""
    lines = source.splitlines(keepends=True)
    wanted = [line.rstrip() for line in old.splitlines()]
    stripped = [line.rstrip() for line in lines]
    starts = [i for i in range(len(lines) - len(wanted) + 1) if stripped[i : i + len(wanted)] == wanted]
    if len(starts) != 1:
        return None
    start = starts[0]
    return "".join(lines[:start]) + new + "".join(lines[start + len(wanted) :])


def update_welcome_source(source: str) -> str | None:
    """The table cell with the safe update (#49), or None when the 0.1.30 handler or the helper's anchor is not found."""
    if source.count(HELPER_ANCHOR) != 1:
        return None
    replaced = _replace_lines(source, HANDLER_OLD, HANDLER_NEW)
    if replaced is None:
        return None
    return replaced.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)


def _update_welcome(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    notebook = json.loads(text)
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    sources = ["".join(cell["source"]) if isinstance(cell.get("source"), list) else cell.get("source", "") for cell in cells]
    if any(WELCOME_DONE in source for source in sources):
        return "unchanged: the Update button already keeps a backup"
    for cell, source in zip(cells, sources):
        new = update_welcome_source(source.replace("\r\n", "\n"))
        if new is None:
            continue
        cell["source"] = new.splitlines(keepends=True) if isinstance(cell.get("source"), list) else new
        rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
        if text.endswith(("\n", "\r\n")):
            rendered += "\n"
        helpers.write_text(path, rendered.replace("\n", newline))
        return "updated: the Update button asks first and keeps the current notebook as a backup"
    return f"SKIPPED: the update code of the notebook table was not found (customised); apply the 0.1.31 change by hand. See {DOCS_URL}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Requirements that disagree stop the merge (#48); the notebook update asks first and keeps a backup (#49). Idempotent, surgical."""
    _ = context
    helpers = _load("v0_1_9_to_v0_1_10.py", "template_sync_helpers_0_1_31")
    changed = 0

    def report(name: Path, step: Callable[[], str], must_exist: bool = True) -> None:
        nonlocal changed
        if must_exist and not (repo_root / name).is_file():
            log(f"{name}: not present in this repository (nothing was created)")
            return
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    report(MERGE, partial(_replace_merge, helpers, repo_root))
    report(TOOLS_REQUIREMENTS, partial(_add_packaging, helpers, repo_root))
    report(OVERRIDES, partial(_seed_overrides, helpers, repo_root), must_exist=False)
    report(NOTEBOOK_WORKFLOW, partial(_blocks_step, helpers, repo_root / NOTEBOOK_WORKFLOW, NOTEBOOK_WORKFLOW_BLOCKS,
                                      "runs the merge when requirements_overrides.yaml changes",
                                      "after a change of requirements_overrides.yaml, run Updates on notebook change by hand."))
    report(RELEASE_WORKFLOW, partial(_update_release_workflow, helpers, repo_root, repo_root / RELEASE_WORKFLOW))
    report(WELCOME_TEMPLATE, partial(_update_welcome, helpers, repo_root / WELCOME_TEMPLATE))

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; the fixes reach users with the next release of the app.")
    else:
        log("done: no repository changes were required.")

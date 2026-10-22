"""Template 0.1.18 -> 0.1.19: src_change.yaml in every installer, Python series (not patch release) in setup.py.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_19.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. .tools/python/bump_constructor.py. `ensure_extra_files` removed the `.tools/meta/src_change.yaml` mapping from `extra_files` and then
   re-added it only if it had not been there before, so every other release built an installer without `src_change.yaml` and the Welcome
   notebook of that release could not offer source-code updates. The known block is replaced by one that always re-adds the mapping.
2. .tools/python/merge_requirements.py. It wrote the exact Python release pinned for the installer into setup.py (`python_requires=">=3.12.13"`),
   so `pip install -e .` failed on any earlier patch release of the same series. The known block is replaced by one that writes the series
   (`>=3.12`); environment.yaml keeps the exact pin.
3. setup.py. A `python_requires=">=X.Y.Z"` written by the old merge_requirements.py becomes `">=X.Y"` (only that exact shape).

Rules this migration keeps: surgical (only the known blocks and the one known setup.py shape), idempotent (a second run reports no changes),
never raises for a customised file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_19.md"
TAG = "[migration 0.1.19]"
BUMP_CONSTRUCTOR = Path(".tools/python/bump_constructor.py")
MERGE_REQUIREMENTS = Path(".tools/python/merge_requirements.py")
SETUP = Path("setup.py")

# Blocks of the 0.1.18 template scripts and their 0.1.19 replacements (LF; each file's own line endings are used when matching and writing).
BUMP_OLD = """            src_change_dst = f"{project_folder}/src_change.yaml"
            if src_change_file not in existing_sources and src_change_dst not in existing_dests:
                normalized_items.append({src_change_file: src_change_dst})
                src_added += 1
                debug(f"Included src change marker: {src_change_file} -> {src_change_dst}")
            else:
                debug("src change marker mapping already present")
"""
BUMP_NEW = """            # The mapping was removed from normalized_items above, so always re-add it here;
            # checking existing_sources would drop it on every other release.
            src_change_dst = f"{project_folder}/src_change.yaml"
            normalized_items.append({src_change_file: src_change_dst})
            src_added += 1
            debug(f"Included src change marker: {src_change_file} -> {src_change_dst}")
"""
MERGE_OLD = """        setup_text = setup_path.read_text(encoding="utf-8")
        pattern = r'python_requires\\s*=\\s*["\\'][><=\\.0-9 ]+(PYTHON\\_VERSION)?["\\']'
        new_setup_text = re.sub(pattern, f'python_requires=">={python_version}"', setup_text)
        setup_path.write_text(new_setup_text, encoding="utf-8")
        print(f"Updated {setup_path} with python_requires>={python_version}")
"""
MERGE_NEW = """        setup_text = setup_path.read_text(encoding="utf-8")
        # Require only the major.minor series: environment.yaml pins the exact
        # patch release for the installer, but a source install should accept
        # any patch release of that series (e.g. 3.12.3 for 3.12.13).
        python_series = ".".join(str(python_version).split(".")[:2])
        pattern = r'python_requires\\s*=\\s*["\\'][><=\\.0-9 ]+(PYTHON\\_VERSION)?["\\']'
        new_setup_text = re.sub(pattern, f'python_requires=">={python_series}"', setup_text)
        setup_path.write_text(new_setup_text, encoding="utf-8")
        print(f"Updated {setup_path} with python_requires>={python_series}")
"""
SETUP_PATCH_PIN = re.compile(r'(python_requires\s*=\s*)(["\'])>=(\d+)\.(\d+)\.\d+\2')
SETUP_SERIES_PIN = re.compile(r'python_requires\s*=\s*(["\'])>=(\d+\.\d+|PYTHON_VERSION)\1')


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers() -> Any:
    """Reuse the helpers of the 0.1.10 migration (newline-preserving file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_19", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_block(helpers: Any, path: Path, old: str, new: str) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    old_here, new_here = old.replace("\n", newline), new.replace("\n", newline)
    if new_here in text:
        return "unchanged: already has the 0.1.19 block"
    count = text.count(old_here)
    if count == 0:
        return f"SKIPPED: the 0.1.18 block was not found (customised file); apply the change by hand. See {DOCS_URL}"
    if count > 1:
        return f"SKIPPED: the 0.1.18 block was found {count} times; apply the change by hand. See {DOCS_URL}"
    helpers.write_text(path, text.replace(old_here, new_here, 1))
    return "updated"


def _migrate_setup(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    matches = SETUP_PATCH_PIN.findall(text)
    if not matches:
        if SETUP_SERIES_PIN.search(text):
            return "unchanged: python_requires already names a Python series"
        return "unchanged: no python_requires=\">=X.Y.Z\" written by merge_requirements.py"
    if len(matches) > 1:
        return f"SKIPPED: {len(matches)} python_requires lines; set \">=X.Y\" by hand. See {DOCS_URL}"
    new_text = SETUP_PATCH_PIN.sub(lambda m: f"{m.group(1)}{m.group(2)}>={m.group(3)}.{m.group(4)}{m.group(2)}", text, count=1)
    helpers.write_text(path, new_text)
    series = SETUP_SERIES_PIN.search(new_text)
    return f"updated: python_requires >={series.group(2) if series else '?'}"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Fix bump_constructor.py, merge_requirements.py and setup.py (idempotent, surgical, never raises for a customised file)."""
    _ = context
    helpers = _load_helpers()
    changed = 0

    def report(name: Path, step: Any) -> None:
        nonlocal changed
        if not (repo_root / name).is_file():
            log(f"{name}: SKIPPED, file not found in this repository (nothing was created).")
            return
        try:
            status = step()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        if status.startswith("updated"):
            changed += 1
        log(f"{name}: {status}")

    report(BUMP_CONSTRUCTOR, lambda: _replace_block(helpers, repo_root / BUMP_CONSTRUCTOR, BUMP_OLD, BUMP_NEW))
    report(MERGE_REQUIREMENTS, lambda: _replace_block(helpers, repo_root / MERGE_REQUIREMENTS, MERGE_OLD, MERGE_NEW))
    report(SETUP, lambda: _migrate_setup(helpers, repo_root / SETUP))

    if changed:
        log(f"done: {changed} file(s) changed. Review the pull request before merging; src_change.yaml returns with the next installer build.")
    else:
        log("done: no repository changes were required.")

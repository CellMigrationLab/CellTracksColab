"""Template 0.1.20 -> 0.1.21: build the installer with conda-standalone 26.9.3 or later.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_21.md

What this changes in an app repository (idempotent, never deletes anything):

1. .tools/bash/create_constructor_environment.sh. The constructor environment did not pin conda-standalone, the `_conda.exe` that the
   installer uses to install the packages. Versions before 26.9.3 write the package records without `paths_data` when constructor disables
   conda's safety checks (conda/conda#16856). conda 26.9 and later in the installed app looks up the Windows launcher stub in that metadata,
   cannot find it, and treats the error as a permission problem: the Windows uninstaller then asks twice for administrator rights for
   python.exe (`conda init --reverse`). The known `conda create` line is replaced by one that adds `"conda-standalone>=26.9.3"`.

Rules this migration keeps: surgical (only the one known line), idempotent (a second run reports no changes), never raises for a customised
file (the sync pull request must still be created), the line endings of the file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_21.md"
TAG = "[migration 0.1.21]"
CREATE_ENV = Path(".tools/bash/create_constructor_environment.sh")

# The 0.1.20 line and its 0.1.21 replacement (LF; the file's own line endings are used when matching and writing).
CREATE_OLD = 'conda create -y -n "${CONSTRUCTOR_ENV:?}" -c conda-forge -c defaults python=3.11 constructor\n'
CREATE_NEW = """# conda-standalone (the installer's _conda.exe) older than 26.9.3 writes package
# records without paths_data when constructor disables conda's safety checks.
# conda 26.9 and later in the installed app then cannot find the Windows
# launcher stub, and `conda init --reverse` in the uninstaller asks for admin
# rights (conda/conda#16856, fixed in conda 26.9.3).
conda create -y -n "${CONSTRUCTOR_ENV:?}" -c conda-forge -c defaults python=3.11 constructor "conda-standalone>=26.9.3"
"""


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers() -> Any:
    """Reuse the helpers of the 0.1.10 migration (newline-preserving file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_21", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin_conda_standalone(helpers: Any, path: Path) -> str:
    text = helpers.read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    old_here, new_here = CREATE_OLD.replace("\n", newline), CREATE_NEW.replace("\n", newline)
    if new_here in text:
        return "unchanged: already pins conda-standalone>=26.9.3"
    count = text.count(old_here)
    if count == 0:
        if "conda-standalone" in text:
            return f"SKIPPED: the file already names conda-standalone (customised); check it allows 26.9.3 or later. See {DOCS_URL}"
        return f"SKIPPED: the 0.1.20 conda create line was not found (customised file); add \"conda-standalone>=26.9.3\" by hand. See {DOCS_URL}"
    if count > 1:
        return f"SKIPPED: the 0.1.20 conda create line was found {count} times; add the pin by hand. See {DOCS_URL}"
    helpers.write_text(path, text.replace(old_here, new_here, 1))
    return "updated: conda-standalone>=26.9.3"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Pin conda-standalone in create_constructor_environment.sh (idempotent, surgical, never raises for a customised file)."""
    _ = context
    path = repo_root / CREATE_ENV
    if not path.is_file():
        log(f"{CREATE_ENV}: SKIPPED, file not found in this repository (nothing was created).")
        log("done: no repository changes were required.")
        return
    try:
        status = _pin_conda_standalone(_load_helpers(), path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
    log(f"{CREATE_ENV}: {status}")
    if status.startswith("updated"):
        log("done: 1 file changed. Build and release a new installer: installs made with an older installer keep the admin prompt at uninstall.")
    else:
        log("done: no repository changes were required.")

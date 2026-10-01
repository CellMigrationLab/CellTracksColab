from __future__ import annotations

from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(".tools/bash/test_conda_environment.sh")
WORKFLOW_PATH = Path(".github/workflows/update_on_notebook_change.yml")

SKIP_GUARD = """is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if is_truthy "${SKIP_CONDA_ENV_TEST:-}"; then
    echo "SKIP_CONDA_ENV_TEST is enabled; skipping conda environment validation."
    exit 0
fi

"""


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def update_test_script(repo_root: Path) -> bool:
    script_path = repo_root / SCRIPT_PATH
    if not script_path.exists():
        print(f"Skipping missing file: {SCRIPT_PATH}")
        return False

    original_text = script_path.read_text(encoding="utf-8")
    if "SKIP_CONDA_ENV_TEST is enabled; skipping conda environment validation." in original_text:
        return False

    newline = detect_newline(original_text)
    marker = f"set -u  # Exit on undefined variables{newline}"
    if marker not in original_text:
        raise ValueError(f"Unable to find insertion point in {SCRIPT_PATH}")

    guard_block = SKIP_GUARD.replace("\n", newline)
    updated_text = original_text.replace(marker, marker + newline + guard_block, 1)
    script_path.write_text(updated_text, encoding="utf-8")
    print(f"Updated {SCRIPT_PATH}")
    return True


def update_workflow(repo_root: Path) -> bool:
    workflow_path = repo_root / WORKFLOW_PATH
    if not workflow_path.exists():
        print(f"Skipping missing file: {WORKFLOW_PATH}")
        return False

    original_text = workflow_path.read_text(encoding="utf-8")
    if "SKIP_CONDA_ENV_TEST:" in original_text:
        return False

    lines = original_text.splitlines(keepends=True)
    newline = detect_newline(original_text)

    for index, line in enumerate(lines):
        if line.strip() != "id: test-conda-env":
            continue

        indent = line[: len(line) - len(line.lstrip())]
        run_index = None
        for candidate in range(index + 1, len(lines)):
            if lines[candidate].strip() == "run: |":
                run_index = candidate
                break

        if run_index is None:
            raise ValueError(f"Unable to find run block for test-conda-env in {WORKFLOW_PATH}")

        env_block = [
            f"{indent}env:{newline}",
            f"{indent}  SKIP_CONDA_ENV_TEST: ${{{{ vars.SKIP_CONDA_ENV_TEST }}}}{newline}",
        ]
        updated_lines = lines[:run_index] + env_block + lines[run_index:]
        workflow_path.write_text("".join(updated_lines), encoding="utf-8")
        print(f"Updated {WORKFLOW_PATH}")
        return True

    raise ValueError(f"Unable to find test-conda-env step in {WORKFLOW_PATH}")


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_test_script(repo_root) or changed_any
    changed_any = update_workflow(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

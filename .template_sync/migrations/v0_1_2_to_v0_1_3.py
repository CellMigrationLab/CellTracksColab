from __future__ import annotations

import json

from pathlib import Path
from typing import Any


WELCOME_TEMPLATE_PATH = Path(".tools/templates/Welcome_template.ipynb")
WELCOME_NOTEBOOK_PATH = Path("app/menuinst/Welcome.ipynb")

OLD_NOTEBOOK_URL = (
    'f"https://api.github.com/repos/{github_owner}/{github_repo_name}'
    '/contents/notebooks/{main_folder}/{subfolder}/{subfolder}.ipynb'
    '?ref={github_branch}"'
)
NEW_NOTEBOOK_URL = (
    'f"https://api.github.com/repos/{github_owner}/{github_repo_name}'
    '/contents/notebooks/{main_folder}/{subfolder}.ipynb'
    '?ref={github_branch}"'
)


def update_notebook_url(repo_root: Path, relative_path: Path) -> bool:
    path = repo_root / relative_path
    if not path.exists():
        print(f"Skipping missing file: {relative_path}")
        return False

    original_text = path.read_text(encoding="utf-8")
    try:
        notebook = json.loads(original_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse {relative_path} as JSON") from exc

    changed = False
    found_current = False
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        source_is_list = isinstance(source, list)
        lines = source if source_is_list else [str(source)]
        updated_lines = []
        cell_changed = False

        for line in lines:
            if OLD_NOTEBOOK_URL in line:
                line = line.replace(OLD_NOTEBOOK_URL, NEW_NOTEBOOK_URL, 1)
                changed = True
                cell_changed = True
            if NEW_NOTEBOOK_URL in line:
                found_current = True
            updated_lines.append(line)

        if cell_changed:
            cell["source"] = updated_lines if source_is_list else updated_lines[0]

    if not changed:
        if found_current:
            return False
        print(f"Skipping {relative_path}: no legacy notebook URL was found")
        return False

    newline = "\r\n" if "\r\n" in original_text else "\n"
    rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
    if original_text.endswith(("\n", "\r\n")):
        rendered += newline
    path.write_text(rendered, encoding="utf-8", newline="")
    print(f"Updated {relative_path}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_notebook_url(repo_root, WELCOME_TEMPLATE_PATH) or changed_any
    changed_any = update_notebook_url(repo_root, WELCOME_NOTEBOOK_PATH) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any


SRC_WORKFLOW_PATH = Path(".github/workflows/update_on_src_change.yaml")
BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
WELCOME_TEMPLATE_PATH = Path(".tools/templates/Welcome_template.ipynb")
EXTERNAL_CODE_DOC_PATH = Path(".tools/docs/external_code_upload.md")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def replace_text(
    repo_root: Path,
    relative_path: Path,
    old: str,
    new: str,
    *,
    already_updated: str | None = None,
) -> bool:
    path = repo_root / relative_path
    if not path.exists():
        print(f"Skipping missing file: {relative_path}")
        return False

    original_text = read_text(path)
    if old in original_text:
        updated_text = original_text.replace(old, new, 1)
        write_text(path, updated_text)
        print(f"Updated {relative_path}")
        return True

    if already_updated and already_updated in original_text:
        return False

    raise ValueError(f"Unable to find expected text in {relative_path}")


def update_src_change_workflow(repo_root: Path) -> bool:
    return replace_text(
        repo_root,
        SRC_WORKFLOW_PATH,
        "      - 'src/**/*.py'",
        "      - 'src/**'",
        already_updated="      - 'src/**'",
    )


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text

    old_dst = '        dst = f"{project_folder}/src/{project_name}/{rel.replace(\'src/\', \'\')}"'
    new_dst = '        dst = f"{project_folder}/{rel}"'
    if old_dst in updated_text:
        updated_text = updated_text.replace(old_dst, new_dst, 1)
    elif new_dst not in updated_text:
        raise ValueError(f"Unable to find constructor src destination mapping in {BUMP_CONSTRUCTOR_PATH}")

    # Match the small formatting cleanup made in the template update.
    updated_text = updated_text.replace(
        f"import re{newline}{newline}{newline}def load_construct",
        f"import re{newline}{newline}def load_construct",
        1,
    )

    if updated_text == original_text:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def update_welcome_template(repo_root: Path) -> bool:
    path = repo_root / WELCOME_TEMPLATE_PATH
    if not path.exists():
        print(f"Skipping missing file: {WELCOME_TEMPLATE_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)

    try:
        notebook = json.loads(original_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse {WELCOME_TEMPLATE_PATH} as JSON") from exc

    legacy_assignment = re.compile(
        r"^(?P<indent>[ \t]*)src_folder\s*=\s*"
        r"Path\((?P<quote1>['\"])\.\.(?P=quote1)\)\s*/\s*"
        r"(?P<quote2>['\"])src(?P=quote2)\s*/\s*"
        r"(?P<quote3>['\"])[^'\"]+(?P=quote3)\s*$"
    )
    current_assignment = re.compile(
        r"^[ \t]*src_folder\s*=\s*"
        r"Path\((?P<quote1>['\"])\.\.(?P=quote1)\)\s*/\s*"
        r"(?P<quote2>['\"])src(?P=quote2)\s*$"
    )

    found_current = False
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_is_list = isinstance(source, list)
        lines = source if source_is_list else str(source).splitlines(keepends=True)
        updated_lines = []
        cell_changed = False

        for line in lines:
            line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            body = line[:-len(line_ending)] if line_ending else line

            if current_assignment.match(body):
                found_current = True
                updated_lines.append(line)
                continue

            match = legacy_assignment.match(body)
            if match:
                updated_lines.append(
                    f'{match.group("indent")}src_folder = Path("..") / "src"{line_ending}'
                )
                changed = True
                cell_changed = True
                continue

            updated_lines.append(line)

        if cell_changed:
            cell["source"] = updated_lines if source_is_list else "".join(updated_lines)

    if not changed:
        if found_current:
            return False
        print(
            f"Skipping {WELCOME_TEMPLATE_PATH}: no legacy src_folder assignment was found"
        )
        return False

    rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
    if original_text.endswith(("\n", "\r\n")):
        rendered += newline
    write_text(path, rendered)
    print(f"Updated {WELCOME_TEMPLATE_PATH}")
    return True


def detect_python_package_name(repo_root: Path, text: str = "") -> str:
    src_root = repo_root / "src"
    if src_root.is_dir():
        candidates = sorted(
            child.name
            for child in src_root.iterdir()
            if child.is_dir()
            and child.name.isidentifier()
            and (child / "__init__.py").is_file()
        )
        if len(candidates) == 1:
            return candidates[0]

    setup_path = repo_root / "setup.py"
    if setup_path.is_file():
        setup_text = read_text(setup_path)
        match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', setup_text)
        if match:
            return match.group(1).replace("-", "_")

    import_match = re.search(
        r"(?m)^import\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", text
    )
    if import_match:
        return import_match.group(1)

    return "PYTHON_PROJ_NAME"


def update_external_code_docs(repo_root: Path) -> bool:
    path = repo_root / EXTERNAL_CODE_DOC_PATH
    if not path.exists():
        print(f"Skipping missing file: {EXTERNAL_CODE_DOC_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    package_name = detect_python_package_name(repo_root, original_text)

    changed = False
    updated_text = original_text

    old_tree = "\n".join(
        [
            "```",
            "src",
            "|-- __init__.py",
            "|-- my_script.py",
            "|-- subpackage/",
            "    |-- __init__.py",
            "    |-- submodule1.py",
            "```",
        ]
    ).replace("\n", newline)
    new_tree = "\n".join(
        [
            "```text",
            "src/",
            f"|-- {package_name}/",
            "|   |-- __init__.py",
            "|   |-- my_script.py",
            "|   |-- subpackage/",
            "|       |-- __init__.py",
            "```",
        ]
    ).replace("\n", newline)

    if old_tree in updated_text:
        updated_text = updated_text.replace(old_tree, new_tree, 1)
        changed = True
    elif new_tree not in updated_text:
        print(
            f"Skipping directory tree update in {EXTERNAL_CODE_DOC_PATH}: "
            "the document uses a custom structure"
        )

    old_init = "# src/__init__.py"
    new_init = f"# src/{package_name}/__init__.py"
    if old_init in updated_text:
        updated_text = updated_text.replace(old_init, new_init, 1)
        changed = True

    old_submodule_pattern = re.compile(
        rf"(?m)^from\s+{re.escape(package_name)}\s+import\s+subpackage\s*$"
    )
    new_submodule = f"from {package_name}.subpackage import submodule1"
    if old_submodule_pattern.search(updated_text):
        updated_text = old_submodule_pattern.sub(new_submodule, updated_text, count=1)
        changed = True

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {EXTERNAL_CODE_DOC_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_src_change_workflow(repo_root) or changed_any
    changed_any = update_bump_constructor(repo_root) or changed_any
    changed_any = update_welcome_template(repo_root) or changed_any
    changed_any = update_external_code_docs(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

from __future__ import annotations

from pathlib import Path
from typing import Any


BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def replace_exact_block(text: str, old_block: str, new_block: str, path: Path, description: str) -> tuple[str, bool]:
    if old_block in text:
        return text.replace(old_block, new_block, 1), True
    if new_block in text:
        return text, False
    raise ValueError(f"Unable to find {description} in {path}")


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)

    old_block = newline.join(
        [
            "    # Include all the Python scripts under src/ directory",
            "    src_added = 0",
            "    included_src_flag = False",
            '    for py_file in src_root.rglob("*.py"):',
            "        rel = py_file.relative_to(repo_root).as_posix()",
            "        # Avoid any python files that are not in the src/ directory or any __init__.py files that are not in the src/ directory ",
            '        if not rel.startswith("src/") or py_file.name == "__init__.py":',
            "            continue",
            "        src = rel",
            '        dst = f"{project_folder}/{rel}"',
            "",
            "        # if src in existing_sources or dst in existing_dests:",
            "        #     continue",
            "",
            "        normalized_items.append({src: dst})",
            "        included_src_flag = True",
            "        src_added += 1",
            '        debug(f"Included source file: {src} -> {dst}")',
            "",
        ]
    )

    new_block = newline.join(
        [
            "    # Include all the Python scripts under src/ directory",
            "    src_added = 0",
            "    only_init_files = True",
            "    included_src_flag = False",
            '    for py_file in src_root.rglob("*.py"):',
            "        rel = py_file.relative_to(repo_root).as_posix()",
            "        ",
            "        # Avoid any python files that are not in the src/ directory ",
            '        if not rel.startswith("src/"):',
            "            continue",
            "",
            "        # Control if we only have __init__.py files",
            '        if not py_file.name == "__init__.py":',
            "            only_init_files = False",
            "        ",
            "        src = rel",
            '        dst = f"{project_folder}/{rel}"',
            "",
            "        # if src in existing_sources or dst in existing_dests:",
            "        #     continue",
            "",
            "        normalized_items.append({src: dst})",
            "        included_src_flag = True",
            "        src_added += 1",
            '        debug(f"Included source file: {src} -> {dst}")',
            "",
            "    if only_init_files:",
            '        debug("Only __init__.py files found under src/, skipping setup.py and src change marker packaging")',
            "        included_src_flag = False",
            "",
        ]
    )

    updated_text, changed = replace_exact_block(
        original_text,
        old_block,
        new_block,
        path,
        "src packaging block",
    )

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_bump_constructor(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

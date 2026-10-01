from __future__ import annotations

from pathlib import Path
from typing import Any
import re


BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
BUMP_VERSION_PATH = Path(".tools/python/bump_version.py")
CONSTRUCT_PATH = Path("construct.yaml")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def detect_project_folder(repo_root: Path) -> str:
    construct_path = repo_root / CONSTRUCT_PATH
    if construct_path.is_file():
        for raw_line in read_text(construct_path).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("name:"):
                value = stripped.split(":", 1)[1].strip().strip("\"'")
                if value:
                    return value
    return "PROJECT_NAME"


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text
    changed = False
    project_folder = detect_project_folder(repo_root)

    if 'DEBUG_PREFIX = "[bump_constructor]"' not in original_text:
        helper_anchor = "import re" + newline
        helper_block = newline.join(
            [
                "import re",
                "",
                'DEBUG_PREFIX = "[bump_constructor]"',
                "",
                "",
                "def debug(message: str) -> None:",
                '    print(f"{DEBUG_PREFIX} {message}")',
                "",
            ]
        )
        if helper_anchor not in updated_text:
            raise ValueError(f"Unable to find import insertion point in {BUMP_CONSTRUCTOR_PATH}")
        updated_text = updated_text.replace(helper_anchor, helper_block + newline, 1)
        changed = True

    requirements_start = "def ensure_requirements_in_extra_files(construct_data: dict):"
    requirements_end = "def ensure_extra_files(construct_data: dict, notebooks_root: Path, src_root: Path) -> int:"
    requirements_replacement = newline.join(
        [
            "def ensure_requirements_in_extra_files(construct_data: dict):",
            '    extra_files = construct_data.get("extra_files")',
            "    if extra_files is None:",
            "        extra_files = []",
            '        construct_data["extra_files"] = extra_files',
            "",
            "    # Check if requirements.txt is included, if not, add it",
            "    requirements_included = any(",
            '        isinstance(item, dict) and "requirements.txt" in item for item in extra_files',
            "    )",
            "    if not requirements_included:",
            '        extra_files.append({"requirements.txt": "PROJECT_NAME/requirements.txt"})',
            '        debug("Added mapping for requirements.txt")',
            "    else:",
            '        debug("requirements.txt mapping already present")',
            "",
            "    gpu_requirements_included = any(",
            '        isinstance(item, dict) and "requirements_gpu.txt" in item for item in extra_files',
            "    )",
            '    if not gpu_requirements_included and Path("requirements_gpu.txt").exists():',
            '        extra_files.append({"requirements_gpu.txt": "PROJECT_NAME/requirements_gpu.txt"})',
            '        debug("Added mapping for requirements_gpu.txt")',
            '    elif Path("requirements_gpu.txt").exists():',
            '        debug("requirements_gpu.txt mapping already present")',
            "    else:",
            '        debug("requirements_gpu.txt not found, skipping GPU requirements mapping")',
            "    ",
            "    # Check if requirements-linux.txt exists and is included, if not, add it",
            '    if Path("requirements-linux.txt").exists():',
            "        linux_included = any(",
            '            isinstance(item, dict) and "requirements-linux.txt" in item for item in extra_files',
            "        )",
            "        if not linux_included:",
            '            extra_files.append({"requirements-linux.txt": "PROJECT_NAME/requirements-linux.txt"})',
            '            debug("Added mapping for requirements-linux.txt")',
            "        else:",
            '            debug("requirements-linux.txt mapping already present")',
            "    else:",
            '        debug("requirements-linux.txt not found, skipping Linux requirements mapping")',
            "    # Check if requirements-windows.txt exists and is included, if not, add it",
            '    if Path("requirements-windows.txt").exists():',
            "        windows_included = any(",
            '            isinstance(item, dict) and "requirements-windows.txt" in item for item in extra_files',
            "        )",
            "        if not windows_included:",
            '            extra_files.append({"requirements-windows.txt": "PROJECT_NAME/requirements-windows.txt"})',
            '            debug("Added mapping for requirements-windows.txt")',
            "        else:",
            '            debug("requirements-windows.txt mapping already present")',
            "    else:",
            '        debug("requirements-windows.txt not found, skipping Windows requirements mapping")',
            "    ",
            "    # Check if requirements-macos.txt exists and is included, if not, add it",
            '    if Path("requirements-macos.txt").exists():',
            "        macos_included = any(",
            '            isinstance(item, dict) and "requirements-macos.txt" in item for item in extra_files',
            "        )",
            "        if not macos_included:",
            '            extra_files.append({"requirements-macos.txt": "PROJECT_NAME/requirements-macos.txt"})',
            '            debug("Added mapping for requirements-macos.txt")',
            "        else:",
            '            debug("requirements-macos.txt mapping already present")',
            "    else:",
            '        debug("requirements-macos.txt not found, skipping macOS requirements mapping")',
            "",
            "    # Update the construct data with the modified extra_files",
            '    construct_data["extra_files"] = extra_files',
            "",
        ]
    ).replace("PROJECT_NAME", project_folder)

    if 'debug("Added mapping for requirements.txt")' not in updated_text:
        start_index = updated_text.find(requirements_start)
        end_index = updated_text.find(requirements_end)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError(f"Unable to replace ensure_requirements_in_extra_files() in {BUMP_CONSTRUCTOR_PATH}")
        updated_text = updated_text[:start_index] + requirements_replacement + updated_text[end_index:]
        changed = True

    extra_start = "def ensure_extra_files(construct_data: dict, notebooks_root: Path, src_root: Path) -> int:"
    extra_end = "def main():"
    extra_replacement = newline.join(
        [
            "def ensure_extra_files(construct_data: dict, notebooks_root: Path, src_root: Path) -> int:",
            '    extra_files = construct_data.get("extra_files")',
            "    if extra_files is None:",
            "        extra_files = []",
            '        construct_data["extra_files"] = extra_files',
            "",
            "    # Extract the project folder name from existing mappings",
            "    project_folder = extract_project_folder(extra_files)",
            '    debug(f"Using project folder: {project_folder}")',
            "",
            "    # Normalize existing entries into a dict for quick lookup",
            "    existing_sources = set()",
            "    existing_dests = set()",
            "    normalized_items = []",
            "",
            "    for item in extra_files:",
            "        # Items can be either dicts (k: v) or strings with mapping? Assume dicts per example",
            "        if isinstance(item, dict):",
            "            for src, dst in item.items():",
            "                existing_sources.add(str(src))",
            "                existing_dests.add(str(dst))",
            "                normalized_items.append({str(src): str(dst)})",
            "        else:",
            "            # If strings are present, keep them",
            "            normalized_items.append(item)",
            "",
            "    ntbk_added = 0",
            '    repo_root = Path(".").resolve()',
            "",
            "    # For safety, remove any existing notebooks/*.ipynb entries to avoid duplicates, we will re-add them with correct paths",
            '    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src).startswith("notebooks/") and src.endswith(".ipynb") for src in item.keys()))]',
            "",
            "    # Include all notebooks under notebooks/ directory",
            "    for ipynb in notebooks_root.rglob(\"*.ipynb\"):",
            "        rel = ipynb.relative_to(repo_root).as_posix()",
            '        if not rel.startswith("notebooks/"):',
            "            continue",
            "        src = rel",
            '        dst = f"{project_folder}/{rel}"',
            "",
            "        # if src in existing_sources or dst in existing_dests:",
            "        #     continue",
            "",
            "        normalized_items.append({src: dst})",
            "        ntbk_added += 1",
            '        debug(f"Included notebook: {src} -> {dst}")',
            "",
            "    # First get the name of the package from setup.py",
            '    setup_path = repo_root / "setup.py"',
            '    project_name = "PROJECT_NAME"',
            "    if setup_path.exists():",
            '        setup_text = setup_path.read_text(encoding="utf-8")',
            '        name_match = re.search(r\'name\\s*=\\s*["\\\']([^"\\\']+)["\\\']\', setup_text)',
            "        if name_match:",
            "            project_name = name_match.group(1)",
            '        debug(f"Detected setup.py package name: {project_name}")',
            "    else:",
            '        debug("setup.py not found at repository root")',
            "",
            "    # For safety, remove any existing src/ entries to avoid duplicates, we will re-add them with correct paths",
            '    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src).startswith("src/") for src in item.keys()))]',
            "",
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
            "    # For safety also remove setup.py and src_change.yaml if they exist in the extra_files to avoid duplicates, we will re-add them with correct paths if src/ is included",
            '    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src) in ["setup.py", ".tools/meta/src_change.yaml"] for src in item.keys()))]  ',
            "",
            "    if included_src_flag:",
            "        # Also include the setup.py file at the root if not already included",
            '        setup_src = "setup.py"',
            '        setup_dst = f"{project_folder}/setup.py"',
            "        if setup_src not in existing_sources and setup_dst not in existing_dests:",
            "            normalized_items.append({setup_src: setup_dst})",
            "            src_added += 1",
            '            debug(f"Included setup.py: {setup_src} -> {setup_dst}")',
            "        else:",
            '            debug("setup.py mapping already present")',
            "",
            "        # Include the external code change tracking file if it exists",
            '        src_change_file = ".tools/meta/src_change.yaml"',
            "        if Path(src_change_file).exists():",
            '            src_change_dst = f"{project_folder}/src_change.yaml"',
            "            if src_change_file not in existing_sources and src_change_dst not in existing_dests:",
            "                normalized_items.append({src_change_file: src_change_dst})",
            "                src_added += 1",
            '                debug(f"Included src change marker: {src_change_file} -> {src_change_dst}")',
            "            else:",
            '                debug("src change marker mapping already present")',
            "        else:",
            '            debug("src change marker not found, skipping")',
            "    else:",
            "        # If the user has removed the src/directory, remove the generated setup.py and src_change.yaml",
            '        debug("No non-__init__.py files found under src/, skipping setup.py and src change marker packaging")',
            '        setup_dst = Path(project_folder) / "setup.py"',
            "        if setup_dst.exists():",
            "            setup_dst.unlink()",
            '        src_change_dst = Path(project_folder) / "src_change.yaml"',
            "        if src_change_dst.exists():",
            "            src_change_dst.unlink()",
            "",
            "    # Optionally sort entries (dicts by their single key) for determinism",
            "    def sort_key(item):",
            "        if isinstance(item, dict):",
            "            # single-key dict",
            "            k = next(iter(item.keys()))",
            "            return (0, k)",
            "        return (1, str(item))",
            "",
            "    normalized_items.sort(key=sort_key)",
            '    construct_data["extra_files"] = normalized_items',
            '    debug(f"Final extra_files entries: {len(normalized_items)}")',
            "    for item in normalized_items:",
            '        debug(f"extra_files entry: {item}")',
            "",
            "    return ntbk_added, src_added",
            "",
        ]
    ).replace("PROJECT_NAME", project_folder)

    if 'debug(f"Included notebook: {src} -> {dst}")' not in updated_text:
        start_index = updated_text.find(extra_start)
        end_index = updated_text.find(extra_end)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError(f"Unable to replace ensure_extra_files() in {BUMP_CONSTRUCTOR_PATH}")
        updated_text = updated_text[:start_index] + extra_replacement + updated_text[end_index:]
        changed = True

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def update_bump_version(repo_root: Path) -> bool:
    path = repo_root / BUMP_VERSION_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_VERSION_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text
    changed = False

    old_regex = 'THANKS_LINE_RE = re.compile(r\'(Thank you! You have successfully installed [^\\n]*)(\\d+\\.\\d+\\.\\d+)(!)\')'
    new_regex = newline.join(
        [
            "CONCLUSION_LINE_RE = re.compile(",
            '    r\'^(conclusion_text:\\s*)(?P<quote>["\\\'])(?P<body>.*?)(?P=quote)(?P<trailing>\\s*(?:#.*)?)$\',',
            "    re.MULTILINE,",
            ")",
        ]
    )
    if "CONCLUSION_LINE_RE = re.compile(" not in updated_text:
        if old_regex not in updated_text:
            raise ValueError(f"Unable to find conclusion regex in {BUMP_VERSION_PATH}")
        updated_text = updated_text.replace(old_regex, new_regex, 1)
        changed = True

    new_function = newline.join(
        [
            "def bump_construct_text(text: str, old_version: str, new_version: str) -> str:",
            '    """Return updated file text with bumped version, preserving YAML structure/comments.',
            "",
            "    - Updates the `version: \"X.Y.Z\"` line.",
            "    - Updates the version inside `conclusion_text`, replacing either",
            "      `VERSION_NUMBER` or the previous explicit version if present.",
            '    """',
            "    # 1) Update the explicit version line",
            "    def _repl_version(m: re.Match) -> str:",
            "        # Preserve original quoting and trailing comment/whitespace",
            "        return f\"{m.group(1)}{m.group('quote')}{new_version}{m.group('quote')}{m.group('trailing')}\"",
            "",
            "    text = VERSION_LINE_RE.sub(_repl_version, text, count=1)",
            "",
            "    # 2) Update the version inside conclusion_text if present.",
            "    def _repl_conclusion(m: re.Match) -> str:",
            "        body = m.group(\"body\")",
            "        if \"VERSION_NUMBER\" in body:",
            "            updated_body = body.replace(\"VERSION_NUMBER\", new_version)",
            "        else:",
            "            updated_body = body.replace(old_version, new_version)",
            "        return f\"{m.group(1)}{m.group('quote')}{updated_body}{m.group('quote')}{m.group('trailing')}\"",
            "",
            "    text = CONCLUSION_LINE_RE.sub(_repl_conclusion, text, count=1)",
            "    return text",
        ]
    ) + newline

    if "text = CONCLUSION_LINE_RE.sub(_repl_conclusion, text, count=1)" not in updated_text:
        start_marker = "def bump_construct_text(text: str, old_version: str, new_version: str) -> str:"
        end_marker = "def bump_post_install_bat(new_version: str) -> None:"
        start_index = updated_text.find(start_marker)
        end_index = updated_text.find(end_marker)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError(f"Unable to replace bump_construct_text() in {BUMP_VERSION_PATH}")
        updated_text = updated_text[:start_index] + new_function + updated_text[end_index:]
        changed = True

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_VERSION_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_bump_constructor(repo_root) or changed_any
    changed_any = update_bump_version(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

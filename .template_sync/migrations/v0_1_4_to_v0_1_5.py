from __future__ import annotations

from pathlib import Path
from typing import Any
import re


MERGE_REQUIREMENTS_PATH = Path(".tools/python/merge_requirements.py")
BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
POST_INSTALL_BAT_PATH = Path("app/bash_bat_scripts/post_install.bat")
POST_INSTALL_SH_PATH = Path("app/bash_bat_scripts/post_install.sh")
CONSTRUCT_PATH = Path("construct.yaml")
REQUIREMENTS_GPU_PATH = Path("requirements_gpu.txt")


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


def replace_text(
    text: str,
    old: str,
    new: str,
    *,
    already_updated: str | None = None,
) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new, 1), True

    if already_updated and already_updated in text:
        return text, False

    raise ValueError("Unable to find expected text to replace")


def update_merge_requirements(repo_root: Path) -> bool:
    path = repo_root / MERGE_REQUIREMENTS_PATH
    if not path.exists():
        print(f"Skipping missing file: {MERGE_REQUIREMENTS_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text
    changed = False

    old_doc = newline.join(
        [
            "It also recognizes optional per-OS fields (`windows_dependencies`,",
            "`linux_dependencies`, `macos_dependencies`) and emits dedicated requirements",
            "outputs for each platform containing only the entries from their OS-specific",
            "lists, in addition to the common file.",
            "",
            "It resolves simple version conflicts by choosing the newest version seen for a",
        ]
    )
    new_doc = newline.join(
        [
            "It also recognizes optional per-OS fields (`windows_dependencies`,",
            "`linux_dependencies`, `macos_dependencies`) and emits dedicated requirements",
            "outputs for each platform containing only the entries from their OS-specific",
            "lists, in addition to the common file.",
            "",
            "Alongside the common requirements file, it emits a GPU-targeted variant. The",
            "GPU file rewrites `tensorflow` to `tensorflow[and-cuda]` and adds the PyTorch",
            "CUDA wheel index when `torch`, `torchvision`, or `torchaudio` are present.",
            "",
            "It resolves simple version conflicts by choosing the newest version seen for a",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_doc,
        new_doc,
        already_updated="GPU file rewrites `tensorflow` to `tensorflow[and-cuda]`",
    )
    changed = did_change or changed

    old_constants = newline.join(
        [
            'OS_DISPLAY_NAMES = {',
            '    "windows": "Windows",',
            '    "linux": "Linux",',
            '    "macos": "macOS",',
            "}",
        ]
    )
    new_constants = newline.join(
        [
            'OS_DISPLAY_NAMES = {',
            '    "windows": "Windows",',
            '    "linux": "Linux",',
            '    "macos": "macOS",',
            "}",
            'GPU_DISPLAY_NAME = "NVIDIA GPU"',
            'GPU_TORCH_EXTRA_INDEX_URL = "--extra-index-url https://download.pytorch.org/whl/cu128"',
            'TORCH_GPU_PACKAGES = {"torch", "torchvision", "torchaudio"}',
            "PLAIN_PKG_RE = re.compile(",
            '    r"^(?P<name>[A-Za-z0-9_.+-]+)(?P<extras>\\[[^\\]]+\\])?(?P<marker>\\s*;.*)?$"',
            ")",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_constants,
        new_constants,
        already_updated='GPU_DISPLAY_NAME = "NVIDIA GPU"',
    )
    changed = did_change or changed

    old_helper_block = newline.join(
        [
            "def get_os_output_path(output: Path, os_name: str) -> Path:",
            "    suffix = output.suffix",
            "    stem = output.stem",
            '    return output.with_name(f"{stem}-{os_name}{suffix}")',
            "",
            "",
            "def write_requirements_output(path: Path, files: list[Path], final_lines: list[str], target_os: str | None = None):",
        ]
    )
    new_helper_block = newline.join(
        [
            "def get_os_output_path(output: Path, os_name: str) -> Path:",
            "    suffix = output.suffix",
            "    stem = output.stem",
            '    return output.with_name(f"{stem}-{os_name}{suffix}")',
            "",
            "",
            "def get_gpu_output_path(output: Path) -> Path:",
            "    suffix = output.suffix",
            "    stem = output.stem",
            '    return output.with_name(f"{stem}_gpu{suffix}")',
            "",
            "",
            "def merge_requirement_extras(extras: str, extra_name: str) -> str:",
            "    if not extras:",
            '        return f"[{extra_name}]"',
            "",
            "    extras_content = extras[1:-1]",
            '    parts = [part.strip() for part in extras_content.split(",") if part.strip()]',
            "    if extra_name not in parts:",
            "        parts.append(extra_name)",
            '    return "[" + ",".join(parts) + "]"',
            "",
            "",
            "def rewrite_tensorflow_for_gpu(line: str) -> str:",
            "    match = pkg_re.match(line)",
            "    if match:",
            '        name = match.group("name")',
            '        if name.lower() != "tensorflow":',
            "            return line",
            '        extras = merge_requirement_extras(match.group("extras") or "", "and-cuda")',
            '        marker = match.group("marker") or ""',
            '        return f"{name}{extras}{match.group(\'op\')}{match.group(\'ver\')}{marker}"',
            "",
            "    match = PLAIN_PKG_RE.match(line)",
            '    if match and match.group("name").lower() == "tensorflow":',
            '        extras = merge_requirement_extras(match.group("extras") or "", "and-cuda")',
            '        marker = match.group("marker") or ""',
            '        return f"{match.group(\'name\')}{extras}{marker}"',
            "",
            "    return line",
            "",
            "",
            "def build_gpu_lines(final_lines: list[str]) -> list[str]:",
            "    gpu_lines: list[str] = []",
            "    has_torch_package = False",
            "    has_torch_index = False",
            "",
            "    for line in final_lines:",
            '        if "download.pytorch.org/whl" in line:',
            "            has_torch_index = True",
            "",
            '        match = pkg_re.match(line) or PLAIN_PKG_RE.match(line)',
            "        if match:",
            '            package_name = match.group("name").lower()',
            '            if package_name == "tensorflow":',
            "                line = rewrite_tensorflow_for_gpu(line)",
            "            if package_name in TORCH_GPU_PACKAGES:",
            "                has_torch_package = True",
            "",
            "        gpu_lines.append(line)",
            "",
            "    if has_torch_package and not has_torch_index:",
            "        insertion_index = 0",
            '        while insertion_index < len(gpu_lines) and gpu_lines[insertion_index].startswith("-"):',
            "            insertion_index += 1",
            "        gpu_lines.insert(insertion_index, GPU_TORCH_EXTRA_INDEX_URL)",
            "",
            "    return gpu_lines",
            "",
            "",
            "def write_requirements_output(",
            "    path: Path,",
            "    files: list[Path],",
            "    final_lines: list[str],",
            "    target_os: str | None = None,",
            "    target_accelerator: str | None = None,",
            "):",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_helper_block,
        new_helper_block,
        already_updated="def get_gpu_output_path(output: Path) -> Path:",
    )
    changed = did_change or changed

    old_header_block = newline.join(
        [
            "    if target_os:",
            '        header_lines.append(f"# Target OS: {OS_DISPLAY_NAMES.get(target_os, target_os.capitalize())}")',
            '    header_lines.append("# ---")',
        ]
    )
    new_header_block = newline.join(
        [
            "    if target_os:",
            '        header_lines.append(f"# Target OS: {OS_DISPLAY_NAMES.get(target_os, target_os.capitalize())}")',
            "    if target_accelerator:",
            '        header_lines.append(f"# Target accelerator: {target_accelerator}")',
            '    header_lines.append("# ---")',
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_header_block,
        new_header_block,
        already_updated='        header_lines.append(f"# Target accelerator: {target_accelerator}")',
    )
    changed = did_change or changed

    updated_text, did_change = replace_text(
        updated_text,
        "    skip_outputs = {output.resolve()}",
        "    skip_outputs = {output.resolve()}" + newline + "    skip_outputs.add(get_gpu_output_path(output).resolve())",
        already_updated="    skip_outputs.add(get_gpu_output_path(output).resolve())",
    )
    changed = did_change or changed

    old_no_files = newline.join(
        [
            '        print(f"No requirements files found under {source}, writing empty {output}")',
            '        output.write_text("# Generated requirements (none found)\\n", encoding="utf-8")',
            "        return 0",
        ]
    )
    new_no_files = newline.join(
        [
            '        print(f"No requirements files found under {source}, writing empty {output}")',
            '        output.write_text("# Generated requirements (none found)\\n", encoding="utf-8")',
            "        gpu_output = get_gpu_output_path(output)",
            '        gpu_output.write_text("# Generated GPU requirements (none found)\\n", encoding="utf-8")',
            "        return 0",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_no_files,
        new_no_files,
        already_updated='        gpu_output.write_text("# Generated GPU requirements (none found)\\n", encoding="utf-8")',
    )
    changed = did_change or changed

    old_gpu_write_block = newline.join(
        [
            "    base_final_lines = build_final_lines(base_ctx, args.sort)",
            "    write_requirements_output(output, files, base_final_lines)",
            '    print(f"Wrote {output} with {len(base_final_lines)} entries (from {len(files)} files)")',
            "",
            "    for os_name, ctx in os_contexts.items():",
        ]
    )
    new_gpu_write_block = newline.join(
        [
            "    base_final_lines = build_final_lines(base_ctx, args.sort)",
            "    write_requirements_output(output, files, base_final_lines)",
            '    print(f"Wrote {output} with {len(base_final_lines)} entries (from {len(files)} files)")',
            "",
            "    gpu_output = get_gpu_output_path(output)",
            "    gpu_final_lines = build_gpu_lines(base_final_lines)",
            "    write_requirements_output(",
            "        gpu_output,",
            "        files,",
            "        gpu_final_lines,",
            "        target_accelerator=GPU_DISPLAY_NAME,",
            "    )",
            '    print(f"Wrote {gpu_output} with {len(gpu_final_lines)} entries (from {len(files)} files, target {GPU_DISPLAY_NAME})")',
            "",
            "    for os_name, ctx in os_contexts.items():",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_gpu_write_block,
        new_gpu_write_block,
        already_updated="    gpu_output = get_gpu_output_path(output)",
    )
    changed = did_change or changed

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {MERGE_REQUIREMENTS_PATH}")
    return True


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    if '        isinstance(item, dict) and "requirements_gpu.txt" in item for item in extra_files' in original_text:
        return False

    lines = original_text.splitlines(keepends=True)
    project_folder = detect_project_folder(repo_root)
    gpu_block = (
        newline.join(
            [
                "    gpu_requirements_included = any(",
                '        isinstance(item, dict) and "requirements_gpu.txt" in item for item in extra_files',
                "    )",
                '    if not gpu_requirements_included and Path("requirements_gpu.txt").exists():',
                f'        extra_files.append({{"requirements_gpu.txt": "{project_folder}/requirements_gpu.txt"}})',
            ]
        )
        + newline
    )
    func_start = None
    func_end = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ensure_requirements_in_extra_files("):
            func_start = index
            continue
        if func_start is not None and stripped.startswith("def ") and not line.startswith((" ", "\t")):
            func_end = index
            break

    if func_start is None:
        raise ValueError(f"Unable to find ensure_requirements_in_extra_files() in {BUMP_CONSTRUCTOR_PATH}")

    insert_at = None
    candidate_prefixes = (
        "# Check if requirements-linux.txt",
        '# Check if requirements-windows.txt',
        '# Check if requirements-macos.txt',
        'if Path("requirements-linux.txt").exists():',
        "if Path('requirements-linux.txt').exists():",
        'if Path("requirements-windows.txt").exists():',
        "if Path('requirements-windows.txt').exists():",
        'if Path("requirements-macos.txt").exists():',
        "if Path('requirements-macos.txt').exists():",
        '# Update the construct data with the modified extra_files',
        'construct_data["extra_files"] = extra_files',
        "construct_data['extra_files'] = extra_files",
    )

    for index in range(func_start + 1, func_end):
        stripped = lines[index].strip()
        if stripped.startswith(candidate_prefixes):
            insert_at = index
            break

    if insert_at is None:
        requirements_append_re = re.compile(
            r"extra_files\.append\(\{['\"]requirements\.txt['\"]:\s*['\"].*requirements\.txt['\"]\}\)"
        )
        for index in range(func_start + 1, func_end):
            if requirements_append_re.search(lines[index]):
                insert_at = index + 1
                while insert_at < func_end and lines[insert_at].strip() == "":
                    insert_at += 1
                break

    if insert_at is None:
        insert_at = func_end
        while insert_at > func_start + 1 and lines[insert_at - 1].strip() == "":
            insert_at -= 1

    lines.insert(insert_at, newline)
    lines.insert(insert_at, gpu_block)
    updated_text = "".join(lines)
    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def update_post_install_bat(repo_root: Path) -> bool:
    path = repo_root / POST_INSTALL_BAT_PATH
    if not path.exists():
        print(f"Skipping missing file: {POST_INSTALL_BAT_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    lines = original_text.splitlines(keepends=True)
    changed = False

    project_folder = detect_project_folder(repo_root)
    gpu_requirements_line = f'SET "GPU_REQUIREMENTS=%PREFIX%\\{project_folder}\\requirements_gpu.txt"'
    if gpu_requirements_line not in original_text:
        echo_line = 'echo Running post_install > "%PREFIX%\\menuinst_debug.log"'
        selected_install_line = '"%PREFIX%\\python.exe" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"'
        block_lines = [
            f'SET "BASE_REQUIREMENTS=%PREFIX%\\{project_folder}\\requirements.txt"',
            gpu_requirements_line,
            'SET "SELECTED_REQUIREMENTS=%BASE_REQUIREMENTS%"',
            "",
            'IF EXIST "%GPU_REQUIREMENTS%" (',
            "    where nvidia-smi >NUL 2>&1",
            "    IF ERRORLEVEL 1 (",
            '        echo NVIDIA GPU not detected, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
            "    ) ELSE (",
            '        echo NVIDIA GPU detected, installing GPU requirements from "%GPU_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
            '        SET "SELECTED_REQUIREMENTS=%GPU_REQUIREMENTS%"',
            "    )",
            ") ELSE (",
            '    echo GPU requirements file not found, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
            ")",
            "",
        ]

        if not any(line.rstrip("\r\n") == "SETLOCAL" for line in lines):
            for index, line in enumerate(lines):
                if line.rstrip("\r\n") == "@ECHO OFF":
                    lines.insert(index + 1, "SETLOCAL" + newline)
                    changed = True
                    break

        echo_index = None
        install_index = None
        for index, line in enumerate(lines):
            stripped = line.rstrip("\r\n")
            if stripped == echo_line:
                echo_index = index
            if (
                stripped.startswith('"%PREFIX%\\python.exe" -m pip install -r ')
                and "requirements.txt" in stripped
                and "requirements-windows.txt" not in stripped
                and "%SELECTED_REQUIREMENTS%" not in stripped
            ):
                install_index = index
                break

        if echo_index is None or install_index is None:
            raise ValueError(f"Unable to find insertion point in {POST_INSTALL_BAT_PATH}")

        insertion = [entry + newline for entry in block_lines]
        for offset, entry in enumerate(insertion, start=1):
            lines.insert(echo_index + offset, entry)
        changed = True

        install_index += len(insertion)
        lines[install_index] = selected_install_line + newline

    has_endlocal = any(line.rstrip("\r\n") == "ENDLOCAL" for line in lines)
    if not has_endlocal:
        replaced = False
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == "SetLocal EnableDelayedExpansion":
                lines[index] = "ENDLOCAL" + newline
                replaced = True
                changed = True
                break

        if not replaced:
            for index, line in enumerate(lines):
                if line.rstrip("\r\n") == "echo Post-install completed!":
                    lines.insert(index + 1, "ENDLOCAL" + newline)
                    changed = True
                    replaced = True
                    break

        if not replaced:
            lines.append("ENDLOCAL" + newline)
            changed = True

    if not changed:
        return False

    updated_text = "".join(lines)
    write_text(path, updated_text)
    print(f"Updated {POST_INSTALL_BAT_PATH}")
    return True


def update_post_install_sh(repo_root: Path) -> bool:
    path = repo_root / POST_INSTALL_SH_PATH
    if not path.exists():
        print(f"Skipping missing file: {POST_INSTALL_SH_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    project_folder = detect_project_folder(repo_root)
    gpu_requirements_line = f'GPU_REQUIREMENTS="$PREFIX/{project_folder}/requirements_gpu.txt"'
    if gpu_requirements_line in original_text:
        return False

    lines = original_text.splitlines(keepends=True)
    echo_line = 'echo "Running post_install" > "$PREFIX/menuinst_debug.log"'
    selected_install_line = '"$PREFIX/bin/python" -m pip install -r "$SELECTED_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"'
    block_lines = [
        f'BASE_REQUIREMENTS="$PREFIX/{project_folder}/requirements.txt"',
        gpu_requirements_line,
        'SELECTED_REQUIREMENTS="$BASE_REQUIREMENTS"',
        "",
        'if [ -f "$GPU_REQUIREMENTS" ]; then',
        '    if [[ "$OSTYPE" == "darwin"* ]]; then',
        '        echo "macOS detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"',
        "    elif command -v nvidia-smi >/dev/null 2>&1; then",
        '        echo "NVIDIA GPU detected, installing GPU requirements from $GPU_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"',
        '        SELECTED_REQUIREMENTS="$GPU_REQUIREMENTS"',
        "    else",
        '        echo "NVIDIA GPU not detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"',
        "    fi",
        "else",
        '    echo "GPU requirements file not found, installing CPU requirements from $BASE_REQUIREMENTS" >> "$PREFIX/menuinst_debug.log"',
        "fi",
        "",
    ]

    echo_index = None
    install_index = None
    for index, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if stripped == echo_line:
            echo_index = index
        if (
            stripped.startswith('"$PREFIX/bin/python" -m pip install -r ')
            and 'requirements.txt" >> "$PREFIX/menuinst_debug.log"' in stripped
            and "requirements-macos.txt" not in stripped
            and "requirements-linux.txt" not in stripped
            and "$SELECTED_REQUIREMENTS" not in stripped
        ):
            install_index = index
            break

    if echo_index is None or install_index is None:
        raise ValueError(f"Unable to find insertion point in {POST_INSTALL_SH_PATH}")

    insertion = [entry + newline for entry in block_lines]
    for offset, entry in enumerate(insertion, start=1):
        lines.insert(echo_index + offset, entry)

    install_index += len(insertion)
    lines[install_index] = selected_install_line + newline

    updated_text = "".join(lines)
    write_text(path, updated_text)
    print(f"Updated {POST_INSTALL_SH_PATH}")
    return True


def update_construct(repo_root: Path) -> bool:
    path = repo_root / CONSTRUCT_PATH
    if not path.exists():
        print(f"Skipping missing file: {CONSTRUCT_PATH}")
        return False

    original_text = read_text(path)
    if "- requirements_gpu.txt:" in original_text:
        return False

    pattern = re.compile(
        r"^(?P<indent>\s*-\s*)requirements\.txt:\s*(?P<dst_prefix>.+?/)?requirements\.txt\s*$",
        re.MULTILINE,
    )
    match = pattern.search(original_text)
    if not match:
        raise ValueError(f"Unable to find requirements.txt mapping in {CONSTRUCT_PATH}")

    dst_prefix = match.group("dst_prefix") or ""
    replacement = (
        match.group(0)
        + detect_newline(original_text)
        + f"{match.group('indent')}requirements_gpu.txt: {dst_prefix}requirements_gpu.txt"
    )
    updated_text = pattern.sub(replacement, original_text, count=1)
    write_text(path, updated_text)
    print(f"Updated {CONSTRUCT_PATH}")
    return True


def ensure_requirements_gpu(repo_root: Path) -> bool:
    path = repo_root / REQUIREMENTS_GPU_PATH
    if path.exists():
        return False

    write_text(path, "")
    print(f"Created {REQUIREMENTS_GPU_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_merge_requirements(repo_root) or changed_any
    changed_any = update_bump_constructor(repo_root) or changed_any
    changed_any = update_post_install_bat(repo_root) or changed_any
    changed_any = update_post_install_sh(repo_root) or changed_any
    changed_any = update_construct(repo_root) or changed_any
    changed_any = ensure_requirements_gpu(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

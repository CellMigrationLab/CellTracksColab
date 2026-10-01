from __future__ import annotations

from pathlib import Path
from typing import Any
import re


BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
BUMP_VERSION_PATH = Path(".tools/python/bump_version.py")


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


def update_bump_version(repo_root: Path) -> bool:
    path = repo_root / BUMP_VERSION_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_VERSION_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    changed = False

    old_conclusion_regex_block = newline.join(
        [
            "CONCLUSION_LINE_RE = re.compile(",
            '    r\'^(conclusion_text:\\s*)(?P<quote>["\\\'])(?P<body>.*?)(?P=quote)(?P<trailing>\\s*(?:#.*)?)$\',',
            "    re.MULTILINE,",
            ")",
        ]
    )
    new_conclusion_regex_block = newline.join(
        [
            "CONCLUSION_LINE_RE = re.compile(",
            '    r\'^(conclusion_text:\\s*)(?:(?P<quote>["\\\'])(?P<quoted_body>.*?)(?P=quote)|(?P<bare_body>.*?))(?P<trailing>\\s*(?:#.*)?)$\',',
            "    re.MULTILINE,",
            ")",
        ]
    )

    updated_text, block_changed = replace_exact_block(
        original_text,
        old_conclusion_regex_block,
        new_conclusion_regex_block,
        path,
        "CONCLUSION_LINE_RE block",
    )
    changed = changed or block_changed

    old_repl_conclusion_block = newline.join(
        [
            "    # 2) Update the version inside conclusion_text if present.",
            "    def _repl_conclusion(m: re.Match) -> str:",
            '        body = m.group("body")',
            '        if "VERSION_NUMBER" in body:',
            '            updated_body = body.replace("VERSION_NUMBER", new_version)',
            "        else:",
            "            updated_body = body.replace(old_version, new_version)",
            '        return f"{m.group(1)}{m.group(\'quote\')}{updated_body}{m.group(\'quote\')}{m.group(\'trailing\')}"',
        ]
    )
    new_repl_conclusion_block = newline.join(
        [
            "    # 2) Update the version inside conclusion_text if present.",
            "    def _repl_conclusion(m: re.Match) -> str:",
            '        quote = m.group("quote") or ""',
            '        body = m.group("quoted_body") if quote else (m.group("bare_body") or "")',
            '        if "VERSION_NUMBER" in body:',
            '            updated_body = body.replace("VERSION_NUMBER", new_version)',
            "        else:",
            "            updated_body = body.replace(old_version, new_version)",
            '        return f"{m.group(1)}{quote}{updated_body}{quote}{m.group(\'trailing\')}"',
        ]
    )

    updated_text, block_changed = replace_exact_block(
        updated_text,
        old_repl_conclusion_block,
        new_repl_conclusion_block,
        path,
        "_repl_conclusion block",
    )
    changed = changed or block_changed

    updated_block = newline.join(
        [
            "def bump_version_in_download_executable_md(new_version: str) -> None:",
            "    possible_download_md_paths = [",
            '        ROOT / "docs" / "download_executable.md",',
            '        ROOT / ".tools" / "docs" / "download_executable.md"',
            "    ]",
            '    template_md = ROOT / ".tools" / "templates" / "download_executable_template.md"',
            "",
            "    for download_md in possible_download_md_paths:",
            "        # We only want to update the existing files",
            "        if download_md.exists():",
            "            # Replace download_executable.md with the template (but only if it exists)",
            "            if template_md.exists():",
            "                # Remove existing download_executable.md if present",
            "                if download_md.exists():",
            "                    download_md.unlink()",
            "                # Copy the template to the docs folder using shutil for cross-platform support",
            "                shutil.copy(template_md, download_md)",
            "            else:",
            '                print("Template for download_executable.md not found! Skipping creation of download_executable.md")',
            "                return",
            "",
            "            # This file contains placeholders, update them with the new version",
            '            text = download_md.read_text(encoding="utf-8")',
            "            updated_text = replace_version_placeholder(text, new_version)",
            "            if updated_text != text:",
            '                download_md.write_text(updated_text, encoding="utf-8")',
            '                print(f"Updated download_executable.md to version {new_version}")',
            "            else:",
            '                print("No version string found in download_executable.md to update.")',
        ]
    )

    start_marker = "def bump_version_in_download_executable_md(new_version: str) -> None:"
    end_marker = f"{newline}def main() -> None:"

    start = updated_text.find(start_marker)
    if start == -1:
        raise ValueError(f"Unable to find function start in {BUMP_VERSION_PATH}")

    end = updated_text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"Unable to find function end before main() in {BUMP_VERSION_PATH}")

    current_block = updated_text[start:end].rstrip("\r\n")
    if current_block == updated_block:
        if not changed:
            return False
        write_text(path, updated_text)
        print(f"Updated {BUMP_VERSION_PATH}")
        return True

    updated_text = updated_text[:start] + updated_block + newline + updated_text[end + len(newline):]
    write_text(path, updated_text)
    print(f"Updated {BUMP_VERSION_PATH}")
    return True


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)

    updated_block = newline.join(
        [
            "        normalized_items.append({setup_src: setup_dst})",
            "        src_added += 1",
            '        debug(f"Included setup.py: {setup_src} -> {setup_dst}")',
        ]
    )

    if updated_block in original_text and 'if setup_src not in existing_sources and setup_dst not in existing_dests:' not in original_text:
        return False

    patterns = [
        re.compile(
            r'        if setup_src not in existing_sources and setup_dst not in existing_dests:\r?\n'
            r'            normalized_items\.append\(\{setup_src: setup_dst\}\)\r?\n'
            r'            src_added \+= 1\r?\n'
            r'(?:            debug\(f"Included setup\.py: \{setup_src\} -> \{setup_dst\}"\)\r?\n)?'
            r'        else:\r?\n'
            r'(?:            debug\("setup\.py mapping already present"\)\r?\n)?'
        ),
        re.compile(
            r'        if setup_src not in existing_sources and setup_dst not in existing_dests:\r?\n'
            r'            normalized_items\.append\(\{setup_src: setup_dst\}\)\r?\n'
            r'            src_added \+= 1\r?\n'
        ),
    ]

    replacement = updated_block + newline
    updated_text = original_text
    changed = False

    for pattern in patterns:
        updated_text, replacements = pattern.subn(replacement, updated_text, count=1)
        if replacements:
            changed = True
            break

    if not changed:
        raise ValueError(f"Unable to find setup.py conditional block in {BUMP_CONSTRUCTOR_PATH}")

    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_bump_version(repo_root) or changed_any
    changed_any = update_bump_constructor(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

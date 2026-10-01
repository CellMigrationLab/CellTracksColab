from __future__ import annotations

from pathlib import Path
from typing import Any


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


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text
    changed = False

    old_src_filter = newline.join(
        [
            '        if not rel.startswith("src/"):',
            "            continue",
        ]
    )
    new_src_filter = newline.join(
        [
            "        # Avoid any python files that are not in the src/ directory or any __init__.py files that are not in the src/ directory ",
            '        if not rel.startswith("src/") or py_file.name == "__init__.py":',
            "            continue",
        ]
    )
    # Later template versions deliberately include __init__.py files and track
    # whether they are the only source files. Treat that feature as newer than
    # this migration instead of rewriting it back to the 0.1.4 intermediate form.
    if "only_init_files = True" not in updated_text:
        updated_text, did_change = replace_text(
            updated_text,
            old_src_filter,
            new_src_filter,
            already_updated=new_src_filter,
        )
        changed = did_change or changed

    old_setup_cleanup_anchor = newline.join(
        [
            "        normalized_items.append({src: dst})",
            "        included_src_flag = True",
            "        src_added += 1",
            "",
            "    if included_src_flag:",
        ]
    )
    new_setup_cleanup_anchor = newline.join(
        [
            "        normalized_items.append({src: dst})",
            "        included_src_flag = True",
            "        src_added += 1",
            "",
            "    # For safety also remove setup.py and src_change.yaml if they exist in the extra_files to avoid duplicates, we will re-add them with correct paths if src/ is included",
            '    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src) in ["setup.py", ".tools/meta/src_change.yaml"] for src in item.keys()))]  ',
            "",
            "    if included_src_flag:",
        ]
    )
    cleanup_feature = 'any(str(src) in ["setup.py", ".tools/meta/src_change.yaml"]'
    if cleanup_feature not in updated_text:
        updated_text, did_change = replace_text(
            updated_text,
            old_setup_cleanup_anchor,
            new_setup_cleanup_anchor,
            already_updated=new_setup_cleanup_anchor,
        )
        changed = did_change or changed

    old_else_anchor = newline.join(
        [
            "        if Path(src_change_file).exists():",
            '            src_change_dst = f"{project_folder}/src_change.yaml"',
            "            if src_change_file not in existing_sources and src_change_dst not in existing_dests:",
            "                normalized_items.append({src_change_file: src_change_dst})",
            "                src_added += 1",
            "",
            "    # Optionally sort entries (dicts by their single key) for determinism",
        ]
    )
    new_else_anchor = newline.join(
        [
            "        if Path(src_change_file).exists():",
            '            src_change_dst = f"{project_folder}/src_change.yaml"',
            "            if src_change_file not in existing_sources and src_change_dst not in existing_dests:",
            "                normalized_items.append({src_change_file: src_change_dst})",
            "                src_added += 1",
            "    else:",
            "        # If the user has removed the src/directory, remove the generated setup.py and src_change.yaml",
            '        setup_dst = Path(project_folder) / "setup.py"',
            "        if setup_dst.exists():",
            "            setup_dst.unlink()",
            '        src_change_dst = Path(project_folder) / "src_change.yaml"',
            "        if src_change_dst.exists():",
            "            src_change_dst.unlink()",
            "",
            "    # Optionally sort entries (dicts by their single key) for determinism",
        ]
    )
    removal_feature = 'setup_dst = Path(project_folder) / "setup.py"'
    if removal_feature not in updated_text:
        updated_text, did_change = replace_text(
            updated_text,
            old_else_anchor,
            new_else_anchor,
            already_updated=new_else_anchor,
        )
        changed = did_change or changed

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

    old_helper_anchor = 'POST_INSTALL = ROOT / "app" / "bash_bat_scripts" / "post_install.bat"' + newline + newline
    new_helper_anchor = newline.join(
        [
            'POST_INSTALL = ROOT / "app" / "bash_bat_scripts" / "post_install.bat"',
            "",
            "def replace_version_placeholder(text: str, new_version: str) -> str:",
            '    """ Replace version placeholders \'VERSION_NUMBER\' in the text with the new version."""',
            '    return text.replace("VERSION_NUMBER", new_version)',
            "",
        ]
    )
    helper_signature = "def replace_version_placeholder(text: str, new_version: str) -> str:"
    if helper_signature not in updated_text:
        if old_helper_anchor not in updated_text:
            raise ValueError(f"Unable to find helper insertion point in {BUMP_VERSION_PATH}")
        updated_text = updated_text.replace(old_helper_anchor, new_helper_anchor, 1)
        changed = True

    old_download_function = newline.join(
        [
            "def bump_version_in_download_executable_md(new_version: str) -> None:",
            '    download_md = ROOT / "docs" / "download_executable.md"',
            "    if not download_md.exists():",
            "        return",
            "    # This file may contain the version in various formats, so we use the flexible replacement function",
            "    if replace_version_in_file(download_md, new_version, new_version):",
            '        print(f"Updated download_executable.md to version {new_version}")',
            "    else:",
            '        print("No version string found in download_executable.md to update.")',
        ]
    )
    new_download_function = newline.join(
        [
            "def bump_version_in_download_executable_md(new_version: str) -> None:",
            '    download_md = ROOT / "docs" / "download_executable.md"',
            '    template_md = ROOT / ".tools" / "templates" / "download_executable_template.md"',
            "",
            "    # On the first release replace download_executable.md with the template",
            "    # (but only if it exists)",
            "    if template_md.exists():",
            "        # Remove existing download_executable.md if present",
            "        if download_md.exists():",
            "            download_md.unlink()",
            "        # Copy the template to the docs folder using shutil for cross-platform support",
            "        shutil.copy(template_md, download_md)",
            "    else:",
            '        print("Template for download_executable.md not found! Skipping creation of download_executable.md")',
            "        return",
            "    ",
            "    # This file contains placeholders, update them with the new version",
            '    text = download_md.read_text(encoding="utf-8")',
            "    updated_text = replace_version_placeholder(text, new_version)",
            "    if updated_text != text:",
            '        download_md.write_text(updated_text, encoding="utf-8")',
            '        print(f"Updated download_executable.md to version {new_version}")',
            "    else:",
            '        print("No version string found in download_executable.md to update.")',
        ]
    )
    if old_download_function in updated_text:
        updated_text = updated_text.replace(old_download_function, new_download_function, 1)
        changed = True
    elif '    template_md = ROOT / ".tools" / "templates" / "download_executable_template.md"' not in updated_text:
        # Early initialized repositories did not yet have this helper at all.
        main_anchor = "def main() -> None:"
        if main_anchor not in updated_text:
            raise ValueError(f"Unable to find main() insertion point in {BUMP_VERSION_PATH}")
        updated_text = updated_text.replace(
            main_anchor,
            new_download_function + newline + newline + main_anchor,
            1,
        )
        changed = True

    old_main_template_copy = newline.join(
        [
            "    # On the first release replace download_executable.md with the template",
            "    # (but only if it exists)",
            '    if pathlib.Path(".tools/templates/download_executable_template.md").exists():',
            "        # Remove existing download_executable.md if present",
            '        if pathlib.Path(".tools/docs/download_executable.md").exists():',
            '            pathlib.Path(".tools/docs/download_executable.md").unlink()',
            "        # Copy the template to the docs folder using shutil for cross-platform support",
            "        shutil.copy(",
            '            ".tools/templates/download_executable_template.md",',
            '            ".tools/docs/download_executable.md",',
            "        )",
            "",
        ]
    )
    updated_text, did_change = replace_text(
        updated_text,
        old_main_template_copy,
        "",
        already_updated="    text, current = read_current_version()",
    )
    changed = did_change or changed

    extra_blank_line = (
        '        sys.exit("Version must look like X.Y.Z")'
        + newline
        + newline
        + newline
        + "    text, current = read_current_version()"
    )
    normalized_spacing = (
        '        sys.exit("Version must look like X.Y.Z")'
        + newline
        + newline
        + "    text, current = read_current_version()"
    )
    if extra_blank_line in updated_text:
        updated_text = updated_text.replace(extra_blank_line, normalized_spacing, 1)
        changed = True

    old_download_comment = "    # Update download_executable.md if it exists"
    new_download_block = newline.join(
        [
            "    # Update download_executable.md",
            "    bump_version_in_download_executable_md(args.new_version)",
        ]
    )
    if old_download_comment in updated_text:
        updated_text = updated_text.replace(
            old_download_comment,
            "    # Update download_executable.md",
            1,
        )
        changed = True
    elif "bump_version_in_download_executable_md(args.new_version)" not in updated_text:
        post_install_call = "    bump_post_install_bat(args.new_version)"
        if post_install_call not in updated_text:
            raise ValueError(f"Unable to find post-install call in {BUMP_VERSION_PATH}")
        updated_text = updated_text.replace(
            post_install_call,
            post_install_call + newline + newline + new_download_block,
            1,
        )
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

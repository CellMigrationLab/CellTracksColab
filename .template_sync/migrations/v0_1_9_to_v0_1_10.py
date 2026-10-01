from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "v0_1_9_to_v0_1_10"
CONSTRUCT_PATH = Path("construct.yaml")
ENVIRONMENT_PATH = Path("environment.yaml")
POST_INSTALL_BAT_PATH = Path("app/bash_bat_scripts/post_install.bat")
POST_INSTALL_SH_PATH = Path("app/bash_bat_scripts/post_install.sh")
NOTEBOOK_LAUNCHER_PATH = Path("app/menuinst/notebook_launcher.json")
LAUNCH_JUPYTER_PATH = Path("app/python_scripts/launch_jupyter.py")
TROUBLESHOOTING_PATH = Path(".tools/docs/troubleshooting.md")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def strip_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def extract_yaml_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
    if not match:
        return None
    return strip_quotes(match.group(1))


def extract_project_folder(construct_text: str) -> str | None:
    for pattern in (
        r"app/menuinst/Welcome\.ipynb:\s*([^\r\n/]+)/notebooks/Welcome\.ipynb",
        r"app/menuinst/notebook_launcher\.json:\s*([^\r\n/]+)/notebook_launcher\.json",
        r"requirements\.txt:\s*([^\r\n/]+)/requirements\.txt",
    ):
        match = re.search(pattern, construct_text)
        if match:
            return match.group(1).strip()
    return None


def parse_github_owner(remote_url: str) -> str | None:
    normalized = remote_url.strip()
    for pattern in (
        r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
    ):
        match = re.search(pattern, normalized)
        if match:
            return match.group("owner")
    return None


def detect_github_owner(repo_root: Path, fallback_text: str = "") -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
        )
        owner = parse_github_owner(result.stdout)
        if owner:
            return owner
    except OSError:
        pass

    match = re.search(r'^SET\s+"PUBLISHER=([^"]+)"\s*$', fallback_text, re.MULTILINE)
    if match and match.group(1) != "GITHUB_OWNER":
        return match.group(1)
    return "GITHUB_OWNER"


def detect_python_package_name(repo_root: Path) -> str:
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
        match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', read_text(setup_path))
        if match:
            return match.group(1).replace("-", "_")
    return "PYTHON_PROJ_NAME"


def normalize_posix(path_value: str) -> str:
    return path_value.replace("\\", "/")


def derive_icon_paths(repo_root: Path, construct_icon: str, project_name: str) -> dict[str, str]:
    existing_icons: dict[str, str] = {}
    launcher_path = repo_root / NOTEBOOK_LAUNCHER_PATH
    if launcher_path.exists():
        try:
            launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
            platforms = launcher.get("menu_items", [{}])[0].get("platforms", {})
            for platform_key, placeholder in (
                ("win", "ICON_ICO_IMAGE_PATH"),
                ("linux", "ICON_IMAGE_PATH"),
                ("osx", "ICON_ICNS_IMAGE_PATH"),
            ):
                icon = platforms.get(platform_key, {}).get("icon")
                if isinstance(icon, str) and icon.strip():
                    existing_icons[placeholder] = normalize_posix(icon.strip())
        except (json.JSONDecodeError, IndexError, AttributeError):
            pass

    normalized_icon = normalize_posix(construct_icon)
    stem = normalized_icon.rsplit(".", 1)[0] if normalized_icon else f"app/logo/{project_name}_logo"
    return {
        "ICON_IMAGE_PATH": existing_icons.get("ICON_IMAGE_PATH", normalized_icon or f"{stem}.png"),
        "ICON_ICO_IMAGE_PATH": existing_icons.get("ICON_ICO_IMAGE_PATH", f"{stem}.ico"),
        "ICON_ICNS_IMAGE_PATH": existing_icons.get("ICON_ICNS_IMAGE_PATH", f"{stem}.icns"),
    }


def load_replacements(repo_root: Path) -> dict[str, str]:
    construct_text = read_text(repo_root / CONSTRUCT_PATH)
    construct_name = extract_yaml_value(construct_text, "name") or "UNDERSCORED_PROJECT_NAME"
    project_name = extract_project_folder(construct_text) or construct_name
    version = extract_yaml_value(construct_text, "version") or "VERSION_NUMBER"
    construct_icon = extract_yaml_value(construct_text, "icon_image") or ""

    if project_name == "PROJECT_NAME" or construct_name == "UNDERSCORED_PROJECT_NAME":
        github_owner = "GITHUB_OWNER"
    else:
        post_install_path = repo_root / POST_INSTALL_BAT_PATH
        post_install_text = read_text(post_install_path) if post_install_path.exists() else ""
        github_owner = detect_github_owner(repo_root, post_install_text)

    replacements = {
        "PROJECT_NAME": project_name,
        "UNDERSCORED_PROJECT_NAME": construct_name,
        "VERSION_NUMBER": version,
        "GITHUB_OWNER": github_owner,
        "PYTHON_PROJ_NAME": detect_python_package_name(repo_root),
    }
    replacements.update(derive_icon_paths(repo_root, construct_icon, project_name))
    return replacements


def render_template(text: str, replacements: dict[str, str], newline: str) -> str:
    rendered = text.replace("\r\n", "\n")
    for key in sorted(replacements, key=len, reverse=True):
        rendered = rendered.replace(key, replacements[key])
    return rendered.replace("\n", newline)


def copy_template(
    repo_root: Path,
    resource_relative_path: str,
    destination_relative_path: Path,
    replacements: dict[str, str] | None = None,
) -> bool:
    source_path = RESOURCE_ROOT / resource_relative_path
    if not source_path.exists():
        raise ValueError(f"Missing migration resource: {source_path}")

    destination_path = repo_root / destination_relative_path
    source_text = read_text(source_path)
    if destination_path.exists():
        existing_text = read_text(destination_path)
        newline = detect_newline(existing_text)
    else:
        existing_text = ""
        newline = detect_newline(source_text)

    rendered_text = render_template(source_text, replacements or {}, newline)
    if existing_text == rendered_text:
        return False

    write_text(destination_path, rendered_text)
    print(f"Updated {destination_relative_path}")
    return True


def _dependency_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    value = stripped[2:].split("#", 1)[0].strip()
    return re.split(r"[<>=!~\s]", value, maxsplit=1)[0]


def update_environment(repo_root: Path) -> bool:
    path = repo_root / ENVIRONMENT_PATH
    if not path.exists():
        print(f"Skipping missing file: {ENVIRONMENT_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    lines = original_text.splitlines()
    if not lines:
        raise ValueError(f"Unexpected empty file: {ENVIRONMENT_PATH}")

    changed = False
    filtered_lines: list[str] = []
    for line in lines:
        if _dependency_name(line) == "pip-system-certs":
            changed = True
            continue
        filtered_lines.append(line)
    lines = filtered_lines

    dependency_indices: dict[str, int] = {}
    for index, line in enumerate(lines):
        name = _dependency_name(line)
        if name:
            dependency_indices.setdefault(name, index)

    pip_index = dependency_indices.get("pip")
    if pip_index is None:
        raise ValueError(f"Unable to find pip dependency in {ENVIRONMENT_PATH}")

    indent_match = re.match(r"^(\s*)-", lines[pip_index])
    indent = indent_match.group(1) if indent_match else "  "
    desired_pip = f"{indent}- pip>=24.2"
    if lines[pip_index] != desired_pip:
        lines[pip_index] = desired_pip
        changed = True

    dependency_indices = {
        name: index
        for index, line in enumerate(lines)
        if (name := _dependency_name(line)) is not None
    }
    desired = (
        ("setuptools", f"{indent}- setuptools"),
        ("wheel", f"{indent}- wheel"),
        ("menuinst", f"{indent}- menuinst>=2"),
        ("truststore", f"{indent}- truststore>=0.10.4"),
        ("certifi", f"{indent}- certifi"),
    )
    missing = [line for name, line in desired if name not in dependency_indices]
    if missing:
        insert_index = dependency_indices.get("jupyterlab", len(lines))
        lines[insert_index:insert_index] = missing
        changed = True

    if not changed:
        return False

    updated_text = newline.join(lines) + newline
    write_text(path, updated_text)
    print(f"Updated {ENVIRONMENT_PATH}")
    return True


def update_construct(repo_root: Path, replacements: dict[str, str]) -> bool:
    path = repo_root / CONSTRUCT_PATH
    if not path.exists():
        print(f"Skipping missing file: {CONSTRUCT_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    mapping_line = f"- app/python_scripts/launch_jupyter.py: {replacements['PROJECT_NAME']}/launch_jupyter.py"
    if mapping_line in original_text:
        return False

    lines = original_text.splitlines(keepends=True)
    insert_index = None
    for marker in (
        "app/python_scripts/include_path.py:",
        "app/python_scripts/hide_code_cells.py:",
        "notebooks/notebook_latest_versions.yaml:",
    ):
        for index, line in enumerate(lines):
            if marker in line:
                insert_index = index + 1
                break
        if insert_index is not None:
            break

    if insert_index is None:
        for index, line in enumerate(lines):
            if line.startswith("post_install:"):
                insert_index = index
                break
    if insert_index is None:
        raise ValueError(f"Unable to find extra_files insertion point in {CONSTRUCT_PATH}")

    lines.insert(insert_index, mapping_line + newline)
    write_text(path, "".join(lines))
    print(f"Updated {CONSTRUCT_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context
    replacements = load_replacements(repo_root)

    changed_any = False
    changed_any = update_environment(repo_root) or changed_any
    changed_any = update_construct(repo_root, replacements) or changed_any
    changed_any = copy_template(repo_root, "app/bash_bat_scripts/post_install.bat", POST_INSTALL_BAT_PATH, replacements) or changed_any
    changed_any = copy_template(repo_root, "app/bash_bat_scripts/post_install.sh", POST_INSTALL_SH_PATH, replacements) or changed_any
    changed_any = copy_template(repo_root, "app/menuinst/notebook_launcher.json", NOTEBOOK_LAUNCHER_PATH, replacements) or changed_any
    changed_any = copy_template(repo_root, "app/python_scripts/launch_jupyter.py", LAUNCH_JUPYTER_PATH, replacements) or changed_any
    changed_any = copy_template(repo_root, ".tools/docs/troubleshooting.md", TROUBLESHOOTING_PATH, replacements) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

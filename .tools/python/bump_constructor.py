import sys
from pathlib import Path
import yaml
import re

DEBUG_PREFIX = "[bump_constructor]"


def debug(message: str) -> None:
    print(f"{DEBUG_PREFIX} {message}")


def load_construct(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def save_construct_surgical(path: Path, data: dict, original_text: str) -> str:
    """Update only extra_files section while preserving the rest of the file structure."""
    # Convert updated extra_files back to YAML string
    extra_files = data.get("extra_files", [])
    extra_files_yaml = "extra_files:\n"
    for item in extra_files:
        if isinstance(item, dict):
            for src, dst in item.items():
                extra_files_yaml += f"- {src}: {dst}\n"
        else:
            extra_files_yaml += f"- {item}\n"
    
    # Find and replace the extra_files section in the original text
    pattern = r"extra_files:\n((?:- .+\n)*)"
    if re.search(pattern, original_text):
        updated = re.sub(pattern, extra_files_yaml, original_text)
        path.write_text(updated, encoding="utf-8")
    else:
        # Fallback: if no extra_files found, use standard dump
        path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def extract_project_folder(extra_files: list) -> str:
    """Extract the project folder name from existing mappings (e.g., from Welcome.ipynb dest)."""
    for item in extra_files:
        if isinstance(item, dict):
            for src, dst in item.items():
                if "Welcome.ipynb" in src or "Welcome.ipynb" in dst:
                    # Extract first folder from destination path
                    parts = dst.split("/")
                    if parts:
                        return parts[0]
    # Fallback
    return "CellTracksColab"

def ensure_requirements_in_extra_files(construct_data: dict):
    extra_files = construct_data.get("extra_files")
    if extra_files is None:
        extra_files = []
        construct_data["extra_files"] = extra_files

    # Check if requirements.txt is included, if not, add it
    requirements_included = any(
        isinstance(item, dict) and "requirements.txt" in item for item in extra_files
    )
    if not requirements_included:
        extra_files.append({"requirements.txt": "CellTracksColab/requirements.txt"})
        debug("Added mapping for requirements.txt")
    else:
        debug("requirements.txt mapping already present")

    gpu_requirements_included = any(
        isinstance(item, dict) and "requirements_gpu.txt" in item for item in extra_files
    )
    if not gpu_requirements_included and Path("requirements_gpu.txt").exists():
        extra_files.append({"requirements_gpu.txt": "CellTracksColab/requirements_gpu.txt"})
        debug("Added mapping for requirements_gpu.txt")
    elif Path("requirements_gpu.txt").exists():
        debug("requirements_gpu.txt mapping already present")
    else:
        debug("requirements_gpu.txt not found, skipping GPU requirements mapping")
    
    # Check if requirements-linux.txt exists and is included, if not, add it
    if Path("requirements-linux.txt").exists():
        linux_included = any(
            isinstance(item, dict) and "requirements-linux.txt" in item for item in extra_files
        )
        if not linux_included:
            extra_files.append({"requirements-linux.txt": "CellTracksColab/requirements-linux.txt"})
            debug("Added mapping for requirements-linux.txt")
        else:
            debug("requirements-linux.txt mapping already present")
    else:
        debug("requirements-linux.txt not found, skipping Linux requirements mapping")
    # Check if requirements-windows.txt exists and is included, if not, add it
    if Path("requirements-windows.txt").exists():
        windows_included = any(
            isinstance(item, dict) and "requirements-windows.txt" in item for item in extra_files
        )
        if not windows_included:
            extra_files.append({"requirements-windows.txt": "CellTracksColab/requirements-windows.txt"})
            debug("Added mapping for requirements-windows.txt")
        else:
            debug("requirements-windows.txt mapping already present")
    else:
        debug("requirements-windows.txt not found, skipping Windows requirements mapping")
    
    # Check if requirements-macos.txt exists and is included, if not, add it
    if Path("requirements-macos.txt").exists():
        macos_included = any(
            isinstance(item, dict) and "requirements-macos.txt" in item for item in extra_files
        )
        if not macos_included:
            extra_files.append({"requirements-macos.txt": "CellTracksColab/requirements-macos.txt"})
            debug("Added mapping for requirements-macos.txt")
        else:
            debug("requirements-macos.txt mapping already present")
    else:
        debug("requirements-macos.txt not found, skipping macOS requirements mapping")

    # Update the construct data with the modified extra_files
    construct_data["extra_files"] = extra_files

def ensure_extra_files(construct_data: dict, notebooks_root: Path, src_root: Path) -> int:
    extra_files = construct_data.get("extra_files")
    if extra_files is None:
        extra_files = []
        construct_data["extra_files"] = extra_files

    # Extract the project folder name from existing mappings
    project_folder = extract_project_folder(extra_files)
    debug(f"Using project folder: {project_folder}")

    # Normalize existing entries into a dict for quick lookup
    existing_sources = set()
    existing_dests = set()
    normalized_items = []

    for item in extra_files:
        # Items can be either dicts (k: v) or strings with mapping? Assume dicts per example
        if isinstance(item, dict):
            for src, dst in item.items():
                existing_sources.add(str(src))
                existing_dests.add(str(dst))
                normalized_items.append({str(src): str(dst)})
        else:
            # If strings are present, keep them
            normalized_items.append(item)

    ntbk_added = 0
    repo_root = Path(".").resolve()

    # For safety, remove any existing notebooks/*.ipynb entries to avoid duplicates, we will re-add them with correct paths
    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src).startswith("notebooks/") and src.endswith(".ipynb") for src in item.keys()))]

    # Include all notebooks under notebooks/ directory
    for ipynb in notebooks_root.rglob("*.ipynb"):
        rel = ipynb.relative_to(repo_root).as_posix()
        if not rel.startswith("notebooks/"):
            continue
        src = rel
        dst = f"{project_folder}/{rel}"

        # if src in existing_sources or dst in existing_dests:
        #     continue

        normalized_items.append({src: dst})
        ntbk_added += 1
        debug(f"Included notebook: {src} -> {dst}")

    # First get the name of the package from setup.py
    setup_path = repo_root / "setup.py"
    project_name = "CellTracksColab"
    if setup_path.exists():
        setup_text = setup_path.read_text(encoding="utf-8")
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', setup_text)
        if name_match:
            project_name = name_match.group(1)
        debug(f"Detected setup.py package name: {project_name}")
    else:
        debug("setup.py not found at repository root")

    # For safety, remove any existing src/ entries to avoid duplicates, we will re-add them with correct paths
    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src).startswith("src/") for src in item.keys()))]

    # Include all the Python scripts under src/ directory
    src_added = 0
    only_init_files = True
    included_src_flag = False
    for py_file in src_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        
        # Avoid any python files that are not in the src/ directory 
        if not rel.startswith("src/"):
            continue

        # Control if we only have __init__.py files
        if not py_file.name == "__init__.py":
            only_init_files = False
        
        src = rel
        dst = f"{project_folder}/{rel}"

        # if src in existing_sources or dst in existing_dests:
        #     continue

        normalized_items.append({src: dst})
        included_src_flag = True
        src_added += 1
        debug(f"Included source file: {src} -> {dst}")

    if only_init_files:
        debug("Only __init__.py files found under src/, skipping setup.py and src change marker packaging")
        included_src_flag = False

    # For safety also remove setup.py and src_change.yaml if they exist in the extra_files to avoid duplicates, we will re-add them with correct paths if src/ is included
    normalized_items = [item for item in normalized_items if not (isinstance(item, dict) and any(str(src) in ["setup.py", ".tools/meta/src_change.yaml"] for src in item.keys()))]  

    if included_src_flag:
        # Also include the setup.py file at the root if not already included
        setup_src = "setup.py"
        setup_dst = f"{project_folder}/setup.py"

        normalized_items.append({setup_src: setup_dst})
        src_added += 1
        debug(f"Included setup.py: {setup_src} -> {setup_dst}")

        # Include the external code change tracking file if it exists
        src_change_file = ".tools/meta/src_change.yaml"
        if Path(src_change_file).exists():
            src_change_dst = f"{project_folder}/src_change.yaml"
            if src_change_file not in existing_sources and src_change_dst not in existing_dests:
                normalized_items.append({src_change_file: src_change_dst})
                src_added += 1
                debug(f"Included src change marker: {src_change_file} -> {src_change_dst}")
            else:
                debug("src change marker mapping already present")
        else:
            debug("src change marker not found, skipping")
    else:
        # If the user has removed the src/directory, remove the generated setup.py and src_change.yaml
        debug("No non-__init__.py files found under src/, skipping setup.py and src change marker packaging")
        setup_dst = Path(project_folder) / "setup.py"
        if setup_dst.exists():
            setup_dst.unlink()
        src_change_dst = Path(project_folder) / "src_change.yaml"
        if src_change_dst.exists():
            src_change_dst.unlink()

    # Optionally sort entries (dicts by their single key) for determinism
    def sort_key(item):
        if isinstance(item, dict):
            # single-key dict
            k = next(iter(item.keys()))
            return (0, k)
        return (1, str(item))

    normalized_items.sort(key=sort_key)
    construct_data["extra_files"] = normalized_items
    debug(f"Final extra_files entries: {len(normalized_items)}")
    for item in normalized_items:
        debug(f"extra_files entry: {item}")

    return ntbk_added, src_added


def main():
    construct_path = Path("construct.yaml") if len(sys.argv) < 2 else Path(sys.argv[1])
    notebooks_root = Path("notebooks").resolve()
    src_root = Path("src").resolve()

    if not construct_path.exists():
        print(f"construct.yaml not found at: {construct_path}", file=sys.stderr)
        sys.exit(1)

    # Read original text to preserve formatting
    original_text = construct_path.read_text(encoding="utf-8")
    
    data = load_construct(construct_path)
    # Check if the requirements files are included in extra_files, if not, add them
    ensure_requirements_in_extra_files(data)

    # Update extra_files with notebooks and src files
    ntbk_added, src_added = ensure_extra_files(data, notebooks_root, src_root)
    
    # Write using surgical update to preserve platform-specific scripts
    save_construct_surgical(construct_path, data, original_text)
    print(f"Added {ntbk_added} notebook(s) and {src_added} source file(s) to extra_files.")

if __name__ == "__main__":
    main()

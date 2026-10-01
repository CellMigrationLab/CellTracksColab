#!/usr/bin/env python3
"""Merge requirements files under a directory (e.g. notebooks/*) into a single
root requirements file.

The script now supports both classic requirements.txt files and the new
requirements.yaml format used by notebooks. For YAML files, it merges the
`python_version` (picking the newest) and the `dependencies` list while
discarding per-notebook descriptions.

It also recognizes optional per-OS fields (`windows_dependencies`,
`linux_dependencies`, `macos_dependencies`) and emits dedicated requirements
outputs for each platform containing only the entries from their OS-specific
lists, in addition to the common file.

Alongside the common requirements file, it emits a GPU-targeted variant. The
GPU file rewrites `tensorflow` to `tensorflow[and-cuda]` and adds the PyTorch
CUDA wheel index when `torch`, `torchvision`, or `torchaudio` are present.

It resolves simple version conflicts by choosing the newest version seen for a
given package (and emits a pinned `pkg==<newest>`). It still preserves
non-package lines (pip options, VCS/URLs) and supports simple `-r` includes.

It also automatically adds ipywidgets if missing, with a version determined by
Python, JupyterLab, and matplotlib versions from environment.yaml.
"""
from pathlib import Path
import argparse
import sys
from collections import OrderedDict
from datetime import datetime
import re
import yaml


# Regex to capture name, optional extras, operator and version, and optional markers
pkg_re = re.compile(r"^(?P<name>[A-Za-z0-9_.+-]+)(?P<extras>\[[^\]]+\])?\s*(?P<op>==|>=|<=|~=|!=|>|<)\s*(?P<ver>[^;\s]+)(?P<marker>\s*;.*)?$")

OS_FIELD_MAP = {
    "windows": "windows_dependencies",
    "linux": "linux_dependencies",
    "macos": "macos_dependencies",
}
OS_DISPLAY_NAMES = {
    "windows": "Windows",
    "linux": "Linux",
    "macos": "macOS",
}
GPU_DISPLAY_NAME = "NVIDIA GPU"
GPU_TORCH_EXTRA_INDEX_URL = "--extra-index-url https://download.pytorch.org/whl/cu128"
TORCH_GPU_PACKAGES = {"torch", "torchvision", "torchaudio"}
PLAIN_PKG_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.+-]+)(?P<extras>\[[^\]]+\])?(?P<marker>\s*;.*)?$"
)


class RequirementContext:
    def __init__(self):
        self.pkg_order: list[str] = []
        self.pkgs: dict[str, dict] = {}
        self.other_lines: OrderedDict[str, None] = OrderedDict()


def normalize_line(line: str) -> str:
    # Remove inline comments and surrounding whitespace.
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


try:
    # Prefer packaging.version for robust version comparison
    from packaging.version import Version, InvalidVersion

    def parse_version(v: str):
        try:
            return Version(v)
        except InvalidVersion:
            return v

    def is_version_greater(a, b):
        try:
            return Version(a) > Version(b)
        except Exception:
            return str(a) > str(b)

except Exception:
    # Fallback: simple numeric-aware comparator
    def _split_parts(v: str):
        parts = re.split(r"[.\-+_]|(?=[a-zA-Z])", v)
        key = []
        for p in parts:
            if not p:
                continue
            if p.isdigit():
                key.append((0, int(p)))
            else:
                key.append((1, p))
        return tuple(key)

    def parse_version(v: str):
        return _split_parts(v)

    def is_version_greater(a, b):
        return parse_version(a) > parse_version(b)


def merge_python_version(filepath: str | None, current: str | None, candidate: str | None):
    """Pick the newest python_version string, if provided."""
    if not candidate:
        return current
    if current is None:
        return candidate
    if current == candidate:
        return current
    else:
        raise ValueError(f"Conflicting python_version found {current} vs {candidate} when processing {filepath}")


def process_requirement_line(line: str, pkg_order: list, pkgs: dict, other_lines: OrderedDict, base_dir: Path | None = None):
    """Handle a single requirement line (shared between txt and yaml inputs)."""
    if not line:
        return

    parts = line.split(maxsplit=1)
    if parts[0] in ("-r", "--requirement") and len(parts) == 2:
        include_path = Path(parts[1]) if base_dir is None else (base_dir / parts[1])
        include = include_path.resolve()
        if include.exists():
            read_requirements_file(include, pkg_order, pkgs, other_lines)
        return

    # Keep pip option lines (indexes, find-links, etc.) as-is
    if line.startswith("--") or line.startswith("-f ") or line.startswith("-i ") or line.startswith("-e "):
        other_lines.setdefault(line, None)
        return

    # VCS or URL lines (git+, http(s)://, file:) keep as-is
    if any(line.startswith(prefix) for prefix in ("git+", "http://", "https://", "file:", "ssh://")):
        other_lines.setdefault(line, None)
        return

    m = pkg_re.match(line)
    if not m:
        # plain package (unpinned) or marker-only spec
        key = line.split(";", 1)[0].strip().lower()
        if key not in pkgs:
            pkgs[key] = {"name": line, "pinned": False, "ver": None}
            pkg_order.append(key)
        return

    name = m.group("name")
    extras = m.group("extras") or ""
    ver = m.group("ver")
    marker = m.group("marker") or ""

    base = name.lower()
    if base not in pkgs:
        pkgs[base] = {"name": name + (extras or ""), "pinned": True, "ver": ver, "marker": marker}
        pkg_order.append(base)
    else:
        existing = pkgs[base]
        if not existing.get("pinned"):
            pkgs[base] = {"name": name + (extras or ""), "pinned": True, "ver": ver, "marker": marker}
        else:
            try:
                if is_version_greater(ver, existing.get("ver")):
                    pkgs[base] = {"name": name + (extras or ""), "pinned": True, "ver": ver, "marker": marker}
            except Exception:
                if str(ver) > str(existing.get("ver")):
                    pkgs[base] = {"name": name + (extras or ""), "pinned": True, "ver": ver, "marker": marker}


def read_requirements_file(path: Path, pkg_order: list, pkgs: dict, other_lines: OrderedDict):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return

    for raw in text.splitlines():
        line = normalize_line(raw)
        process_requirement_line(line, pkg_order, pkgs, other_lines, base_dir=path.parent)


def read_requirements_yaml(path: Path, pkg_order: list, pkgs: dict, other_lines: OrderedDict, os_contexts: dict[str, RequirementContext]) -> str | None:
    """Read requirements.yaml and return its python_version if present."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    python_version = data.get("python_version")
    dependencies = data.get("dependencies", []) or []

    for dep in dependencies:
        line = normalize_line(str(dep))
        process_requirement_line(line, pkg_order, pkgs, other_lines, base_dir=path.parent)

    for os_name, field in OS_FIELD_MAP.items():
        os_deps = data.get(field) or []
        if not os_deps:
            continue
        ctx = os_contexts.get(os_name)
        if ctx is None:
            continue
        for dep in os_deps:
            line = normalize_line(str(dep))
            process_requirement_line(line, ctx.pkg_order, ctx.pkgs, ctx.other_lines, base_dir=path.parent)

    return python_version


def find_requirements_files(source_dir):
    if not source_dir.exists():
        return []

    patterns = ["requirements.yaml", "requirements.yml", "requirements*.txt"]

    files = []
    for pattern in patterns:
        files.extend(source_dir.rglob(pattern))
    return sorted({f.resolve() for f in files})


def resolve_ipywidgets_version(py_version, jl_version):
    """Resolve ipywidgets version based on Python, JupyterLab versions.
    
    Returns a version spec string like ">=7.0.0,<8.0.0" or a pinned version.
    """
    
    # Simple heuristic: ipywidgets 8.x for modern stacks, 7.x for older
    try:
        py_major, py_minor = map(int, py_version.split(".")[:2])
        jl_major = int(jl_version.split(".")[0])
        
        if jl_major >= 4 and py_major >= 3 and py_minor >= 9:
            return "8.1.7"  # Latest stable for modern Python + JupyterLab 4
        elif py_major >= 3 and py_minor >= 7:
            return "8.1.6"  # Good for Python 3.7+
        else:
            return "7.7.2"  # Fallback for older environments
    except Exception:
        return ">=7.0.0"  # Safe fallback


def determine_jupyterlab_version(pkgs: dict, default: str = "4.4.0") -> str:
    jl_info = pkgs.get("jupyterlab")
    if jl_info and jl_info.get("pinned") and jl_info.get("ver"):
        return jl_info["ver"]
    return default


def ensure_jl_hidecode(ctx: RequirementContext):
    if "jl-hidecode" in ctx.pkgs:
        return
    ctx.pkgs["jl-hidecode"] = {
        "name": "jl-hidecode",
        "pinned": True,
        "ver": "0.0.1",
        "marker": "",
    }
    ctx.pkg_order.append("jl-hidecode")
    print("Added jl-hidecode==0.0.1 (enforced)")


def ensure_ipywidgets(ctx: RequirementContext, python_version: str | None, jl_ver: str):
    if "ipywidgets" in ctx.pkgs:
        return
    iw_version = resolve_ipywidgets_version(python_version, jl_ver)
    ctx.pkgs["ipywidgets"] = {
        "name": "ipywidgets",
        "pinned": True,
        "ver": iw_version,
        "marker": "",
    }
    ctx.pkg_order.append("ipywidgets")
    print(f"Added ipywidgets=={iw_version} (resolved)")


def build_final_lines(ctx: RequirementContext, sort: bool) -> list[str]:
    final_lines: list[str] = []
    for ol in ctx.other_lines.keys():
        final_lines.append(ol)

    pkg_keys = sorted(ctx.pkg_order, key=str.casefold) if sort else ctx.pkg_order

    for key in pkg_keys:
        info = ctx.pkgs.get(key)
        if not info:
            continue
        if info.get("pinned") and info.get("ver"):
            final_lines.append(f"{info['name']}=={info['ver']}{info.get('marker','')}")
        else:
            final_lines.append(info["name"])
    return final_lines


def get_os_output_path(output: Path, os_name: str) -> Path:
    suffix = output.suffix
    stem = output.stem
    return output.with_name(f"{stem}-{os_name}{suffix}")


def get_gpu_output_path(output: Path) -> Path:
    suffix = output.suffix
    stem = output.stem
    return output.with_name(f"{stem}_gpu{suffix}")


def merge_requirement_extras(extras: str, extra_name: str) -> str:
    if not extras:
        return f"[{extra_name}]"

    extras_content = extras[1:-1]
    parts = [part.strip() for part in extras_content.split(",") if part.strip()]
    if extra_name not in parts:
        parts.append(extra_name)
    return "[" + ",".join(parts) + "]"


def rewrite_tensorflow_for_gpu(line: str) -> str:
    match = pkg_re.match(line)
    if match:
        name = match.group("name")
        if name.lower() != "tensorflow":
            return line
        extras = merge_requirement_extras(match.group("extras") or "", "and-cuda")
        marker = match.group("marker") or ""
        return f"{name}{extras}{match.group('op')}{match.group('ver')}{marker}"

    match = PLAIN_PKG_RE.match(line)
    if match and match.group("name").lower() == "tensorflow":
        extras = merge_requirement_extras(match.group("extras") or "", "and-cuda")
        marker = match.group("marker") or ""
        return f"{match.group('name')}{extras}{marker}"

    return line


def build_gpu_lines(final_lines: list[str]) -> list[str]:
    gpu_lines: list[str] = []
    has_torch_package = False
    has_torch_index = False

    for line in final_lines:
        if "download.pytorch.org/whl" in line:
            has_torch_index = True

        match = pkg_re.match(line) or PLAIN_PKG_RE.match(line)
        if match:
            package_name = match.group("name").lower()
            if package_name == "tensorflow":
                line = rewrite_tensorflow_for_gpu(line)
            if package_name in TORCH_GPU_PACKAGES:
                has_torch_package = True

        gpu_lines.append(line)

    if has_torch_package and not has_torch_index:
        insertion_index = 0
        while insertion_index < len(gpu_lines) and gpu_lines[insertion_index].startswith("-"):
            insertion_index += 1
        gpu_lines.insert(insertion_index, GPU_TORCH_EXTRA_INDEX_URL)

    return gpu_lines


def write_requirements_output(
    path: Path,
    files: list[Path],
    final_lines: list[str],
    target_os: str | None = None,
    target_accelerator: str | None = None,
):
    header_lines = [
        f"# Generated by .tools/python/merge_requirements.py on {datetime.utcnow().isoformat()}Z",
        f"# Source files: {', '.join(str(p) for p in files)}",
    ]
    if target_os:
        header_lines.append(f"# Target OS: {OS_DISPLAY_NAMES.get(target_os, target_os.capitalize())}")
    if target_accelerator:
        header_lines.append(f"# Target accelerator: {target_accelerator}")
    header_lines.append("# ---")

    content = "\n".join(header_lines) + "\n"
    if final_lines:
        content += "\n".join(final_lines)
    if not content.endswith("\n"):
        content += "\n"

    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple requirements files (txt or yaml) into one.")
    parser.add_argument("--source-dir", default="notebooks", help="Directory to scan for requirements files")
    parser.add_argument("--output", default="requirements.txt", help="Output requirements file to write")
    parser.add_argument("--sort", action="store_true", help="Sort final requirements alphabetically")
    args = parser.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output)

    discovered_files = find_requirements_files(source)

    skip_outputs = {output.resolve()}
    skip_outputs.add(get_gpu_output_path(output).resolve())
    for os_name in OS_FIELD_MAP:
        skip_outputs.add(get_os_output_path(output, os_name).resolve())

    files = [f for f in discovered_files if f.resolve() not in skip_outputs]
    if not files:
        print(f"No requirements files found under {source}, writing empty {output}")
        output.write_text("# Generated requirements (none found)\n", encoding="utf-8")
        gpu_output = get_gpu_output_path(output)
        gpu_output.write_text("# Generated GPU requirements (none found)\n", encoding="utf-8")
        return 0

    base_ctx = RequirementContext()
    os_contexts = {name: RequirementContext() for name in OS_FIELD_MAP}
    python_version = None

    for f in files:
        if f.suffix.lower() in {".yaml", ".yml"}:
            candidate_py = read_requirements_yaml(f, base_ctx.pkg_order, base_ctx.pkgs, base_ctx.other_lines, os_contexts)
            python_version = merge_python_version(f, python_version, candidate_py)
        else:
            read_requirements_file(f, base_ctx.pkg_order, base_ctx.pkgs, base_ctx.other_lines)

    ensure_jl_hidecode(base_ctx)
    jl_ver = determine_jupyterlab_version(base_ctx.pkgs)
    ensure_ipywidgets(base_ctx, python_version, jl_ver)

    base_final_lines = build_final_lines(base_ctx, args.sort)
    write_requirements_output(output, files, base_final_lines)
    print(f"Wrote {output} with {len(base_final_lines)} entries (from {len(files)} files)")

    gpu_output = get_gpu_output_path(output)
    gpu_final_lines = build_gpu_lines(base_final_lines)
    write_requirements_output(
        gpu_output,
        files,
        gpu_final_lines,
        target_accelerator=GPU_DISPLAY_NAME,
    )
    print(f"Wrote {gpu_output} with {len(gpu_final_lines)} entries (from {len(files)} files, target {GPU_DISPLAY_NAME})")

    for os_name, ctx in os_contexts.items():
        if not ctx.pkg_order and not ctx.other_lines:
            continue
        os_final_lines = build_final_lines(ctx, args.sort)
        os_output = get_os_output_path(output, os_name)
        write_requirements_output(os_output, files, os_final_lines, target_os=os_name)
        print(f"Wrote {os_output} with {len(os_final_lines)} entries (from {len(files)} files, target {os_name})")

    # Create the environment.yaml if not present
    env_path = Path("environment.yaml")
    if not env_path.exists():
        print(f"{env_path} not found, please be sure to create one.")
        raise FileNotFoundError(f"{env_path} not found.")
    else:
        data = yaml.safe_load(env_path.read_text(encoding="utf-8"))
        dependencies = data.get("dependencies", [])
        python_updated = False
        jl_updated = False
        for i, dep in enumerate(dependencies):
            if not isinstance(dep, str):
                continue
            if python_version and dep.startswith("python="):
                dependencies[i] = f"python={python_version}"
                python_updated = True
            if dep.startswith("jupyterlab="):
                dependencies[i] = f"jupyterlab={jl_ver}"
                jl_updated = True
        data["dependencies"] = dependencies
        env_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        if python_updated or jl_updated:
            parts = []
            if python_updated:
                parts.append(f"python={python_version}")
            if jl_updated:
                parts.append(f"jupyterlab={jl_ver}")
            print(f"Updated {env_path} with {', '.join(parts)}")
        else:
            print(f"No python/jupyterlab entries updated in {env_path}")

    # Replace on setup.py the Python version
    setup_path = Path("setup.py")
    if setup_path.exists() and python_version:
        setup_text = setup_path.read_text(encoding="utf-8")
        pattern = r'python_requires\s*=\s*["\'][><=\.0-9 ]+(PYTHON\_VERSION)?["\']'
        new_setup_text = re.sub(pattern, f'python_requires=">={python_version}"', setup_text)
        setup_path.write_text(new_setup_text, encoding="utf-8")
        print(f"Updated {setup_path} with python_requires>={python_version}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

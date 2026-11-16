#!/usr/bin/env python3
"""Merge requirements files under a directory (e.g. notebooks/*) into a single
root requirements file.

The script supports both classic requirements.txt files and the
requirements.yaml format used by notebooks. For YAML files, it merges the
`python_version` (every notebook must agree) and the `dependencies` list while
discarding per-notebook descriptions.

It also recognizes optional per-OS fields (`windows_dependencies`,
`linux_dependencies`, `macos_dependencies`) and emits dedicated requirements
outputs for each platform containing only the entries from their OS-specific
lists, in addition to the common file.

Alongside the common requirements file, it emits a GPU-targeted variant. The
GPU file rewrites `tensorflow` to `tensorflow[and-cuda]` on Linux only (TensorFlow's CUDA
wheels exist only for Linux; other systems keep plain `tensorflow`) and adds the
PyTorch CUDA wheel index when `torch`, `torchvision`, or `torchaudio` are present.

Version requirements are never changed silently (template 0.1.31, CellMigrationLab/LabConstrictor#48):
  * the requirements of a package are kept as written: `numpy>=1.24` stays `numpy>=1.24`, and the
    requirements of several notebooks are combined (`numpy<2,>=1.24`); the lock step then finds the versions;
  * when notebooks pin a package to different versions (`numpy==1.26.4` and `numpy==2.0.0`), or a pin does not
    meet another notebook's requirement, nothing is written and the script fails, naming every notebook and its
    requirement. The developer makes the notebooks agree, or decides explicitly in requirements_overrides.yaml
    (next to the output), with the reason; the decision is written at the top of every generated file;
  * requirements with different environment markers (`; sys_platform == "win32"`) are kept as separate lines;
  * package names are compared as pip compares them (`NumPy`, `numpy` and `num_py` are one package).
`python merge_requirements.py --check` only looks for conflicts: it writes nothing (the release runs it).

It preserves non-package lines (pip options, VCS/URLs) and supports simple `-r` includes.

It also automatically adds ipywidgets if missing, with a version determined by
Python, JupyterLab, and matplotlib versions from environment.yaml.
"""
import argparse
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_31.md"
OVERRIDES_FILE = "requirements_overrides.yaml"

# Regex to capture name, optional extras, operator and version, and optional markers (used for the GPU rewrite of final lines)
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
ADDED_BY_SCRIPT = "added by merge_requirements.py"


@dataclass
class Entry:
    """One requirement line of one file, as written."""

    source: str
    line: str
    name: str
    extras: tuple[str, ...]
    spec: str
    marker: str


@dataclass
class Override:
    """A decision of requirements_overrides.yaml: every notebook gets `use` for this package."""

    name: str
    use: str
    reason: str
    used: bool = False


@dataclass
class Resolved:
    """The line written for one package (and marker), and how it was decided."""

    entries: list[Entry]
    spec: str
    extras: list[str]
    override: Override | None = None

    @property
    def name(self) -> str:
        return self.entries[0].name

    @property
    def marker(self) -> str:
        return self.entries[0].marker

    def line(self) -> str:
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        marker = f" ; {self.marker}" if self.marker else ""
        return f"{self.name}{extras}{self.spec}{marker}"


@dataclass
class RequirementContext:
    groups: OrderedDict[tuple[str, str], list[Entry]] = field(default_factory=OrderedDict)
    other_lines: OrderedDict[str, None] = field(default_factory=OrderedDict)
    errors: list[str] = field(default_factory=list)


class Conflict(Exception):
    """Requirements of one package that no single version meets, or that the script will not choose between."""

    def __init__(self, package: str, reason: str, entries: list[Entry], where: str = ""):
        super().__init__(reason)
        self.package = package
        self.reason = reason
        self.entries = entries
        self.where = where


def normalize_line(line: str) -> str:
    # Remove inline comments and surrounding whitespace.
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def merge_python_version(filepath: str | Path | None, current: str | None, candidate: str | None):
    """Every notebook that names a python_version must name the same one."""
    if not candidate:
        return current
    if current is None:
        return candidate
    if current == candidate:
        return current
    else:
        raise ValueError(f"Conflicting python_version found {current} vs {candidate} when processing {filepath}")


def process_requirement_line(line: str, ctx: RequirementContext, source: str, base_dir: Path | None = None):
    """Handle a single requirement line (shared between txt and yaml inputs)."""
    if not line:
        return

    parts = line.split(maxsplit=1)
    if parts[0] in ("-r", "--requirement") and len(parts) == 2:
        include_path = Path(parts[1]) if base_dir is None else (base_dir / parts[1])
        include = include_path.resolve()
        if include.exists():
            read_requirements_file(include, ctx)
        return

    # Keep pip option lines (indexes, find-links, etc.) as-is
    if line.startswith("--") or line.startswith("-f ") or line.startswith("-i ") or line.startswith("-e "):
        ctx.other_lines.setdefault(line, None)
        return

    # VCS or URL lines (git+, http(s)://, file:) keep as-is
    if any(line.startswith(prefix) for prefix in ("git+", "http://", "https://", "file:", "ssh://")):
        ctx.other_lines.setdefault(line, None)
        return

    try:
        requirement = Requirement(line)
    except InvalidRequirement as exc:
        ctx.errors.append(f"{source}: '{line}' is not a valid requirement ({exc})")
        return
    if requirement.url:
        # name @ url: installed from that address, as written
        ctx.other_lines.setdefault(line, None)
        return
    marker = str(requirement.marker) if requirement.marker else ""
    entry = Entry(source, line, requirement.name, tuple(sorted(requirement.extras)), str(requirement.specifier), marker)
    ctx.groups.setdefault((canonicalize_name(requirement.name), marker), []).append(entry)


def read_requirements_file(path: Path, ctx: RequirementContext):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return

    for raw in text.splitlines():
        line = normalize_line(raw)
        process_requirement_line(line, ctx, display_path(path), base_dir=path.parent)


def read_requirements_yaml(path: Path, ctx: RequirementContext, os_contexts: dict[str, RequirementContext]) -> str | None:
    """Read requirements.yaml and return its python_version if present."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    python_version = data.get("python_version")
    dependencies = data.get("dependencies", []) or []
    source = display_path(path)

    for dep in dependencies:
        line = normalize_line(str(dep))
        process_requirement_line(line, ctx, source, base_dir=path.parent)

    for os_name, field_name in OS_FIELD_MAP.items():
        os_deps = data.get(field_name) or []
        if not os_deps:
            continue
        os_ctx = os_contexts.get(os_name)
        if os_ctx is None:
            continue
        for dep in os_deps:
            line = normalize_line(str(dep))
            process_requirement_line(line, os_ctx, f"{source} ({field_name})", base_dir=path.parent)

    return python_version


def find_requirements_files(source_dir):
    if not source_dir.exists():
        return []

    patterns = ["requirements.yaml", "requirements.yml", "requirements*.txt"]

    files = []
    for pattern in patterns:
        files.extend(source_dir.rglob(pattern))
    return sorted({f.resolve() for f in files})


# --------------------------------------------------------------------------------------------------------------------------------
# Deciding the requirement of each package
# --------------------------------------------------------------------------------------------------------------------------------
def load_overrides(path: Path) -> dict[str, Override]:
    """The decisions of requirements_overrides.yaml, by package name as pip compares them (none when the file is absent)."""
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{display_path(path)} is not valid YAML: {exc}") from exc
    entries = data.get("overrides") if isinstance(data, dict) else None
    if entries is None:
        return {}
    if not isinstance(entries, dict):
        raise ValueError(f"{display_path(path)}: `overrides` must map package names to {{use: ..., reason: ...}}")
    overrides: dict[str, Override] = {}
    for name, value in entries.items():
        use = value.get("use") if isinstance(value, dict) else None
        reason = value.get("reason") if isinstance(value, dict) else None
        if not isinstance(use, str) or not use.strip():
            raise ValueError(f"{display_path(path)}: {name} needs `use`, the requirement every notebook gets, for example \"==2.0.0\"")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{display_path(path)}: {name} needs `reason`, why this requirement is right for every notebook that uses {name}")
        try:
            SpecifierSet(use.strip())
        except InvalidSpecifier as exc:
            raise ValueError(f"{display_path(path)}: {name}: `use: {use}` is not a version requirement such as \"==2.0.0\" or \">=1.24,<2\" ({exc})") from exc
        overrides[canonicalize_name(str(name))] = Override(str(name), use.strip(), " ".join(reason.split()))
    return overrides


def _exact_versions(entry: Entry) -> list[str]:
    return [s.version for s in SpecifierSet(entry.spec) if s.operator in ("==", "===") and "*" not in s.version]


def _same_version(a: str, b: str) -> bool:
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def _union_extras(entries: list[Entry]) -> list[str]:
    extras: list[str] = []
    for entry in entries:
        extras.extend(extra for extra in entry.extras if extra not in extras)
    return extras


def resolve_group(package: str, entries: list[Entry], checked_with: list[Entry], override: Override | None, where: str = "") -> Resolved:
    """The requirement written for `entries` (one package, one marker); `checked_with` are other requirements that apply too.

    An override decides. Otherwise the requirements are combined: several different exact pins, or a pin that does not meet another
    requirement, raise Conflict (whatever the order of the files). Without a pin, the combined ranges are written as they are.
    """
    extras = _union_extras(entries)
    if override is not None:
        override.used = True
        return Resolved(entries, override.use, extras, override)
    everything = entries + checked_with
    pins: list[tuple[str, Entry]] = [(version, entry) for entry in everything for version in _exact_versions(entry)]
    distinct: list[str] = []
    for version, _ in pins:
        if not any(_same_version(version, seen) for seen in distinct):
            distinct.append(version)
    if len(distinct) > 1:
        raise Conflict(package, f"pinned to different versions ({', '.join(sorted(distinct))})", everything, where)
    if distinct:
        pin = distinct[0]
        for entry in everything:
            if entry.spec and not SpecifierSet(entry.spec).contains(pin, prereleases=True):
                raise Conflict(package, f"the pin =={pin} does not meet {entry.name}{entry.spec} of {entry.source}", everything, where)
    elif not satisfiable(SpecifierSet(",".join(entry.spec for entry in everything if entry.spec))):
        raise Conflict(package, "no version meets all of these requirements", everything, where)
    # The line keeps the requirements of its own entries; `checked_with` (the common requirements of an OS file) only had to agree.
    own_pins = [version for entry in entries for version in _exact_versions(entry)]
    if own_pins:
        return Resolved(entries, f"=={own_pins[0]}", extras)
    return Resolved(entries, str(SpecifierSet(",".join(entry.spec for entry in entries if entry.spec))), extras)


def satisfiable(specifiers: SpecifierSet) -> bool:
    """Whether some version meets every specifier (`<2,>=2.1` does not).

    Every version a specifier names is tried, and versions just above it (2.1, 2.1.1, 2.1.0.1, 2.1.0.0.1): a range that only versions
    closer than that could meet is reported as unsatisfiable, which stops the merge (an override then decides) instead of guessing.
    """
    if not len(specifiers):
        return True
    candidates: list[str] = []
    for specifier in specifiers:
        version = specifier.version.removesuffix(".*")
        candidates += [version, f"{version}.1", f"{version}.0.1", f"{version}.0.0.1"]
    for candidate in candidates:
        try:
            if specifiers.contains(candidate, prereleases=True):
                return True
        except InvalidVersion:
            continue
    return False


def resolve_context(ctx: RequirementContext, overrides: dict[str, Override], conflicts: list[Conflict],
                    base: RequirementContext | None = None, where: str = "") -> list[Resolved]:
    """Resolve every package of a context; an OS context is also checked against the common requirements that apply on that OS."""
    resolved: list[Resolved] = []
    for (package, marker), entries in ctx.groups.items():
        checked_with: list[Entry] = []
        if base is not None:
            for (base_package, base_marker), base_entries in base.groups.items():
                if base_package == package and base_marker in ("", marker):
                    checked_with.extend(base_entries)
        try:
            resolved.append(resolve_group(package, entries, checked_with, overrides.get(package), where))
        except Conflict as conflict:
            conflicts.append(conflict)
    return resolved


def conflict_report(conflicts: list[Conflict], errors: list[str], overrides_path: Path) -> str:
    lines = ["The notebooks ask for incompatible package versions, so no requirements file was written.", ""]
    for error in errors:
        lines.append(f"  {error}")
    for conflict in sorted(conflicts, key=lambda c: (c.package, c.where)):
        where = f" ({conflict.where})" if conflict.where else ""
        lines.append(f"  {conflict.package}{where}: {conflict.reason}")
        for entry in sorted(conflict.entries, key=lambda e: (e.source, e.line)):
            lines.append(f"    {entry.source}: {entry.line}")
    if conflicts:
        example = sorted(conflicts, key=lambda c: c.package)[0].package
        lines += [
            "",
            f"Make the notebooks agree (edit their requirements.yaml), or decide in {display_path(overrides_path)} which requirement every",
            "notebook gets, and why. It is written at the top of the generated requirements files. For example:",
            "",
            "overrides:",
            f"  {example}:",
            '    use: "==<the version every notebook was checked with>"',
            f'    reason: "<why this version is right for every notebook that uses {example}>"',
        ]
    lines += ["", f"See {DOCS_URL}"]
    return "\n".join(lines)


def override_header(resolved: list[Resolved]) -> list[str]:
    lines = []
    for item in resolved:
        if item.override is None:
            continue
        asked = "; ".join(f"{entry.source}: {entry.line}" for entry in item.entries if entry.source != ADDED_BY_SCRIPT)
        lines.append(f"# Decided in {OVERRIDES_FILE}: {item.line()} (the notebooks ask for: {asked}). Reason: {item.override.reason}")
    return lines


# --------------------------------------------------------------------------------------------------------------------------------
# Packages the app always needs
# --------------------------------------------------------------------------------------------------------------------------------
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


def _has_package(ctx: RequirementContext, package: str) -> bool:
    return any(name == package for name, _ in ctx.groups)


def _add_package(ctx: RequirementContext, name: str, spec: str) -> None:
    ctx.groups[(canonicalize_name(name), "")] = [Entry(ADDED_BY_SCRIPT, f"{name}{spec}", name, (), spec, "")]


def determine_jupyterlab_version(resolved: list[Resolved], default: str = "4.4.0") -> str:
    for item in resolved:
        if canonicalize_name(item.name) == "jupyterlab" and not item.marker and item.spec.startswith("==") and "," not in item.spec:
            return item.spec[2:]
    return default


def ensure_jl_hidecode(ctx: RequirementContext):
    if _has_package(ctx, "jl-hidecode"):
        return
    _add_package(ctx, "jl-hidecode", "==0.0.1")
    print("Added jl-hidecode==0.0.1 (enforced)")


def ensure_ipywidgets(ctx: RequirementContext, python_version: str | None, jl_ver: str) -> bool:
    if _has_package(ctx, "ipywidgets"):
        return False
    iw_version = resolve_ipywidgets_version(python_version, jl_ver)
    spec = iw_version if iw_version[:1] in "<>=!~" else f"=={iw_version}"
    _add_package(ctx, "ipywidgets", spec)
    print(f"Added ipywidgets{spec} (resolved)")
    return True


def build_final_lines(ctx: RequirementContext, resolved: list[Resolved], sort: bool) -> list[str]:
    final_lines: list[str] = list(ctx.other_lines.keys())
    items = sorted(resolved, key=lambda item: item.name.casefold()) if sort else resolved
    final_lines.extend(item.line() for item in items)
    return final_lines


# --------------------------------------------------------------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------------------------------------------------------------
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


# TensorFlow's `and-cuda` extra pulls nvidia-* wheels that exist only for Linux (nvidia-nccl-cu12 has never had a Windows or macOS
# wheel), and the extra itself carries no platform marker. Without a marker the universal lock of requirements_gpu.txt pins them for
# every system, so `pip install` fails on a Windows computer with an NVIDIA driver (the installer picks the GPU file there). TensorFlow
# has had no native Windows GPU build since 2.10, so Windows and macOS keep plain `tensorflow` in the GPU file.
GPU_TENSORFLOW_PLATFORM = 'sys_platform == "linux"'
NON_GPU_TENSORFLOW_PLATFORM = 'sys_platform != "linux"'


def _with_platform(marker: str, platform: str) -> str:
    """Return ` ; <platform>`, combined with the line's own marker (`; <marker>`) when it has one."""
    existing = marker.split(";", 1)[1].strip() if ";" in marker else ""
    return f" ; ({existing}) and {platform}" if existing else f" ; {platform}"


def tensorflow_gpu_lines(line: str) -> list[str]:
    """Return the GPU-file lines for a tensorflow requirement: `[and-cuda]` on Linux, the plain requirement elsewhere.

    A line that is not a tensorflow requirement is returned unchanged (as a one-item list).
    """
    match = pkg_re.match(line)
    if match:
        if match.group("name").lower() != "tensorflow":
            return [line]
        name, extras = match.group("name"), match.group("extras") or ""
        spec = f"{match.group('op')}{match.group('ver')}"
    else:
        match = PLAIN_PKG_RE.match(line)
        if not match or match.group("name").lower() != "tensorflow":
            return [line]
        name, extras, spec = match.group("name"), match.group("extras") or "", ""
    marker = match.group("marker") or ""
    cuda_extras = merge_requirement_extras(extras, "and-cuda")
    return [
        f"{name}{cuda_extras}{spec}{_with_platform(marker, GPU_TENSORFLOW_PLATFORM)}",
        f"{name}{extras}{spec}{_with_platform(marker, NON_GPU_TENSORFLOW_PLATFORM)}",
    ]


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
            if package_name in TORCH_GPU_PACKAGES:
                has_torch_package = True
            if package_name == "tensorflow":
                gpu_lines.extend(tensorflow_gpu_lines(line))
                continue

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
    decisions: list[str] | None = None,
):
    generated = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    header_lines = [
        f"# Generated by .tools/python/merge_requirements.py on {generated}Z",
        f"# Source files: {', '.join(str(p) for p in files)}",
    ]
    if target_os:
        header_lines.append(f"# Target OS: {OS_DISPLAY_NAMES.get(target_os, target_os.capitalize())}")
    if target_accelerator:
        header_lines.append(f"# Target accelerator: {target_accelerator}")
    header_lines.extend(decisions or [])
    header_lines.append("# ---")

    content = "\n".join(header_lines) + "\n"
    if final_lines:
        content += "\n".join(final_lines)
    if not content.endswith("\n"):
        content += "\n"

    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Merge multiple requirements files (txt or yaml) into one.")
    parser.add_argument("--source-dir", default="notebooks", help="Directory to scan for requirements files")
    parser.add_argument("--output", default="requirements.txt", help="Output requirements file to write")
    parser.add_argument("--sort", action="store_true", help="Sort final requirements alphabetically")
    parser.add_argument("--overrides", help=f"Decisions for conflicting requirements (default: {OVERRIDES_FILE} next to the output)")
    parser.add_argument("--check", action="store_true", help="Only check that the notebooks' requirements do not conflict; write nothing")
    args = parser.parse_args(argv)

    source = Path(args.source_dir)
    output = Path(args.output)
    overrides_path = Path(args.overrides) if args.overrides else output.with_name(OVERRIDES_FILE)

    discovered_files = find_requirements_files(source)

    skip_outputs = {output.resolve()}
    skip_outputs.add(get_gpu_output_path(output).resolve())
    for os_name in OS_FIELD_MAP:
        skip_outputs.add(get_os_output_path(output, os_name).resolve())

    files = [f for f in discovered_files if f.resolve() not in skip_outputs]
    if not files:
        if args.check:
            print(f"No requirements files found under {source}: nothing to check.")
            return 0
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
            candidate_py = read_requirements_yaml(f, base_ctx, os_contexts)
            python_version = merge_python_version(f, python_version, candidate_py)
        else:
            read_requirements_file(f, base_ctx)

    try:
        overrides = load_overrides(overrides_path)
    except ValueError as exc:
        print(f"{exc}\nSee {DOCS_URL}", file=sys.stderr)
        return 1

    ensure_jl_hidecode(base_ctx)
    conflicts: list[Conflict] = []
    base_resolved = resolve_context(base_ctx, overrides, conflicts)
    jl_ver = determine_jupyterlab_version(base_resolved)
    if ensure_ipywidgets(base_ctx, python_version, jl_ver):
        key = next(reversed(base_ctx.groups))
        base_resolved.append(resolve_group(key[0], base_ctx.groups[key], [], overrides.get(key[0])))
    os_resolved = {
        os_name: resolve_context(ctx, overrides, conflicts, base=base_ctx, where=f"{OS_FIELD_MAP[os_name]}, with the common dependencies")
        for os_name, ctx in os_contexts.items()
    }
    errors = base_ctx.errors + [error for ctx in os_contexts.values() for error in ctx.errors]
    if conflicts or errors:
        print(conflict_report(conflicts, errors, overrides_path), file=sys.stderr)
        return 1
    for override in overrides.values():
        if not override.used:
            print(f"Note: {display_path(overrides_path)} decides {override.name}, which no notebook requires any more; the entry can be removed.")
    if args.check:
        print(f"The requirements of {len(files)} file(s) do not conflict"
              + (f" ({sum(o.used for o in overrides.values())} decision(s) from {display_path(overrides_path)})." if overrides else "."))
        return 0

    decisions = override_header(base_resolved)
    base_final_lines = build_final_lines(base_ctx, base_resolved, args.sort)
    write_requirements_output(output, files, base_final_lines, decisions=decisions)
    print(f"Wrote {output} with {len(base_final_lines)} entries (from {len(files)} files)")

    gpu_output = get_gpu_output_path(output)
    gpu_final_lines = build_gpu_lines(base_final_lines)
    write_requirements_output(
        gpu_output,
        files,
        gpu_final_lines,
        target_accelerator=GPU_DISPLAY_NAME,
        decisions=decisions,
    )
    print(f"Wrote {gpu_output} with {len(gpu_final_lines)} entries (from {len(files)} files, target {GPU_DISPLAY_NAME})")

    for os_name, ctx in os_contexts.items():
        if not ctx.groups and not ctx.other_lines:
            continue
        os_final_lines = build_final_lines(ctx, os_resolved[os_name], args.sort)
        os_output = get_os_output_path(output, os_name)
        write_requirements_output(os_output, files, os_final_lines, target_os=os_name, decisions=override_header(os_resolved[os_name]))
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
        # Require only the major.minor series: environment.yaml pins the exact
        # patch release for the installer, but a source install should accept
        # any patch release of that series (e.g. 3.12.3 for 3.12.13).
        python_series = ".".join(str(python_version).split(".")[:2])
        pattern = r'python_requires\s*=\s*["\'][><=\.0-9 ]+(PYTHON\_VERSION)?["\']'
        new_setup_text = re.sub(pattern, f'python_requires=">={python_series}"', setup_text)
        setup_path.write_text(new_setup_text, encoding="utf-8")
        print(f"Updated {setup_path} with python_requires>={python_series}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

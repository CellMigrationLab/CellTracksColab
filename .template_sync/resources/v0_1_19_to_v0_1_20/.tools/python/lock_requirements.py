#!/usr/bin/env python3
"""Lock the merged requirements files: requirements*.txt -> requirements*.lock.txt.

merge_requirements.py writes the requirements of the notebooks (requirements.txt, requirements_gpu.txt and the per-OS
requirements-<os>.txt). Those files name the packages the notebooks need, but not the packages these depend on, so two
installations of the same release could get different versions. This script writes, next to each file, a lock file that lists
every package pip installs, each with its exact version, and post_install installs the lock file when the installer bundles one.

How a lock file is made:
* `uv pip compile --universal`: one lock file that is valid on Windows, macOS and Linux (a package that only one platform needs carries
  an environment marker, e.g. `colorama==0.4.6 ; sys_platform == 'win32'`).
* `--python-version` is the Python of environment.yaml. uv reads it as the lowest Python of the resolution, so a few packages can appear
  twice with a `python_full_version` marker: pip installs the line that matches the Python of the installer.
* The conda packages of environment.yaml (jupyterlab, pip, setuptools, certifi, ...) are installed by the installer itself, so they are
  not written to the lock file (their own dependencies are), and a conda pin such as `jupyterlab=4.4.0` constrains the resolution.
* The per-OS files are resolved with requirements.lock.txt as constraints, so a package that both files need gets the same version.
* Pip options of the requirements file (`--extra-index-url` of the GPU file, ...) are copied to the top of the lock file.

A lock file is written again only when its requirements file, the Python or JupyterLab version (or this script's lock format) changed:
its header holds a fingerprint of those inputs. Re-running the script without such a change rewrites nothing, so a new release
installs the same versions as the previous one. To move every package to its newest version, delete the lock files and run the
workflow "Updates on notebook change" by hand (Actions > Run workflow).

If a file cannot be locked (uv is missing, or the requirements cannot be resolved), the script prints a warning, removes the
out-of-date lock file of that file (an installer must never install versions that no longer match the requirements) and goes on:
the installer then installs the plain requirements file. The script itself always exits 0 for such a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Bump when the uv arguments or the output change, so that every lock file is written again.
LOCK_FORMAT = "1"
BASE_INPUT = "requirements.txt"
GPU_INPUT = "requirements_gpu.txt"
OS_INPUTS = ("requirements-linux.txt", "requirements-windows.txt", "requirements-macos.txt")
INPUTS = (BASE_INPUT, GPU_INPUT, *OS_INPUTS)
FINGERPRINT_PREFIX = "# lock-input: sha256:"
# Pip options that choose where packages come from: copied to the top of the lock file, so pip finds the locked versions where uv found them.
INDEX_OPTION = re.compile(r"^\s*(--index-url|--extra-index-url|-i|--find-links|-f|--trusted-host)(\s|=)")
CONDA_SPEC = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:(==?)\s*([0-9][A-Za-z0-9_.+!*-]*))?(?:=\S+)?\s*$")
PLACEHOLDER = re.compile(r"^[A-Z_]+$")


def lock_path(requirements: Path) -> Path:
    """requirements.txt -> requirements.lock.txt (same folder)."""
    return requirements.with_name(requirements.stem + ".lock.txt")


def read_environment(path: Path) -> tuple[str | None, list[str], list[str]]:
    """(Python version, conda package names, pip constraints from exact conda pins) of environment.yaml."""
    if not path.is_file():
        return None, [], []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    python_version = None
    names: list[str] = []
    constraints: list[str] = []
    for dependency in data.get("dependencies", []) or []:
        if not isinstance(dependency, str):
            continue  # the `pip:` subsection of an environment file
        match = CONDA_SPEC.match(dependency)
        if match:
            name, operator, version = match.group(1), match.group(2), match.group(3)
        else:
            # a spec with a range or alternatives (`pip>=24.2`): only its name is used
            name, operator, version = re.split(r"[\s<>=!|,\[]", dependency.strip(), maxsplit=1)[0], None, None
        if not name:
            continue
        if version and PLACEHOLDER.match(version):
            version = None  # the template itself (python=PYTHON_VERSION)
        if name.lower() == "python":
            python_version = version
            continue
        names.append(name)
        if version:
            # conda `name=1.2` means any 1.2.* release, `name==1.2` exactly 1.2
            constraints.append(f"{name}=={version}" if operator == "==" else f"{name}=={version.removesuffix('.*')}.*")
    return python_version, sorted(set(names), key=str.lower), constraints


def normalized_requirements(text: str) -> str:
    """The requirement and option lines of a requirements file (comments, such as the timestamp of merge_requirements.py, are ignored)."""
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)


def fingerprint(requirements_text: str, python_version: str, constraints: list[str], no_emit: list[str], extra_constraints: str) -> str:
    payload = "\n---\n".join(
        [LOCK_FORMAT, normalized_requirements(requirements_text), python_version, "\n".join(constraints), "\n".join(no_emit), extra_constraints]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_fingerprint(lock: Path) -> str | None:
    if not lock.is_file():
        return None
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line.startswith(FINGERPRINT_PREFIX):
            return line[len(FINGERPRINT_PREFIX):].strip()
    return None


def run_uv(requirements: Path, python_version: str, constraint_files: list[Path], no_emit: list[str]) -> str:
    """The resolution of `requirements` (stdout of uv pip compile); raises RuntimeError with uv's message when it fails."""
    command = [
        sys.executable, "-m", "uv", "pip", "compile", str(requirements),
        "--universal", "--python-version", python_version,
        "--no-header", "--no-annotate", "--no-config", "--quiet",
    ]
    for constraint in constraint_files:
        command += ["--constraint", str(constraint)]
    for name in no_emit:
        command += ["--no-emit-package", name]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    except OSError as exc:
        raise RuntimeError(f"uv could not be started: {exc}") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise RuntimeError(message)
    return result.stdout


def uv_version() -> str:
    try:
        result = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return "uv"
    # "uv 0.13.3 (502f93ca6 2026-10-19 x86_64-pc-windows-msvc)" -> "uv 0.13.3": the build details differ between computers
    return " ".join(result.stdout.split()[:2]) or "uv"


def render_lock(requirements: Path, resolved: str, digest: str, python_version: str, uv: str) -> str:
    options = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines() if INDEX_OPTION.match(line)]
    packages = [line.rstrip() for line in resolved.replace("\r\n", "\n").split("\n") if line.strip() and not line.lstrip().startswith("#")]
    header = [
        f"# Lock file of {requirements.name}: every package the installer installs with pip, with its exact version.",
        f"# Written by .tools/python/lock_requirements.py ({uv}, universal resolution for Python {python_version}); do not edit it by hand.",
        f"# It is written again only when {requirements.name}, the Python or JupyterLab version changes (see the fingerprint below).",
        f"{FINGERPRINT_PREFIX}{digest}",
    ]
    body = options + ([""] if options else []) + packages
    return "\n".join(header + [""] + body) + "\n"


def warn(message: str) -> None:
    # `::warning::` shows the message on the summary page of the GitHub Actions run.
    print(f"::warning::{message}", flush=True)


def lock_file(
    requirements: Path,
    python_version: str,
    constraints: list[str],
    no_emit: list[str],
    base_lock: Path | None,
    work_dir: Path,
    uv: str,
) -> str:
    """Lock one requirements file; returns what happened ("unchanged", "written", "failed")."""
    lock = lock_path(requirements)
    extra = base_lock.read_text(encoding="utf-8") if base_lock is not None and base_lock.is_file() else ""
    digest = fingerprint(requirements.read_text(encoding="utf-8"), python_version, constraints, no_emit, normalized_requirements(extra))
    if current_fingerprint(lock) == digest:
        print(f"{lock}: up to date (unchanged)")
        return "unchanged"
    constraint_files = []
    if constraints:
        conda_constraints = work_dir / "conda_constraints.txt"
        conda_constraints.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        constraint_files.append(conda_constraints)
    if extra:
        # Only the package lines: the pip options of the base lock are not constraints.
        base_constraints = work_dir / "base_constraints.txt"
        base_constraints.write_text(
            "\n".join(line for line in extra.splitlines() if line.strip() and not line.startswith(("#", "-"))) + "\n", encoding="utf-8"
        )
        constraint_files.append(base_constraints)
    try:
        resolved = run_uv(requirements, python_version, constraint_files, no_emit)
    except RuntimeError as exc:
        warn(f"{requirements} could not be locked, so the installer installs it without a lock file: {exc}")
        if lock.exists():
            lock.unlink()
            print(f"{lock}: removed (it no longer matches {requirements})")
        return "failed"
    with lock.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_lock(requirements, resolved, digest, python_version, uv))
    print(f"{lock}: written")
    return "written"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock requirements*.txt into requirements*.lock.txt (every package with its exact version).")
    parser.add_argument("--root", type=Path, default=Path("."), help="Folder with the requirements files and environment.yaml")
    parser.add_argument("--environment", type=Path, default=None, help="Conda environment file (default: <root>/environment.yaml)")
    args = parser.parse_args(argv)
    root: Path = args.root
    environment = args.environment or root / "environment.yaml"

    python_version, conda_names, constraints = read_environment(environment)
    if not python_version:
        print(f"{environment}: no exact Python version (python=X.Y.Z), so nothing is locked.")
        return 0
    no_emit = [name for name in conda_names if name.lower() not in {"git"}]

    # A lock file whose requirements file is gone is removed (the installer must not install it).
    for name in INPUTS:
        requirements = root / name
        if not requirements.is_file() and lock_path(requirements).is_file():
            lock_path(requirements).unlink()
            print(f"{lock_path(requirements)}: removed ({requirements} does not exist)")

    present = [root / name for name in INPUTS if (root / name).is_file()]
    if not present:
        print("No requirements files found: nothing to lock.")
        return 0
    uv = uv_version()
    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        work_dir = Path(temporary)
        for requirements in present:
            base_lock = lock_path(root / BASE_INPUT) if requirements.name in OS_INPUTS else None
            results[requirements.name] = lock_file(requirements, python_version, constraints, no_emit, base_lock, work_dir, uv)
    failed = [name for name, status in results.items() if status == "failed"]
    print(f"Lock files: {len(results) - len(failed)} of {len(results)} requirements files locked" + (f"; not locked: {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

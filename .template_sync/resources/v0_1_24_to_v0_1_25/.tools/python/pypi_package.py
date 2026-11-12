#!/usr/bin/env python3
"""Build the app's Python package (the code in src/) for PyPI, as .github/workflows/publish_pypi.yml does at each release.

The workflow runs only when the repository variable PYPI_PUBLISH is "true" (PyPI) or "testpypi" (TestPyPI, a trial). This script:

1. stops before building, with the reason, when the release cannot be published: no package in src/, a version that is not X.Y.Z,
   a version that is already on the index, another workflow of the repository that also publishes to PyPI, or a setup.py that
   still has a template placeholder;
2. writes the release version into setup.py of the checkout, without committing it: the release tag points to the commit before
   the "Update version" commit of the release workflow, so the committed version cannot be trusted. The name is the one in
   setup.py; `distribution.pip` of labconstrictor-app.yaml, when there is one, must be the same name;
3. builds the source distribution and the wheel (`python -m build`: the wheel is built from the source distribution, so a file
   missing from it shows), and checks their name, version and dependencies.

It never uploads: the next job of the workflow does, with PyPI trusted publishing (no token).

Usage:
  python .tools/python/pypi_package.py --version 1.2.3 [--index pypi|testpypi] [--outdir dist] [--github-output FILE]
"""
from __future__ import annotations

import argparse
import email.parser
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = "publish_pypi.yml"
DOCS = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/external_code_upload.md#4-publish-the-code-on-pypi-optional"
# --index: the value of PYPI_PUBLISH ("true" is PyPI) -> (label, address)
INDEXES = {"pypi": ("PyPI", "https://pypi.org"), "true": ("PyPI", "https://pypi.org"), "testpypi": ("TestPyPI", "https://test.pypi.org")}
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
# Text in a workflow that uploads to PyPI (the action of the template, or twine).
PUBLISH_MARKERS = ("pypa/gh-action-pypi-publish", "twine upload")
PLACEHOLDERS = ("PYTHON_PROJ_NAME", "VERSION_NUMBER", "PYTHON_VERSION")


class PackageError(Exception):
    """A reason not to publish, written to the log of the workflow."""


def normalise(name: str) -> str:
    """The name as PyPI compares names (PEP 503): my_App.tools and my-app-tools are the same project."""
    return re.sub(r"[-_.]+", "-", name).lower()


def packages(root: Path) -> list[str]:
    """The packages of src/: its folders with an __init__.py."""
    src = root / "src"
    if not src.is_dir():
        return []
    return sorted(path.parent.name for path in src.glob("*/__init__.py"))


def other_publishers(root: Path) -> list[str]:
    """The workflows of the repository, other than the template's, that upload to PyPI."""
    found = []
    folder = root / ".github" / "workflows"
    for path in sorted(folder.glob("*.y*ml")) if folder.is_dir() else []:
        if path.name == WORKFLOW:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in PUBLISH_MARKERS):
            found.append(path.relative_to(root).as_posix())
    return found


def manifest_pip_name(root: Path) -> str | None:
    """`distribution.pip` of labconstrictor-app.yaml: the name of the package on PyPI, when the app gives one."""
    import yaml

    path = root / "labconstrictor-app.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PackageError(f"labconstrictor-app.yaml is not valid YAML ({exc}); run the 'Check app manifest' workflow to see where") from exc
    distribution = data.get("distribution") if isinstance(data, dict) else None
    name = distribution.get("pip") if isinstance(distribution, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _value_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^(?P<head>[ \t]*{key}\s*=\s*)(?P<quote>[\"'])(?P<value>[^\"'\n]*)(?P=quote)", re.MULTILINE)


def setup_value(text: str, key: str) -> str | None:
    """The first `key="value"` of setup.py (name, version), or None when it is not written there as a string."""
    match = _value_re(key).search(text)
    return match.group("value") if match else None


def set_setup_value(text: str, key: str, value: str) -> str:
    return _value_re(key).sub(lambda m: f"{m.group('head')}{m.group('quote')}{value}{m.group('quote')}", text, count=1)


def pyproject_project(root: Path) -> dict[str, Any] | None:
    """The [project] table of pyproject.toml, when the package is described there (it then wins over setup.py)."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return None
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8")).get("project")
    except tomllib.TOMLDecodeError as exc:
        raise PackageError(f"pyproject.toml is not valid TOML: {exc}") from exc
    return project if isinstance(project, dict) else None


def version_on_index(index: str, name: str, version: str, opener: Callable[..., Any] = urllib.request.urlopen) -> bool | None:
    """True when the index already has this version of the project, False when not, None when the index could not be asked."""
    url = f"{INDEXES[index][1]}/pypi/{normalise(name)}/{version}/json"
    try:
        with opener(url, timeout=30) as response:
            return bool(response.status == 200)
    except urllib.error.HTTPError as exc:
        return False if exc.code == 404 else None
    except (urllib.error.URLError, OSError):
        return None


def prepare(root: Path, version: str, index: str, check_index: bool = True) -> str:
    """Check that the release can be published and write its version (and name) into setup.py; returns the name on the index."""
    if not VERSION_RE.fullmatch(version):
        raise PackageError(f"The release version {version!r} is not X.Y.Z (the tag must be 1.2.3 or v1.2.3, as for the installers).")
    found = packages(root)
    if not found:
        raise PackageError(
            "There is no Python package in src/ (a folder with an __init__.py, for example src/my_app/__init__.py), so there is nothing to "
            f"publish. Add your code to src/ (see .tools/docs/external_code_upload.md), or delete the repository variable PYPI_PUBLISH. {DOCS}"
        )
    others = other_publishers(root)
    if others:
        raise PackageError(
            f"{', '.join(others)} also publishes to PyPI: with PYPI_PUBLISH set, the release would be uploaded twice. Keep one: delete the "
            f"repository variable PYPI_PUBLISH (your workflow keeps publishing), or delete your workflow (this one publishes at each release). {DOCS}"
        )

    pip_name = manifest_pip_name(root)
    project = pyproject_project(root)
    setup_path = root / "setup.py"
    if project is not None and "name" in project:
        # pyproject.toml describes the package: its name and version are used as they are, and must match.
        name = str(project["name"])
        if pip_name and normalise(pip_name) != normalise(name):
            raise PackageError(f"distribution.pip of labconstrictor-app.yaml is {pip_name}, but pyproject.toml names the package {name}: make them the same.")
        if "version" in project.get("dynamic", []):
            return name
        if project.get("version") != version:
            raise PackageError(
                f"pyproject.toml gives the version {project.get('version')}, not the release version {version}: set `version = \"{version}\"` in "
                "pyproject.toml, commit, and publish the release again."
            )
        return name
    if not setup_path.is_file():
        raise PackageError("There is no setup.py (nor a [project] table in pyproject.toml) to build the package from: restore the template's setup.py.")

    text = setup_path.read_bytes().decode("utf-8")  # the line endings are kept when the version is written
    left = [placeholder for placeholder in PLACEHOLDERS if placeholder in text and placeholder != "VERSION_NUMBER"]
    if left:
        raise PackageError(
            f"setup.py still has the template placeholder {', '.join(left)}: PYTHON_PROJ_NAME is the name of the package (for example my_app), "
            "PYTHON_VERSION the oldest Python series it runs on (for example 3.12). Replace it, commit, and publish the release again."
        )
    setup_name = setup_value(text, "name")
    if pip_name and setup_name and normalise(pip_name) != normalise(setup_name):
        raise PackageError(
            f"distribution.pip of labconstrictor-app.yaml is {pip_name}, but setup.py names the package {setup_name}: make them the same "
            "(distribution.pip is the name the App Centre tells people to pip install; the package of src/ is published under setup.py's name)."
        )
    package_name = setup_name or pip_name
    if not package_name:
        raise PackageError("setup.py has no name=\"...\" for the package: give it the name it should have on PyPI (and in distribution.pip).")
    if check_index:
        label = INDEXES[index][0]
        exists = version_on_index(index, package_name, version)
        if exists:
            raise PackageError(
                f"{label} already has {package_name} {version}, and a version can be uploaded only once (even after it was deleted). "
                f"Publish a new release with a higher version (the installers are built for it too). "
                f"{INDEXES[index][1]}/project/{normalise(package_name)}/{version}/"
            )
        if exists is None:
            print(f"WARNING: {label} could not be asked whether {package_name} {version} already exists; the upload fails if it does.")

    updated = set_setup_value(text, "version", version)
    if updated != text:
        # newline="" keeps the line endings of the file
        with setup_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
    return package_name


def left_out_requirements(root: Path) -> list[str]:
    """Lines of requirements.txt that the template's setup.py does not turn into dependencies (options, paths, URLs)."""
    setup_text = (root / "setup.py").read_text(encoding="utf-8") if (root / "setup.py").is_file() else ""
    path = root / "requirements.txt"
    if "read_requirements()" not in setup_text or not path.is_file():
        return []
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return []
    left = []
    for line in (raw.split("#", 1)[0].strip() for raw in path.read_text(encoding="utf-8").splitlines()):
        if not line:
            continue
        try:
            if Requirement(line).url is None:
                continue
        except InvalidRequirement:
            pass
        left.append(line)
    return left


def build(root: Path, outdir: Path, isolation: bool = True) -> None:
    """`python -m build`: the source distribution, then the wheel built from it."""
    command = [sys.executable, "-m", "build", "--outdir", str(outdir), *([] if isolation else ["--no-isolation"]), str(root)]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise PackageError("The package could not be built: see the output of `python -m build` above.")


def _headers(data: bytes) -> Message:
    return email.parser.BytesParser().parsebytes(data, headersonly=True)


def read_metadata(outdir: Path) -> tuple[Message, Message, list[Path]]:
    """The metadata of the wheel and of the source distribution in outdir, and the two files."""
    wheels = sorted(outdir.glob("*.whl"))
    sdists = sorted(outdir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise PackageError(f"Expected one wheel and one source distribution in {outdir}, found: {', '.join(p.name for p in sorted(outdir.iterdir())) or 'nothing'}")
    with zipfile.ZipFile(wheels[0]) as archive:
        member = next((n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")), None)
        if member is None:
            raise PackageError(f"{wheels[0].name} has no METADATA")
        wheel = _headers(archive.read(member))
    with tarfile.open(sdists[0]) as archive:
        info = next((m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")), None)
        handle = archive.extractfile(info) if info is not None else None
        if handle is None:
            raise PackageError(f"{sdists[0].name} has no PKG-INFO")
        sdist = _headers(handle.read())
    return wheel, sdist, [sdists[0], wheels[0]]


def check(outdir: Path, name: str, version: str) -> list[str]:
    """Check the built files; returns their names."""
    wheel, sdist, files = read_metadata(outdir)
    for label, metadata in (("wheel", wheel), ("source distribution", sdist)):
        if normalise(str(metadata.get("Name", ""))) != normalise(name):
            raise PackageError(f"The {label} is named {metadata.get('Name')}, not {name}: setup.py or pyproject.toml set another name.")
        if metadata.get("Version") != version:
            raise PackageError(f"The {label} has the version {metadata.get('Version')}, not the release version {version}: setup.py or pyproject.toml set it.")
    requirements = wheel.get_all("Requires-Dist") or []
    urls = [requirement for requirement in requirements if " @ " in requirement]
    if urls:
        raise PackageError(
            f"PyPI refuses a package whose dependency is a URL: {'; '.join(urls)}. Publish that dependency on PyPI first and name it with a version, "
            "or keep the URL in requirements.txt only (the installer still installs it)."
        )
    if sdist.get_all("Requires-Dist") is not None and sorted(sdist.get_all("Requires-Dist") or []) != sorted(requirements):
        raise PackageError(
            "The wheel built from the source distribution has other dependencies than the source distribution: a file setup.py reads is not in "
            "the source distribution (requirements.txt needs `include requirements.txt` in MANIFEST.in)."
        )
    return [path.name for path in files]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and check the app's Python package (src/) for PyPI. Nothing is uploaded.")
    parser.add_argument("--version", required=True, help="The release version, X.Y.Z (the tag without the v)")
    parser.add_argument("--index", default="pypi", choices=sorted(INDEXES), help="pypi (or true, the value of PYPI_PUBLISH) or testpypi")
    parser.add_argument("--outdir", default="dist", help="Where to write the two files (default: dist)")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--github-output", help="Append name=, version= and url= to this file (GitHub Actions $GITHUB_OUTPUT)")
    parser.add_argument("--no-isolation", action="store_true", help="Build with the setuptools already installed (no download)")
    parser.add_argument("--offline", action="store_true", help="Do not ask the index whether the version exists")
    args = parser.parse_args(argv)

    root = Path(args.repository_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    label, address = INDEXES[args.index]
    try:
        name = prepare(root, args.version, args.index, check_index=not args.offline)
        for line in left_out_requirements(root):
            print(f"WARNING: the line `{line}` of requirements.txt is not a dependency the package can declare on {label} (an option, a path or a URL): left out.")
        if outdir.exists() and any(outdir.iterdir()):
            raise PackageError(f"{outdir} is not empty: remove it first, so that only this release is uploaded.")
        build(root, outdir, isolation=not args.no_isolation)
        files = check(outdir, name, args.version)
    except PackageError as exc:
        print(f"::error title=The package cannot be published::{exc}", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    url = f"{address}/project/{normalise(name)}/{args.version}/"
    print(f"Built {name} {args.version} for {label}: {', '.join(files)}. The next job uploads it: {url}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"name={name}\nversion={args.version}\nurl={url}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

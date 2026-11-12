"""The Python package of the app: the code in src/ (see .tools/docs/external_code_upload.md).

The installer installs it into the app. With the repository variable PYPI_PUBLISH, .github/workflows/publish_pypi.yml also
publishes it on PyPI at every release, with the version of the release. The package needs what requirements.txt lists, and
its description on PyPI is README.md. Both files are read when the package is built; they may be missing (the installer).
"""
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
# A requirement: a project, optional extras, optional version specifiers, an optional environment marker.
REQUIREMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\s*(\[[A-Za-z0-9._,\s-]*\])?\s*([<>=!~]=?\s*[^\s,;]+(\s*,\s*[<>=!~]=?\s*[^\s,;]+)*)?\s*(;.+)?")


def is_requirement(line):
    """True for a requirement setuptools accepts; options (-r, --index-url), paths and URLs are not requirements."""
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return REQUIREMENT_RE.fullmatch(line) is not None
    try:
        return Requirement(line).url is None
    except InvalidRequirement:
        return False


def read_requirements():
    """The requirements in requirements.txt, one per line (comments and anything else left out); none without the file."""
    path = HERE / "requirements.txt"
    if not path.is_file():
        return []
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and is_requirement(line)]


def read_readme():
    path = HERE / "README.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# The repository on GitHub, when the package is built by a workflow (GitHub sets GITHUB_REPOSITORY).
REPOSITORY = os.environ.get("GITHUB_REPOSITORY")

setup(
    name="PYTHON_PROJ_NAME",
    version="VERSION_NUMBER",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    project_urls={"Source": f"https://github.com/{REPOSITORY}"} if REPOSITORY else {},
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=PYTHON_VERSION",
    install_requires=read_requirements(),
)

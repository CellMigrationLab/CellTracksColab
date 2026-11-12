#!/usr/bin/env python3
"""Which installers a release builds: the systems of `distribution.platforms` in labconstrictor-app.yaml (CellMigrationLab/LabConstrictor#42).

The release workflow builds one installer per system. An app that cannot run on a system (TensorFlow 2.17+ has no wheels for Intel Macs, for
example) leaves that system out of `distribution.platforms`, and then:
  * the release workflow does not build it (this script writes the build matrix of release_constructor.yml);
  * the download page (.tools/docs/download_executable.md, written by bump_version.py) has no button for it.
Without labconstrictor-app.yaml, or without `distribution.platforms` in it, every system below is built, as before template 0.1.24.

Usage: python .tools/python/release_platforms.py [--repository-root PATH] [--github-output FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import yaml

MANIFEST = "labconstrictor-app.yaml"
DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_24.md"


class Build(NamedTuple):
    platform: str  # the name in distribution.platforms
    os_name: str  # matrix.os_name of release_constructor.yml (names the build job, its log and dist/ folder)
    runner: str  # the GitHub runner that builds it
    files: tuple[str, ...]  # how its installers end: <App>-<version><ending>
    label: str  # for people: the download page and the messages


BUILDS = (
    Build("linux-x86_64", "linux", "ubuntu-latest", ("-Linux-x86_64.sh",), "Linux"),
    Build("macos-arm64", "macos-arm64", "macos-15", ("-MacOSX-arm64.pkg", "-MacOSX-arm64.sh"), "Macs with Apple silicon (M1, M2, ...)"),
    Build("macos-x86_64", "macos-intel", "macos-15-intel", ("-MacOSX-x86_64.pkg", "-MacOSX-x86_64.sh"), "Macs with an Intel processor"),
    Build("windows-x86_64", "windows", "windows-latest", ("-Windows-x86_64.exe",), "Windows"),
)


def listed_platforms(root: Path) -> list[str] | None:
    """`distribution.platforms` of the app manifest; None when there is no manifest or the key is left out (every system)."""
    path = root / MANIFEST
    if not path.is_file():
        return None
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{MANIFEST} is not valid YAML: {exc}") from exc
    distribution = data.get("distribution") if isinstance(data, dict) else None
    platforms = distribution.get("platforms") if isinstance(distribution, dict) else None
    if platforms is None:
        return None
    if not isinstance(platforms, list) or not all(isinstance(item, str) for item in platforms):
        raise ValueError(f"distribution.platforms in {MANIFEST} must be a list of system names, such as [windows-x86_64, macos-arm64]")
    return platforms


def builds(root: Path) -> list[Build]:
    """The builds of a release, in the order of BUILDS. Raises ValueError when the manifest lists no system this template builds."""
    listed = listed_platforms(root)
    if listed is None:
        return list(BUILDS)
    chosen = [build for build in BUILDS if build.platform in listed]
    if not chosen:
        known = ", ".join(build.platform for build in BUILDS)
        raise ValueError(
            f"distribution.platforms in {MANIFEST} lists {', '.join(listed) or 'nothing'}, and the release builds installers only for {known}. "
            "Add at least one of them, or remove distribution.installer if the app has no installer."
        )
    return chosen


def skipped_builds(root: Path) -> list[Build]:
    """The systems the release does not build (none without distribution.platforms)."""
    chosen = builds(root)
    return [build for build in BUILDS if build not in chosen]


def matrix(chosen: list[Build]) -> dict[str, list[dict[str, str]]]:
    return {"include": [{"os_name": build.os_name, "runner": build.runner} for build in chosen]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the build matrix of the release: the systems of distribution.platforms in labconstrictor-app.yaml.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path, help="append matrix=<json> to this file (the GITHUB_OUTPUT of a workflow step)")
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    try:
        chosen = builds(root)
        listed = listed_platforms(root)
    except ValueError as exc:
        print(f"{exc} See {DOCS_URL}", file=sys.stderr)
        return 1
    if listed is None:
        print(f"Building every installer: {MANIFEST} does not list distribution.platforms.")
    else:
        print(f"Building the installers of distribution.platforms in {MANIFEST}: {', '.join(build.platform for build in chosen)}.")
        for build in BUILDS:
            if build not in chosen:
                print(f"  Not built: {build.platform} ({build.label}).")
        for name in listed:
            if name not in {build.platform for build in BUILDS}:
                print(f"  Not built: {name} (the release has no runner for it).")
    value = json.dumps(matrix(chosen), separators=(",", ":"))
    print(f"matrix={value}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"matrix={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

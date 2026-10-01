from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def parse_manifest(manifest_path):
    # This function parsers the manifest file withouth the need from external dependencies
    template_version = None
    migrations = []
    current_migration = None
    in_migrations = False

    # Read the manifest file line by line and parse it manually
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if not in_migrations:
            if stripped.startswith("template_version:"):
                template_version = stripped.split(":", 1)[1].strip()
                continue
            if stripped == "migrations:":
                in_migrations = True
                continue
            raise ValueError(f"Unsupported manifest line: {line}")

        if stripped.startswith("- "):
            current_migration = {}
            migrations.append(current_migration)
            key, value = stripped[2:].split(":", 1)
            current_migration[key.strip()] = value.strip()
            continue

        if current_migration is None or ":" not in stripped:
            raise ValueError(f"Unsupported migration line: {line}")

        key, value = stripped.split(":", 1)
        current_migration[key.strip()] = value.strip()

    if not template_version:
        raise ValueError("Manifest is missing template_version")

    for migration in migrations:
        missing_keys = {"from", "to", "script"} - migration.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Migration entry is missing keys: {missing}")

    return {
        "template_version": template_version,
        "migrations": migrations,
    }


def parse_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as exc:
        raise ValueError(f"Invalid version string: {version}") from exc


def resolve_migration_chain(
    migrations: list[dict[str, str]], from_version: str, to_version: str
) -> list[dict[str, str]]:
    if parse_version(from_version) > parse_version(to_version):
        raise ValueError(
            f"Downgrades are not supported: {from_version} -> {to_version}"
        )

    if from_version == to_version:
        return []

    migration_map = {migration["from"]: migration for migration in migrations}
    resolved: list[dict[str, str]] = []
    current_version = from_version

    while current_version != to_version:
        migration = migration_map.get(current_version)
        if migration is None:
            raise ValueError(
                "No migration path found from "
                f"{from_version} to {to_version}. Missing step starting at "
                f"{current_version}."
            )

        next_version = migration["to"]
        if parse_version(next_version) <= parse_version(current_version):
            raise ValueError(
                "Migration versions must move forward strictly: "
                f"{current_version} -> {next_version}"
            )

        resolved.append(migration)
        current_version = next_version

    return resolved


def load_migration(script_path: Path):
    module_name = f"template_sync_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load migration script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    migrate = getattr(module, "migrate", None)
    if not callable(migrate):
        raise RuntimeError(
            f"Migration script does not define callable migrate(): {script_path}"
        )

    return migrate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run template synchronization migrations."
    )
    parser.add_argument("--from-version", required=True)
    parser.add_argument("--to-version", required=True)
    parser.add_argument(
        "--repository-root",
        default=".",
        help="Path to the repository root that should be updated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the migration plan.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repository_root).resolve()
    manifest_path = repo_root / ".template_sync" / "manifest.yaml"
    manifest = parse_manifest(manifest_path)

    target_version = manifest["template_version"]
    if target_version != args.to_version:
        raise ValueError(
            f"Target version {args.to_version} does not match manifest version "
            f"{target_version}"
        )

    migration_chain = resolve_migration_chain(
        manifest["migrations"], args.from_version, args.to_version
    )

    if not migration_chain:
        print("No migrations required.")
        return 0

    print(
        "Applying template sync migrations:",
        " -> ".join(
            [args.from_version] + [migration["to"] for migration in migration_chain]
        ),
    )

    for migration in migration_chain:
        script_path = repo_root / ".template_sync" / migration["script"]
        print(
            f"Running migration {migration['from']} -> {migration['to']} "
            f"using {migration['script']}"
        )
        if args.dry_run:
            continue

        migrate = load_migration(script_path)
        migrate(
            repo_root=repo_root,
            context={
                "from_version": migration["from"],
                "to_version": migration["to"],
                "manifest": manifest,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

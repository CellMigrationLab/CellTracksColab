from __future__ import annotations

from pathlib import Path
from typing import Any


RESOURCE_ROOT = (
    Path(__file__).resolve().parents[1] / "resources" / "v0_1_11_to_v0_1_12"
)
SYNC_WORKFLOW_PATH = Path(".github/workflows/sync_template.yml")
SYNC_GUIDE_PATH = Path(".tools/docs/template_synchronization.md")
ROOT_README_PATH = Path("README.md")
DOCS_INDEX_PATH = Path(".tools/docs/README.md")
BEFORE_GETTING_STARTED_PATH = Path(".tools/docs/before_getting_started.md")
CREATE_REPOSITORY_PATH = Path(".tools/docs/create_repository.md")
INITIALISE_REPOSITORY_PATH = Path(".tools/docs/initialise_repository.md")
PERSONAL_TOKEN_PATH = Path(".tools/docs/personal_access_token.md")
TROUBLESHOOTING_PATH = Path(".tools/docs/troubleshooting.md")
WORKFLOW_STATUS_PATH = Path(".tools/docs/workflow_status.md")
ACCEPT_PR_PATH = Path(".tools/docs/accept_pull_request.md")

ROOT_GUIDE_LINK = ".tools/docs/template_synchronization.md"
DOC_GUIDE_LINK = "template_synchronization.md"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def normalize_newlines(text: str, newline: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", newline)


def copy_resource(repo_root: Path, relative_path: Path) -> bool:
    source_path = RESOURCE_ROOT / relative_path
    destination_path = repo_root / relative_path
    if not source_path.is_file():
        raise ValueError(f"Missing migration resource: {source_path}")

    source_text = read_text(source_path)
    if destination_path.exists():
        existing_text = read_text(destination_path)
        newline = detect_newline(existing_text)
    else:
        existing_text = ""
        newline = detect_newline(source_text)

    rendered = normalize_newlines(source_text, newline)
    if rendered == existing_text:
        return False

    write_text(destination_path, rendered)
    print(f"Updated {relative_path}")
    return True


def insert_before_navigation(text: str, block: str, newline: str) -> str:
    normalized = text.replace("\r\n", "\n")
    normalized_block = block.strip("\n")
    markers = (
        "\n---\n\n<div align=\"center\">",
        "\n---\n<div align=\"center\">",
    )
    for marker in markers:
        if marker in normalized:
            updated = normalized.replace(
                marker,
                f"\n\n{normalized_block}\n{marker}",
                1,
            )
            return updated.replace("\n", newline)

    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return (normalized + "\n" + normalized_block + "\n").replace("\n", newline)


def update_file(repo_root: Path, relative_path: Path, updater) -> bool:
    path = repo_root / relative_path
    if not path.is_file():
        print(f"Skipping missing file: {relative_path}")
        return False

    original = read_text(path)
    updated = updater(original, detect_newline(original))
    if updated == original:
        return False

    write_text(path, updated)
    print(f"Updated {relative_path}")
    return True


def update_root_readme(text: str, newline: str) -> str:
    if ROOT_GUIDE_LINK in text:
        return text

    block = """## 🔄 Automatic Template Updates

LabConstrictor can prepare pull requests that keep this repository aligned with improvements in the main template, including updates to GitHub Actions workflows. Complete the one-time [automatic synchronization setup](.tools/docs/template_synchronization.md) to enable these updates."""
    normalized = text.replace("\r\n", "\n")
    marker = "\n## 🤝 Contributing"
    if marker in normalized:
        normalized = normalized.replace(marker, f"\n\n{block}\n{marker}", 1)
        return normalized.replace("\n", newline)
    return insert_before_navigation(text, block, newline)


def update_docs_index(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text

    normalized = text.replace("\r\n", "\n")
    entry = "- [Configure automatic template synchronization](template_synchronization.md)"
    marker = "## GitHub management"
    if marker in normalized:
        start = normalized.index(marker) + len(marker)
        next_heading = normalized.find("\n## ", start)
        insert_at = next_heading if next_heading != -1 else len(normalized)
        section = normalized[start:insert_at].rstrip()
        normalized = normalized[:start] + section + "\n\n" + entry + "\n" + normalized[insert_at:]
        return normalized.replace("\n", newline)

    return insert_before_navigation(text, "## GitHub management\n\n" + entry, newline)


def update_before_getting_started(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    block = """## 🔄 One-time update setup

After creating your repository, complete the [automatic template synchronization setup](template_synchronization.md). This allows future template improvements to be delivered as pull requests, including necessary GitHub Actions workflow updates."""
    return insert_before_navigation(text, block, newline)


def update_create_repository(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    block = """## Enable future template updates

Before initialising the project, complete the one-time [automatic template synchronization setup](template_synchronization.md). It lets LabConstrictor prepare safe pull requests when the template receives fixes, including changes to GitHub Actions workflows."""
    return insert_before_navigation(text, block, newline)


def add_callout_after_title(text: str, newline: str, callout: str) -> str:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return text
    insert_at = 1
    while insert_at < len(lines) and not lines[insert_at].strip():
        insert_at += 1
    rendered = "\n" + callout.strip() + "\n\n"
    updated = "".join(lines[:insert_at]) + rendered + "".join(lines[insert_at:])
    return updated.replace("\n", newline)


def update_initialise_repository(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    return add_callout_after_title(
        text,
        newline,
        "> Before continuing, complete the one-time [automatic template synchronization setup](template_synchronization.md). This allows future template migrations to update workflow files safely.",
    )


def update_personal_token(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    return add_callout_after_title(
        text,
        newline,
        "> Looking for the token used by the automatic template update workflow? Follow the dedicated [automatic template synchronization guide](template_synchronization.md). It uses a repository-scoped fine-grained token with a different purpose and secret name.",
    )


def update_workflow_status(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    return add_callout_after_title(
        text,
        newline,
        "The template synchronization workflow requires the one-time [`LABCONSTRICTOR_SYNC_TOKEN` setup](template_synchronization.md). Other notebook and installer workflows continue to use GitHub's built-in credentials.",
    )


def update_accept_pr(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text:
        return text
    return add_callout_after_title(
        text,
        newline,
        "LabConstrictor template updates are delivered as pull requests after you complete the [automatic synchronization setup](template_synchronization.md).",
    )


def update_troubleshooting(text: str, newline: str) -> str:
    if DOC_GUIDE_LINK in text and "LABCONSTRICTOR_SYNC_TOKEN" in text:
        return text

    normalized = text.replace("\r\n", "\n")
    section = """## Synchronisation is failing on GitHub Actions

The **Sync with Template Repository** workflow needs the encrypted repository secret `LABCONSTRICTOR_SYNC_TOKEN` because template migrations can update files in `.github/workflows/`. Follow the [automatic template synchronization guide](template_synchronization.md) for the complete beginner-friendly setup.

### The synchronization token is missing

Create the token and save it under **Settings → Secrets and variables → Actions** with this exact name:

```text
LABCONSTRICTOR_SYNC_TOKEN
```

### GitHub refuses to update a workflow file

If the error says GitHub is refusing to create or update a workflow without `workflows` permission, ensure the token has all of these repository permissions:

- **Contents:** Read and write
- **Pull requests:** Read and write
- **Workflows:** Read and write

Repositories created before template version `0.1.12` also need the one-time workflow edit described in the synchronization guide so the existing workflow starts using the secret.

### Bad credentials or authentication failed

The token may have expired or been revoked. Create a replacement token and update the existing `LABCONSTRICTOR_SYNC_TOKEN` secret."""

    heading = "## Synchronisation is failing on GitHub Actions"
    if heading in normalized:
        start = normalized.index(heading)
        next_heading = normalized.find("\n## ", start + len(heading))
        if next_heading == -1:
            normalized = normalized[:start].rstrip() + "\n\n" + section + "\n"
        else:
            normalized = (
                normalized[:start].rstrip()
                + "\n\n"
                + section
                + "\n\n"
                + normalized[next_heading + 1 :]
            )
        return normalized.replace("\n", newline)

    marker = "## JupyterLab does not start"
    if marker in normalized:
        normalized = normalized.replace(marker, section + "\n\n" + marker, 1)
        return normalized.replace("\n", newline)
    return insert_before_navigation(text, section, newline)


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context
    changed = False

    changed = copy_resource(repo_root, SYNC_WORKFLOW_PATH) or changed
    changed = copy_resource(repo_root, SYNC_GUIDE_PATH) or changed

    updates = (
        (ROOT_README_PATH, update_root_readme),
        (DOCS_INDEX_PATH, update_docs_index),
        (BEFORE_GETTING_STARTED_PATH, update_before_getting_started),
        (CREATE_REPOSITORY_PATH, update_create_repository),
        (INITIALISE_REPOSITORY_PATH, update_initialise_repository),
        (PERSONAL_TOKEN_PATH, update_personal_token),
        (TROUBLESHOOTING_PATH, update_troubleshooting),
        (WORKFLOW_STATUS_PATH, update_workflow_status),
        (ACCEPT_PR_PATH, update_accept_pr),
    )
    for relative_path, updater in updates:
        changed = update_file(repo_root, relative_path, updater) or changed

    if not changed:
        print("Template synchronization token integration is already up to date.")

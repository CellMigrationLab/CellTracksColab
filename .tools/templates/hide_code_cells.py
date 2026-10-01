"""Utility to hide code cells in Jupyter notebooks by updating metadata."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import nbformat

# Metadata values required to hide code cells in supported viewers.
CELL_VIEW_VALUE = "form"
COLLAPSED_VALUE = False
SOURCE_HIDDEN_VALUE = True

# Placeholder for repository initialisiation that enables or disables this script.
DISABLED = HIDE_CODE_DISABLED


def collect_notebooks(root: Path) -> Iterable[Path]:
    """Yield every notebook under root (recursively for directories)."""
    if root.is_file():
        if root.suffix == ".ipynb":
            yield root
        return

    for path in root.rglob("*.ipynb"):
        if path.is_file():
            yield path


def hide_code_cells(notebook_path: Path) -> bool:
    """Apply metadata updates to every code cell in the given notebook."""
    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    changed = False

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue

        # Remove saved output
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
            
        metadata = cell.setdefault("metadata", {})

        if metadata.get("cellView") != CELL_VIEW_VALUE:
            metadata["cellView"] = CELL_VIEW_VALUE
            changed = True

        if metadata.get("collapsed") != COLLAPSED_VALUE:
            metadata["collapsed"] = COLLAPSED_VALUE
            changed = True

        jupyter_meta = metadata.get("jupyter") if isinstance(metadata.get("jupyter"), dict) else {}
        if jupyter_meta.get("source_hidden") != SOURCE_HIDDEN_VALUE:
            jupyter_meta["source_hidden"] = SOURCE_HIDDEN_VALUE
            metadata["jupyter"] = jupyter_meta
            changed = True
            
        jupyterlab_meta = metadata.get("jupyterlabHideCode") if isinstance(metadata.get("jupyterlabHideCode"), dict) else {}
        if jupyterlab_meta.get("locked") != True:
            jupyterlab_meta["locked"] = True
            metadata["jupyterlabHideCode"] = jupyterlab_meta
            changed = True

    if changed:
        nbformat.write(notebook, notebook_path)

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hide the code cells in notebooks under the given path.")
    parser.add_argument("path", type=Path, help="File or directory to process for Jupyter notebooks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebooks = list(collect_notebooks(args.path))

    if DISABLED:
        print("Hiding code cells is disabled for this repository.")
        print("To enable, set HIDE_CODE_DISABLED to False in app/python_scripts/hide_code_cells.py")
        print("or go to https://labconstrictor-form.streamlit.app/ and initialise your repository again but with hiding code cells enabled.")
        return

    if not notebooks:
        print("No notebooks found.")
        return

    for notebook_path in notebooks:
        if hide_code_cells(notebook_path):
            print(f"Updated {notebook_path}")
        else:
            print(f"No changes needed for {notebook_path}")


if __name__ == "__main__":
    main()

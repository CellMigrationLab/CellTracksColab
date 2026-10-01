from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any


WELCOME_PATHS = (
    Path(".tools/templates/Welcome_template.ipynb"),
    Path("app/menuinst/Welcome.ipynb"),
)

GRID_HELPERS = r'''GRID_COLUMN_WIDTHS = (
    "minmax(0, 1.4fr) "      # Notebook Name
    "minmax(0, 1.4fr) "      # Notebook Topic
    "minmax(0, 2.8fr) "      # Description
    "minmax(85px, 0.8fr) "   # Local Version
    "minmax(100px, 1.2fr) "  # Status
    "minmax(70px, 0.7fr) "   # Update?
    "minmax(150px, 1.6fr)"   # Open Notebook
)


def apply_grid_layout(grid):
    """Apply responsive sizing while allowing long text to wrap safely."""
    grid.add_class("labconstrictor-responsive-grid")
    grid.layout.width = "100%"
    grid.layout.min_width = "0"
    grid.layout.grid_template_columns = GRID_COLUMN_WIDTHS
    grid.layout.grid_auto_rows = "auto"
    grid.layout.overflow = "auto"


def table_cell(value, align="center", extra_class=""):
    """Create a table cell that can shrink and wrap long scientific identifiers."""
    classes = ["labconstrictor-table-cell"]
    if align == "left":
        classes.append("labconstrictor-table-cell-left")
    else:
        classes.append("labconstrictor-table-cell-center")
    if extra_class:
        classes.append(extra_class)

    class_names = " ".join(classes)

    return widgets.HTML(
        value=f"<div class='{class_names}'>{value}</div>",
        layout=widgets.Layout(width="100%", min_width="0"),
    )


def table_header(value):
    """Create a responsive table header."""
    return widgets.HTML(
        value=f"<div class='grid-header'>{value}</div>",
        layout=widgets.Layout(width="100%", min_width="0"),
    )


'''

WELCOME_CSS_BLOCK = r'''    # Add custom CSS for responsive table styling
    display(widgets.HTML("""
    <style>
    .grid-header {
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        background-color: #ff6600 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: clamp(13px, 1vw, 16px) !important;
        line-height: 1.25 !important;
        padding: 12px 6px !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .labconstrictor-table-cell {
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        font-size: clamp(12px, 0.9vw, 15px) !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        padding: 5px 6px !important;
    }

    .labconstrictor-table-cell-center {
        text-align: center !important;
    }

    .labconstrictor-table-cell-left {
        text-align: left !important;
    }

    .labconstrictor-responsive-grid .widget-html {
        min-width: 0 !important;
        width: 100% !important;
    }

    .labconstrictor-responsive-grid .widget-html-content {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }

    .labconstrictor-responsive-grid .widget-button {
        font-size: clamp(11px, 0.85vw, 14px) !important;
    }

    .widget-button.open-notebook-button {
        background-color: #2196F3 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        padding: 0 12px !important;
        white-space: nowrap !important;
    }
    </style>
    """))


'''


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _replace_welcome_assignments(source: str) -> str:
    """Replace the 0.1.13 HTML cells with wrapping table helpers."""
    replacements = {
        'grid[idx, 0] = widgets.HTML(f"<div style=\'text-align: center;\'>{nb[\'name\']}</div>")':
            'grid[idx, 0] = table_cell(nb["name"])',
        'grid[idx, 1] = widgets.HTML(f"<div style=\'text-align: center;\'>{main_folder}</div>")':
            'grid[idx, 1] = table_cell(main_folder)',
        'grid[idx, 2] = widgets.HTML(f"<div style=\'text-align: center;\'>{nb[\'description\']}</div>")':
            'grid[idx, 2] = table_cell(nb["description"], align="left")',
        'grid[idx, 2] = widgets.HTML(f"<div class=\'table-description\'>{nb[\'description\']}</div>")':
            'grid[idx, 2] = table_cell(nb["description"], align="left")',
        'grid[idx, 2] = widgets.HTML("<div style=\'text-align: center;\'>-</div>")':
            'grid[idx, 2] = table_cell("-")',
        'grid[idx, 3] = widgets.HTML(f"<div style=\'text-align: center;\'>{local_version}</div>")':
            'grid[idx, 3] = table_cell(local_version)',
        'grid[idx, 4] = widgets.HTML("<div style=\'text-align: center;\'><span style=\'color: red;\'>❌ Project Outdated</span></div>")':
            'grid[idx, 4] = table_cell("<span style=\'color: red;\'>❌ Project Outdated</span>")',
        'grid[idx, 5] = widgets.HTML("<div style=\'text-align: center;\'>-</div>")':
            'grid[idx, 5] = table_cell("-")',
        'grid[idx, 4] = widgets.HTML("<div style=\'text-align: center;\'>✅ Up-to-date</div>")':
            'grid[idx, 4] = table_cell("✅ Up-to-date")',
        'grid[idx, 4] = widgets.HTML(f"<div style=\'text-align: center;\'><span style=\'color: red;\'>❌ Needs Update to {online_version}</span></div>")':
            'grid[idx, 4] = table_cell(f"<span style=\'color: red;\'>❌ Needs Update to {online_version}</span>")',
        'grid[row_idx, 3] = widgets.HTML(f"<div style=\'text-align: center;\'>{online_latest_versions[main_folder][subfolder]}</div>")':
            'grid[row_idx, 3] = table_cell(online_latest_versions[main_folder][subfolder])',
        'grid[row_idx, 4] = widgets.HTML("<div style=\'text-align: center;\'>✅ Up-to-date</div>")':
            'grid[row_idx, 4] = table_cell("✅ Up-to-date")',
        'grid[row_idx, 5] = widgets.HTML("<div style=\'text-align: center;\'>-</div>")':
            'grid[row_idx, 5] = table_cell("-")',
        'grid[idx, 3] = widgets.HTML("<div style=\'text-align: center;\'>Not Found</div>")':
            'grid[idx, 3] = table_cell("Not Found")',
        'grid[idx, 4] = widgets.HTML("<div style=\'text-align: center;\'>❌ Missing Notebook</div>")':
            'grid[idx, 4] = table_cell("❌ Missing Notebook")',
        'grid[idx, 0] = widgets.HTML(f"<div style=\'text-align: center;\'>{subfolder}</div>")':
            'grid[idx, 0] = table_cell(subfolder)',
        'grid[idx, 4] = widgets.HTML("<div style=\'text-align: center;\'><span style=\'color: red;\'>❌ Missing Notebook</span></div>")':
            'grid[idx, 4] = table_cell("<span style=\'color: red;\'>❌ Missing Notebook</span>")',
        'grid[idx, 6] = widgets.HTML("<div style=\'text-align: center;\'>-</div>")':
            'grid[idx, 6] = table_cell("-")',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def _upgrade_welcome_source(source: str, relative_path: Path) -> str:
    normalized = source.replace("\r\n", "\n")

    load_marker = "def load_table(version_response, project_version_response, notebooks):\n"
    load_index = normalized.find(load_marker)
    if load_index < 0:
        raise ValueError(f"Unable to find load_table insertion point in {relative_path}")

    # Replace the 0.1.13 responsive helpers in place. If a repository already
    # received the final helper manually, this simply canonicalizes it.
    grid_config_index = normalized.find("GRID_COLUMN_WIDTHS = (")
    if 0 <= grid_config_index < load_index:
        normalized = normalized[:grid_config_index] + GRID_HELPERS + normalized[load_index:]
    else:
        normalized = normalized.replace(load_marker, GRID_HELPERS + load_marker, 1)

    # Keep the existing callback-output capture from 0.1.13, while tolerating a
    # manually edited repository where it may be absent.
    if "_widget_callback_output_cell_1 = widgets.Output()" not in normalized:
        marker = "# Define notebooks with their metadata\n"
        if marker not in normalized:
            raise ValueError(f"Unable to find widget-output insertion point in {relative_path}")
        widget_block = (
            "# Capture stdout, stderr, rich displays, figures, warnings, and tracebacks from widget callbacks.\n"
            "_widget_callback_output_cell_1 = widgets.Output()\n\n\n"
        )
        normalized = normalized.replace(marker, widget_block + marker, 1)

    # Keep the Open control inside its grid column.
    normalized = normalized.replace(
        'layout=widgets.Layout(width="auto")',
        'layout=widgets.Layout(width="auto", min_width="140px")',
    )
    normalized = normalized.replace(
        'layout=widgets.Layout(align_items="center")',
        'layout=widgets.Layout(width="100%", min_width="0", align_items="center", justify_content="center")',
    )

    # Preserve callback output if the local 0.1.13 file was partially edited.
    normalized = normalized.replace(
        "update_src_button.on_click(on_update_src_button_clicked)",
        "update_src_button.on_click(_widget_callback_output_cell_1.capture(clear_output=True, wait=True)(on_update_src_button_clicked))",
    )
    normalized = re.sub(
        r"update_button\.on_click\(button_update\((.*?)\)\)",
        r"update_button.on_click(_widget_callback_output_cell_1.capture(clear_output=True, wait=True)(button_update(\1)))",
        normalized,
    )

    # Replace the complete style block so every text cell, not only the
    # description, can wrap long underscore-separated scientific identifiers.
    css_marker = "    # Add custom CSS for styling\n"
    alt_css_marker = "    # Add custom CSS for responsive table styling\n"
    css_start = normalized.find(css_marker)
    if css_start < 0:
        css_start = normalized.find(alt_css_marker)
    header_start = normalized.find("    grid[0, 0] =", max(css_start, 0))
    if css_start < 0 or header_start < 0:
        raise ValueError(f"Unable to find Welcome CSS/header block in {relative_path}")
    normalized = normalized[:css_start] + WELCOME_CSS_BLOCK + normalized[header_start:]

    header_replacements = {
        'grid[0, 0] = widgets.HTML("<div class=\'grid-header\'><b>Notebook Name</b></div>")':
            'grid[0, 0] = table_header("Notebook Name")',
        'grid[0, 1] = widgets.HTML("<div class=\'grid-header\'><b>Notebook Topic</b></div>")':
            'grid[0, 1] = table_header("Notebook Topic")',
        'grid[0, 2] = widgets.HTML("<div class=\'grid-header\'><b>Description</b></div>")':
            'grid[0, 2] = table_header("Description")',
        'grid[0, 3] = widgets.HTML("<div class=\'grid-header\'><b>Local Version</b></div>")':
            'grid[0, 3] = table_header("Local Version")',
        'grid[0, 4] = widgets.HTML("<div class=\'grid-header\'><b>Status</b></div>")':
            'grid[0, 4] = table_header("Status")',
        'grid[0, 5] = widgets.HTML("<div class=\'grid-header\'><b>Update?</b></div>")':
            'grid[0, 5] = table_header("Update?")',
        'grid[0, 6] = widgets.HTML("<div class=\'grid-header\'><b>⬇️ Click to Open the Notebook ⬇️</b></div>")':
            'grid[0, 6] = table_header("Open Notebook")',
    }
    for old, new in header_replacements.items():
        normalized = normalized.replace(old, new)

    normalized = _replace_welcome_assignments(normalized)

    lines = normalized.splitlines(keepends=True)

    grid_create_index = next(
        (i for i, line in enumerate(lines) if line.strip() == "grid = GridspecLayout(1 + num_rows, 7)"),
        None,
    )
    if grid_create_index is None:
        raise ValueError(f"Unable to find Welcome table grid in {relative_path}")
    next_line = lines[grid_create_index + 1].strip() if grid_create_index + 1 < len(lines) else ""
    if next_line != "apply_grid_layout(grid)":
        indent = lines[grid_create_index][: len(lines[grid_create_index]) - len(lines[grid_create_index].lstrip())]
        lines.insert(grid_create_index + 1, f"{indent}apply_grid_layout(grid)\n")

    # GridspecLayout may recalculate its column template after replacing cells.
    callback_apply_present = False
    for index, line in enumerate(lines):
        if line.strip() != "apply_grid_layout(grid)":
            continue
        previous = lines[index - 1].strip() if index > 0 else ""
        if previous == 'grid[row_idx, 5] = table_cell("-")':
            callback_apply_present = True
            break
    if not callback_apply_present:
        for index, line in enumerate(lines):
            if line.strip() == 'grid[row_idx, 5] = table_cell("-")':
                indent = line[: len(line) - len(line.lstrip())]
                lines.insert(index + 1, f"{indent}apply_grid_layout(grid)\n")
                break
        else:
            raise ValueError(f"Unable to find notebook-update grid block in {relative_path}")

    display_index = next(
        (i for i, line in enumerate(lines) if line.strip() == "display(grid, grip_output)"),
        None,
    )
    if display_index is None:
        raise ValueError(f"Unable to find grid display in {relative_path}")
    previous = lines[display_index - 1].strip() if display_index > 0 else ""
    if previous != "apply_grid_layout(grid)":
        indent = lines[display_index][: len(lines[display_index]) - len(lines[display_index].lstrip())]
        lines.insert(display_index, f"{indent}apply_grid_layout(grid)\n")

    normalized = "".join(lines)

    if "display(_widget_callback_output_cell_1)" not in normalized:
        normalized = (
            normalized.rstrip("\n")
            + "\n\n# Display callback-generated output directly below this cell.\n"
            + "display(_widget_callback_output_cell_1)\n"
        )

    return normalized


def update_welcome_notebook(repo_root: Path, relative_path: Path) -> bool:
    path = repo_root / relative_path
    if not path.is_file():
        print(f"Skipping missing file: {relative_path}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    try:
        notebook = json.loads(original_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse {relative_path} as JSON") from exc

    notebook_changed = False
    found_grid = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_is_list = isinstance(source, list)
        src = "".join(source) if source_is_list else str(source)
        if "grid = GridspecLayout(1 + num_rows, 7)" not in src:
            continue

        found_grid = True
        normalized = _upgrade_welcome_source(src, relative_path)
        original_normalized = src.replace("\r\n", "\n")
        if normalized != original_normalized:
            cell["source"] = normalized.splitlines(keepends=True) if source_is_list else normalized
            notebook_changed = True
        break

    if not found_grid:
        print(f"Skipping {relative_path}: expected Welcome table grid was not found")
        return False
    if not notebook_changed:
        return False

    rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
    if original_text.endswith(("\n", "\r\n")):
        rendered += newline
    write_text(path, rendered)
    print(f"Updated wrapping Welcome table in {relative_path}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    for relative_path in WELCOME_PATHS:
        changed_any = update_welcome_notebook(repo_root, relative_path) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")

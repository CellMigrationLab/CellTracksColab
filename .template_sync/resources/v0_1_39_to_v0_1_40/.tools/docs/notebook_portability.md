# Write notebooks that run in both Google Colab and JupyterLab

A LabConstrictor app runs your notebooks in JupyterLab on your colleagues' computers, while a reviewer may open the same notebook in Google Colab without installing anything. These practices keep one notebook working in both.

## 1. Keep explanatory text visible when code is hidden

LabConstrictor apps often hide code cells (see [Hide code cells in notebooks](code_hiding.md)). Put `# @title` on the first line of a cell: Colab shows it as the cell's header, and JupyterLab still shows it as the first line of the cell:

```python
# @title Data input

# Rest of the cell...
```

For an explanation longer than a line, use a Markdown cell: it stays visible everywhere, whether the code is hidden or not.

## 2. Guard Colab-only setup code

Run Colab-only setup (mounting Google Drive, installing GPU-only packages, `!pip` installs) only on Colab, so the notebook does not fail on a local computer:

```python
import sys

if 'google.colab' in sys.modules:
    print("🚀 Detected Google Colab. Starting installation...")

    !pip install -q "cellpose[all]" tifffile
    !pip install -q instanseg-torch

    from google.colab import userdata
    from google.colab import drive

    drive.mount('/content/gdrive')

    print("✅ Colab setup done")
else:
    # Fallback for local environments
    print("⚠️ Not running in Colab. Please ensure dependencies are installed and data paths are set.")
```

## 3. Prefer cross-platform paths and storage

- Use `pathlib.Path` and **relative paths** when possible.
- Avoid hard-coding Colab paths like `/content` or `/content/gdrive` outside guarded blocks.
- For data files, include a **config cell** where users can set a local path or select a file.

## 4. Avoid Colab-only UI helpers

Colab forms ([Google's guide](https://colab.research.google.com/notebooks/forms.ipynb)) do not work in JupyterLab. Use ipywidgets or simple input prompts instead: they work in both.

For example, turn this Colab form:

```python
model_name = "Model A"  # @param ["Model A", "Model B", "Model C"]
print(f"You selected: {model_name}")
```

![Colab form example](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/colab_vs_local/colab.png)

into an ipywidgets dropdown:

**Option A** - with an observable function:

```python
import ipywidgets as widgets
from IPython.display import display

model_dropdown = widgets.Dropdown(
    options=["Model A", "Model B", "Model C"],
    description="Select a model:",
    style={'description_width': 'initial'}
)
output = widgets.HTML()

def on_model_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        output.value = f"You selected: {change['new']}"
model_dropdown.observe(on_model_change)

display(model_dropdown, output)
```

![ipywidgets dropdown example](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/colab_vs_local/local_option1.png)

**Option B** - With a button to confirm the selection:

```python
import ipywidgets as widgets
from IPython.display import display

model_dropdown = widgets.Dropdown(
    options=["Model A", "Model B", "Model C"],
    description="Select a model:",
    style={'description_width': 'initial'}
)
confirm_button = widgets.Button(description="Confirm Selection")
output = widgets.HTML()

def on_confirm_button_click(b):
    output.value = f"You selected: {model_dropdown.value}"
confirm_button.on_click(on_confirm_button_click)

display(model_dropdown, confirm_button, output)
```

![ipywidgets dropdown with button example](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/colab_vs_local/local_option2.png)

These examples need more code than a Colab form, but they work in both Colab and JupyterLab.

## 5. Clearly label runtime-specific instructions

When a step differs between the two, say so in a short Markdown note, for example:

> **Colab users:** Run the `Colab setup` cell first to mount Google Drive.
>
> **JupyterLab users:** Skip the `Colab setup` cell and configure the local data path instead.

## 6. Test in both environments before release

Before you publish a notebook:

1. Run it from start to end in **JupyterLab** (or local Jupyter), as the app will.
2. Run the same notebook in **Google Colab**.

---

<div align="center">

[🏠 Home](README.md)

</div>

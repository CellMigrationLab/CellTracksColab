# Adding External Code

If you want to use your own Python modules or scripts in Jupyter notebooks, follow these steps to upload them easily.

## 1. Prepare your external code

Organize your external code and follow Python packaging best practices. Set up the right folder structure, add an `__init__.py` file, and list any required dependencies.

**Suggested Directory Structure:**

```text
src/
|-- celltrackscolab/
|   |-- __init__.py
|   |-- my_script.py
|   |-- subpackage/
|       |-- __init__.py
```

### What goes in `__init__.py`?

Think of `__init__.py` as the file that tells Python “this folder is a package”.

 If you leave it empty, that is perfectly fine—the package will still import correctly. When you want to make notebooks feel friendlier, you can re-export helper functions so users can discover them more easily:

```python
# src/celltrackscolab/__init__.py
from .my_script import run_analysis, load_config

__all__ = ["run_analysis", "load_config"]
```

Each folder that you plan to import (for example, `src/subpackage/`) should also have its own `__init__.py`. Keep it empty unless you want to expose selected helpers from that subpackage.

If you need further readings on `__init__.py`, check out this quick [Geeks for Geeks tutorial](https://www.geeksforgeeks.org/python/what-is-__init__-py-file-in-python/) or the [Real Python guide](https://realpython.com/python-init-py/).
## 2. Upload your external code to the repository

Go to the src folder in your repository and upload your files or folders there.

![Upload external code GIF](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Upload_src.gif)

## 3. How to import into your notebook

Once uploaded, your external code will be available to your notebooks as a package. To use it, import it as shown in the examples below:

**Import the whole package:**
```python  
import celltrackscolab
```

If your folder under `src/` is called `celltools`, then `import celltools` will work because `src/celltools/__init__.py` exists.

**Import function:**
```python  
from celltrackscolab import my_script
```

**Import submodule:**
```python  
from celltrackscolab.subpackage import submodule1
```

---

<div align="center">

[← Previous](initialise_repository.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](notebook_requirements.md)


</div>

# Generate the requirements of your notebook

Every notebook you upload needs a `requirements.yaml` file: the Python version and the packages it uses, each with its exact version. The installer installs these packages on your users' computers.

The upload form accepts only exact versions: every dependency must have the form `package==version`. Version ranges such as `numpy>=1.24` can only appear in a `requirements.yaml` that you edit directly in your repository on GitHub, without the form (see [When notebooks ask for different versions of a package](#when-notebooks-ask-for-different-versions-of-a-package)).

There are two ways to make the file:

* **Option 1 (automated):** the Requirements Generator notebook reads the packages from the Python environment where your notebook runs. Use it when you start from scratch.
* **Option 2 (manual):** you write the file yourself and check it with the Requirements Validator notebook. Faster if you already have a `requirements.txt` with versions.

## What you need

- Your notebook, running without errors in a Python environment: on your computer (conda or virtualenv) or in Google Colab.
- For Option 1 on your computer: that same environment, with Jupyter. The generator must run in the environment your notebook uses, because it reads the package versions from it.

## Option 1: Automated extraction of requirements

### 1. Open the Requirements Generator notebook

Open it where your notebook runs:

* **On your computer:** download [Requirements_Generator.ipynb](../notebooks/Requirements_Generator.ipynb) (with the download button shown below) and open it in the environment you use for your notebook. If you have several conda environments, use the one your notebook runs in.

  ![Download Notebook](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Download_Notebook.png)

* **Google Colab:** open the [Requirements Generator in Colab](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Generator.ipynb). Colab runs each notebook in its own session, so the packages your notebook installs must be installed in the generator's session too: the generator finds your notebook's `pip install` commands and offers a cell that runs them.

### 2. Run the Requirements Generator

Run its cells in order and follow the instructions in the notebook: choose your notebook, add helper files your notebook uses if any, then generate the file. It saves `<notebook name>_requirements.yaml` next to the generator (in Colab it is also downloaded). You can upload it with that name: the upload form saves it as `requirements.yaml`.

## Option 2: Manual extraction of requirements

### 1. Create a `requirements.yaml` file

Start from the [example requirements.yaml file](https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/templates/requirements.yaml). It has three fields:

* **`dependencies`:** every package the notebook uses, one per line, each starting with `- ` and fixed to one version with `==` (`pandas==2.2.2`). Optional extras are allowed (`colabfold[alphafold,openmm]==1.6.3`).
* **`python_version`:** the Python version the notebook runs with (for example `3.11.1`). All notebooks of an app must use the same one.
* **`description`:** one sentence on what the notebook does.

```yaml
dependencies:
  - pandas==2.2.2
  - ipython==7.34.0
  - tqdm==4.67.1
description: This notebook is for analyzing cell migration data...
python_version: 3.11.1
```

### 2. Check it with the Requirements Validator notebook

* **Google Colab (recommended, nothing to install):** open the [Requirements Validator in Colab](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Validator.ipynb).
* **On your computer:** download [Requirements_Validator.ipynb](https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor/main/.tools/notebooks/Requirements_Validator.ipynb) and open it in any Python environment that has `ipywidgets`.

Follow the instructions in the notebook and give it your `requirements.yaml` when it asks.

## Check that it worked

- You have a `requirements.yaml` (or `<notebook name>_requirements.yaml`) file.
- Every line under `dependencies` has the form `package==version`, and `python_version` is the version your notebook runs with.

You are ready to [upload your notebook](notebook_upload.md).

## If something does not work

- **A package is missing from the generated file.** The generator finds only packages installed in the environment it runs in. Open it in the environment of your notebook, or add the package by hand with its version.
- **The upload form says "Dependency '...' is not in the correct format".** Fix that line to `package==version`.
- **The workflow stops after the upload with conflicting versions.** Two notebooks ask for different versions of a package: see [When notebooks ask for different versions of a package](#when-notebooks-ask-for-different-versions-of-a-package).

## What happens after you upload: the lock file

When a notebook and its `requirements.yaml` reach the repository, the workflow **Updates on notebook change** merges the requirements of every notebook into `requirements.txt`, plus `requirements_gpu.txt` and, if a notebook needs them, `requirements-windows.txt`, `requirements-linux.txt` or `requirements-macos.txt`.

The installer uses `requirements_gpu.txt` when it finds an NVIDIA GPU on Windows or Linux. In that file `tensorflow` becomes `tensorflow[and-cuda]` on Linux only: TensorFlow has had no Windows GPU build since version 2.10, and its CUDA packages exist only for Linux, so Windows keeps plain `tensorflow` (CPU).

The workflow then writes a **lock file** next to each requirements file (`requirements.lock.txt`, ...). A lock file lists every package the installer installs, including the packages your packages depend on, each with its exact version, for Windows, macOS and Linux. The installer installs the lock file, so every installation of a release gets the same versions.

* A lock file changes only when its requirements file, the Python version or the JupyterLab version changes. Pushing a notebook without new requirements does not move any version.
* To move every package to its newest version, delete the `requirements*.lock.txt` files and run the workflow by hand (**Actions > Updates on notebook change > Run workflow**).
* If the requirements cannot be locked, the workflow shows a warning and the installer installs the plain `requirements.txt` instead.

### When notebooks ask for different versions of a package

The merge never changes a notebook's requirement silently. Each requirement is kept as written in the notebook's `requirements.yaml`: a version range you wrote there directly on GitHub (`numpy>=1.24`) stays a range, and the requirements of several notebooks are combined (`numpy>=1.24,<2`). The workflow stops, and no requirements file is written, when:

* two notebooks pin a package to different versions (`numpy==1.26.4` and `numpy==2.0.0`);
* a pin does not meet another notebook's requirement (`numpy==2.0.0` and `numpy<2`);
* no version meets all the requirements (`numpy<2` and `numpy>=2.1`).

The log of the **Run merge script** step names the package, every notebook and its requirement. Then either:

* **make the notebooks agree:** change their `requirements.yaml` to a version that works for every notebook; or
* **decide explicitly** in `requirements_overrides.yaml`, at the top of the repository, which requirement every notebook gets, and why:

```yaml
overrides:
  numpy:
    use: "==2.0.0"
    reason: "Notebook A was re-run with numpy 2.0.0 and gives the same results; notebook B needs 2.0."
```

The decision and its reason are written at the top of the generated requirements files. Pushing `requirements_overrides.yaml` runs the workflow again. A release also stops while a conflict is not resolved.

Requirements with different platform markers (`; sys_platform == "win32"`) are kept as separate lines, and `NumPy`, `numpy` and `num_py` count as the same package, as for pip.

---

<div align="center">

[← Previous](accept_pull_request.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](notebook_upload.md)


</div>

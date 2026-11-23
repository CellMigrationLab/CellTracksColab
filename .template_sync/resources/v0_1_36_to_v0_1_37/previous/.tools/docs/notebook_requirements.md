# Generate the requirements of your notebook

You can get your Jupyter notebook’s requirements automatically with Option 1 or manually with Option 2.

* **Option 1 (Automated):** Recommended if you need to generate requirements from scratch.
* **Option 2 (Manual):** Recommended if you already have a versioned `requirements.txt` file, as it will be faster.

---

## Option 1: Automated Extraction of Requirements

### 1. Open the Requirements Generator Notebook

**⚠️ IMPORTANT:** You will need to choose the option based on the environment you used to run the target notebook whose requirements you want to extract. 

 * Did you ran your notebook locally on your machine (with Conda/VirtualEnv)? ➡️ Locally
 * Did you ran your notebook on Google Colab? ➡️ Online 

In both cases the Python environment that you use for the notebook, needs to run without errors. Knowing this please choose the option that best suits your case:

* **Locally:** Go to the [Requirements_Generator.ipynb](../notebooks/Requirements_Generator.ipynb) notebook, download it (clicking on the button indicated on the screenshot below ⬇️) and open it in your local Python environment (conda or virtualenv).
  > **Note**: When using it locally, the Python environment where you open the requirements generator notebook needs to be exactly the one you use to run the target notebook whose requirements you want to extract. For example, if you have multiple conda environments, you need to make sure that you use the same conda environment that you use to run the target notebook.
  
![Download Notebook](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Download_Notebook.png)
  
* **Google Colab:** Open the notebook directly in Google Colab by opening the [Requirements Generator in Colab](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Generator.ipynb).

### 2. Run the Requirements Generator

Follow the instructions provided inside the notebook to generate the `requirements.yaml` file. You will be prompted to provide the path to the notebook file you wish to analyse.

---

## Option 2: Manual Extraction of Requirements

### 1. Create a `requirements.yaml` File

You will need to create a `requirements.yaml` file similar to the [example requirements.yaml file](https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/templates/requirements.yaml). This file must contain the following three fields:

* **`dependencies`:** A list of all packages used in the notebook along with their versions. The list needs to be indexed with a hyphen (`-`) followed by a space before each package.
* **`python_version`:** The version of Python used in the notebook (e.g., `3.11.1`).
* **`description`:** A short description of the notebook and its purpose.

**Example File Content:**

```yaml
dependencies:
  - pandas==2.2.2
  - ipython==7.34.0
  - tqdm==4.67.1
description: This notebook is for analyzing cell migration data...
python_version: 3.11.1

```

### 2. Open the Requirements Validator Notebook

We recommend using **Google Colab** for this step, as you do not need a specific Python environment to run the validator.

* **Google Colab (Recommended):** Open the notebook directly by opening the [Requirements Validator in Colab](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Validator.ipynb).
* **Locally:** Download the [Requirements_Validator.ipynb](https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor/main/.tools/notebooks/Requirements_Validator.ipynb) and open it in any Python environment that has `ipywidgets` installed.

### 3. Run the Requirements Validator

Follow the instructions provided inside the notebook to validate your file. You will need to provide the path to your newly created `requirements.yaml` file when prompted.

---

## What happens next: the lock file

When you push a notebook with its `requirements.yaml`, the workflow **Updates on notebook change** merges the requirements of every notebook into `requirements.txt` (plus `requirements_gpu.txt` and, if a notebook needs them, `requirements-windows.txt`, `requirements-linux.txt` or `requirements-macos.txt`).

The installer uses `requirements_gpu.txt` when it finds an NVIDIA GPU on Windows or Linux. In that file `tensorflow` becomes `tensorflow[and-cuda]` on Linux only: TensorFlow has had no Windows GPU build since version 2.10, and its CUDA packages exist only for Linux, so Windows keeps plain `tensorflow` (CPU).

### When notebooks ask for different versions of a package

The merge never changes a notebook's requirement silently. Requirements are kept as written (`numpy>=1.24` stays `numpy>=1.24`), and the requirements of several notebooks are combined (`numpy>=1.24,<2`). The workflow stops, and no requirements file is written, when:

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

The decision and its reason are written at the top of the generated requirements files. Pushing `requirements_overrides.yaml` runs the workflow again. A release also stops while a conflict is not resolved. Requirements with different platform markers (`; sys_platform == "win32"`) are kept as separate lines, and `NumPy`, `numpy` and `num_py` count as the same package, as for pip.

It then writes a **lock file** next to each of them (`requirements.lock.txt`, ...). A lock file lists every package the installer installs, including the packages your packages depend on, each with its exact version, for Windows, macOS and Linux. The installer installs the lock file, so every installation of a release gets the same versions.

* A lock file changes only when its requirements file, the Python version or the JupyterLab version changes. Pushing a notebook without new requirements does not move any version.
* To move every package to its newest version, delete the `requirements*.lock.txt` files and run **Updates on notebook change** by hand (**Actions > Updates on notebook change > Run workflow**).
* If the requirements cannot be locked, the workflow shows a warning and the installer installs the plain `requirements.txt` instead.

---

<div align="center">

[← Previous](accept_pull_request.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](notebook_upload.md)


</div>

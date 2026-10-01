# Obtain the requirements of your Notebook

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

* **Locally:** Go to the [Requirements_Generator.ipynb](../notebooks/Requirements_Generator.ipynb) notebook, download it (clicking on the button indicated on the screenshot bellow ⬇️) and open it in your local Python environment (conda or virtualenv).
  > **Note**: When using it locally, the Python environment where you open the requirements generator notebook needs to be exactly the one you use to run the target notebook whose requirements you want to extract. For example, if you have multiple conda environments, you need to make sure that you use the same conda environment that you use to run the target notebook.
  
![Download Notebook](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Download_Notebook.png)
  
* **Google Colab:** Open the notebook directly in Google Colab by clicking [here](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Generator.ipynb).

### 2. Run the Requirements Generator

Follow the instructions provided inside the notebook to generate the `requirements.yaml` file. You will be prompted to provide the path to the notebook file you wish to analyse.

---

## Option 2: Manual Extraction of Requirements

### 1. Create a `requirements.yaml` File

You will need to create a `requirements.yaml` file similar to the [example requirements.yaml file](https://www.google.com/search?q=../templates/requirements.yaml). This file must contain the following three fields:

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

* **Google Colab (Recommended):** Open the notebook directly by clicking [here](https://colab.research.google.com/github/CellMigrationLab/LabConstrictor/blob/main/.tools/notebooks/Requirements_Validator.ipynb).
* **Locally:** Download the [Requirements_Validator.ipynb](https://raw.githubusercontent.com/CellMigrationLab/LabConstrictor/main/.tools/notebooks/Requirements_Validator.ipynb) and open it in any Python environment that has `ipywidgets` installed.

### 3. Run the Requirements Validator

Follow the instructions provided inside the notebook to validate your file. You will need to provide the path to your newly created `requirements.yaml` file when prompted.

---

<div align="center">

[← Previous](external_code_upload.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](notebook_upload.md)


</div>

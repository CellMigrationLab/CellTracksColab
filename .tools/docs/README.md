# LabConstrictor
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20758747.svg)](https://doi.org/10.5281/zenodo.20758747)
![LabConstrictor Comic](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/comic.png)

**LabConstrictor** turns your Jupyter notebooks into **installable desktop applications**. This lets users launch your work right away, without needing to use Python, pip, or any terminal commands.

This GitHub template handles everything for you. It packages your notebooks, builds installers for different platforms, and provides a simple, user-friendly **Welcome dashboard**. Your users can open your app with a double-click, while you keep a versioned, reproducible workflow.

## 🎯 Who is this for?

LabConstrictor is ideal for:
- Researchers sharing reproducible analysis pipelines or lab tools.
- Developers shipping interactive notebooks to non-technical users.
- Educators running workshops where setup time must be near zero.


## ✨ LabConstrictor's Features 

* **Easy Configuration**: Includes a web form to easily configure repo settings, manage dependencies, and brand your application without manual editing.
* **Cross-Platform**: Automatically builds `.exe` (Windows), `.pkg` (macOS), and `.sh` (Linux) installers.
* **Auto-Hide Code**: Code cells can be hidden, allowing users to see a clean, "app-like" interface, but still being able to reveal code if needed.
* **Dependency Guardrails**: automatic workflows merge and validate requirements, catching conflicts *before* you release.
* **Version Control**: Helper cells track versions and alert users when an update is available.

## 📸 The User Experience

![LabConstrictor Summary GIF](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Summary.gif)

## 🚀 Before Getting Started

Please read the [Before Getting Started Guide](./before_getting_started.md) to familiarise yourself with the LabConstrictor workflow and requirements.

## ⚡ Quick Start

Go from notebook to installer in minutes.

#### 1. [**Create a New Repository from this Template**](./create_repository.md)
#### 2. [**Initialise your repository**](./initialise_repository.md)
#### 3. [*(Optional)* **Upload external code**](./external_code_upload.md)
#### 4. [**Upload Your Notebooks**](./notebook_upload.md) 
#### 5. [**Create Executable Installers**](./executable_creation.md)
> **Requirements:** You only need a GitHub account and the Jupyter notebooks you want to distribute.

Need help writing notebooks that run well in both Colab and JupyterLab? See the [notebook portability guide](./notebook_portability.md).

## 🤝 Contributing

We welcome contributions, including bug fixes, UX improvements, or new packaging strategies.
Please read the [Contributing Guidelines](.github/CONTRIBUTING.md) before submitting PRs.

## 📢 Community

**Using LabConstrictor?**
We’d love to feature your project! [Open an issue](https://github.com/YOUR_USERNAME/LabConstrictor/issues) to let us know about your use case.

## 📚 Use Cases

People have already used LabConstrictor to package their tools in apps. Check out the [Use Cases](./use_cases.md) to see examples of what’s possible.

## ✍️ Citation
If you use LabConstrictor in your research, please cite the project to support its development:

Iván Hidalgo-Cenalmor, Marcela Xiomara Rivera Pineda, Bruno M. Saraiva, Ricardo Henriques, and Guillaume Jacquemet. **Packaging Jupyter notebooks as installable desktop apps using LabConstrictor. arXiv preprint 2026** DOI: https://doi.org/10.48550/arXiv.2603.10704

## GitHub management

- [Configure automatic template synchronization](template_synchronization.md)

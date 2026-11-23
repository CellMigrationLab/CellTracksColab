# Before getting started with LabConstrictor

What LabConstrictor does, and what you need before you create your repository.

## LabConstrictor in a nutshell

LabConstrictor is a GitHub template repository that turns your Jupyter notebooks into a desktop app that people install with one installer. GitHub Actions workflows in your repository do the building.

### Easy configuration

A web form sets up your repository (name, version, description, images) and uploads your notebooks with their requirements. You do not edit the workflows yourself.

### Cross-platform installers

Publishing a release builds installers with [conda constructor](https://github.com/conda/constructor): Windows (`.exe`), macOS (`.pkg`) and Linux (`.sh`), or only the systems you list under `distribution.platforms` in [`labconstrictor-app.yaml`](app_manifest.md#distribution-how-people-get-the-app).

### Environment management

Each notebook comes with a `requirements.yaml` file that lists its packages. The workflows merge the requirements of all notebooks and fix every package version, so each installation of a release gets the same packages. Helper notebooks generate and check these files for you.

### Version control

The first cell of each notebook checks its version. The app's Welcome notebook shows when a newer version of a notebook exists and updates it.

### User-friendly experience

The installed app opens JupyterLab with a Welcome notebook that lists your notebooks. Code cells can be hidden and run with a play button, so a notebook looks like a simple app. Your repository can be public or private; a private repository cannot be listed in the App Centre.

### Offline usage

Once installed, the app runs without internet access, unless your notebooks need it.

## LabConstrictor requirements

- **A GitHub account**, to create your repository from the template.
- **Your Jupyter notebooks.**
- **A Python environment where your notebooks run without errors**, on your computer or in Google Colab. You use it to generate the requirements files. You can also write these files by hand.
- **A GitHub personal access token** (classic, with the **repo** scope), which the web forms use to open pull requests in your repository: see [Generate a GitHub personal access token](personal_access_token.md).
- **Optional: images** (logo, banner, header) to brand the installer: see [Customise the installer images](installer_images.md).

Next: [Create your repository](create_repository.md).


---

<div align="center">

← Previous &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](create_repository.md)


</div>

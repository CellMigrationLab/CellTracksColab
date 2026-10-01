# 📋 Before Getting Started with LabConstrictor

Welcome to LabConstrictor! This guide will walk you through the basics and show you what you need to get started.

## 🎯 LabConstrictor in a Nutshell

LabConstrictor is a template repository that lets you turn Jupyter notebooks into desktop apps. It handles the setup for you via automated GitHub Actions. Here are its main features:

### ⚙️ Easy Configuration

LabConstrictor has a web form that makes it easy to set up your repository, add your branding, and upload or update your notebooks and their dependencies.

### 🖥️ Cross-Platform Installers

Automatic workflows create installers for Windows (.exe), macOS (.pkg), and Linux (.sh). This makes it simple for users on any platform to install and run your notebooks. The installers are created using [Conda Constructor](https://github.com/conda/constructor).

### 📦 Environment Management

LabConstrictor uses `requirements.yaml` files to manage dependencies. Automatic workflows ensure your notebooks run consistently across different systems. LabConstrictor also includes helper notebooks for generating and validating requirement files.

### 📈 Version Control

LabConstrictor includes helper cells that track notebook versions and notify users when updates are available. This way, everyone can use the latest features and fixes.

### ✨ User-Friendly Experience

Installers handle the setup process, allowing users to launch your notebooks with a desktop app. The installers take care of the setup, so users can open your notebooks with a desktop app. The app shows a Welcome Dashboard where users can check versions and pick notebooks. Notebooks have hidden code and a play button to run cells, giving a clean, app-like feel. private or public, giving you full control over who can access and use your notebooks.

### 📱 Offline Usage

If your notebooks don’t need internet access, users can run the desktop apps offline after they install them.

## ✅ LabConstrictor Requirements

To get started with LabConstrictor, you only need the following:

- **🐙 GitHub Account:** You’ll need a GitHub account to make a new repository from the LabConstrictor template. 
- **📒 Jupyter Notebooks:** Have your Jupyter notebooks ready to package into desktop apps.
- **🐍 Python Environment:** You’ll need a Python environment (local or online) where your notebooks run without errors. This is needed to create and check requirement files. Alternatively, you can create the requirement files yourself, but this may be prone to errors.
- **(Optional) 🖼️ Images, logo and icons:** for branding your executable installer. We recommend following the [images guidelines](./installer_images.md) for further information and guidance.



## 🔄 One-time update setup

After creating your repository, complete the [automatic template synchronization setup](template_synchronization.md). This allows future template improvements to be delivered as pull requests, including necessary GitHub Actions workflow updates.

---

<div align="center">

← Previous &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](create_repository.md)


</div>

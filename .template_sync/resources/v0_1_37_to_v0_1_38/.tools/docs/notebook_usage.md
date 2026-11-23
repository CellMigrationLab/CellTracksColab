# Open and use an app

You installed a LabConstrictor app ([how to install](install.md)). From here: open it, try the app's example, then use your own data.

Want to see the idea first? The [interactive demo](https://cellmigrationlab.github.io/LabConstrictor-Site/demo/) lets you try a notebook, napari and Fiji with example data before you install anything.

## Open the app

Start the app from the Start Menu (Windows), the Applications folder (macOS) or your applications menu (Linux): search for its name and click its icon.

A terminal window opens first. It keeps JupyterLab running: leave it open while you work. After a few seconds, JupyterLab opens in your web browser with the **Welcome notebook**:

![Welcome Notebook](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Welcome_Notebook.png)

## Open a notebook from the Welcome notebook

Run the cell of the Welcome notebook. It shows a table of the app's notebooks, with their version and whether a newer version exists. Click **Open the notebook** in the last column to open a notebook in a new tab.

If the app's repository on GitHub is private, the Welcome notebook asks for a GitHub personal access token to check for new versions: see [Generate a GitHub personal access token](personal_access_token.md).

## Run a notebook

Follow the instructions written in the notebook, and run its cells from top to bottom. Notebooks made with LabConstrictor often hide their code. Run a cell with the play button next to it, if there is one, or select the cell and click the play button in the top toolbar:

![Toolbar Play Button](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Notebook_Play_Button.png)

The **Show/Hide Code** button in the top toolbar shows or hides the code of every cell:

![Show Hide Code Button](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Notebook_ShowHide_Button.png)

New to JupyterLab? See [this JupyterLab guide](https://github.com/CellMigrationLab/docs-hub/blob/main/docs/JupyterLab/JupyterLab_Guide.md).

## Try the app's example

Look for an example in the app's documentation: the **Documentation** or **README** link on its page in the [App Centre](https://cellmigrationlab.github.io/LabConstrictor-Site/apps/). If it has one, run it first and compare your result with the one it describes. What an app does, and what its notebooks expect, is described there.

## Use it in napari, Fiji or QuPath

Some apps also have **tools**: analyses you run from napari, Fiji or QuPath on the image you have open, with a form for the settings and the results drawn on the image. The app's page in the [App Centre](https://cellmigrationlab.github.io/LabConstrictor-Site/apps/) shows which programs it works in. Install the plugin for your program once: [Use apps inside napari, Fiji and QuPath](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/image-software/) explains how.

## Keep your results

Each notebook decides where it writes its results. Note the folder it names before you close JupyterLab, and copy what you want to keep.

## Use it without installing

Many apps also run in Google Colab. The app's page in the [App Centre](https://cellmigrationlab.github.io/LabConstrictor-Site/apps/) lists its Colab notebooks and, when it has one, a pip command.

## Update a notebook

When a newer version of a notebook is published, the Welcome notebook shows **Update** next to it.

1. Close the notebook's tab first. Otherwise JupyterLab can save the old copy over the new one.
2. Click **Update**, then confirm with **Back up and update**.

Your current copy, with your changes and outputs, is kept as `<notebook>.backup-<date>.ipynb` in the same folder. If the download fails, your notebook is not changed.

## Close the app

When you have finished, stop JupyterLab with **File > Shut Down** (or click the terminal window and press `Ctrl+C`). The terminal then asks you to press `Enter`, and closes.

## If something does not work

- Windows and macOS warn about installers that are not code-signed yet, and both have a way to continue: see [Install an app](install.md).
- [Troubleshooting an installed app](troubleshooting_installed_app.md) covers problems with an app on your computer.
- [Find your problem](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/help/) lists what you may see and where each is explained.
- Open an issue in the app's repository: the **Report a problem** link on its page in the App Centre, when the repository has an issue tracker.


---


<div align="center">

[← Previous](download_executable.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Next →


</div>

# Use notebooks after installation

How to open an installed app, run its notebooks and update them.

## Open the app

To open the application, go to the Start Menu (Windows), the Applications folder (macOS) or your applications menu (Linux). Search for the application name and click its icon to start it.

A terminal window opens first. It keeps JupyterLab running: leave it open while you work. After a few seconds, JupyterLab opens in your web browser with the Welcome notebook:

![Welcome Notebook](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Welcome_Notebook.png)

## Open a notebook

Run the cell of the Welcome notebook. It shows a table of the app's notebooks, with their version and whether a newer version exists. Click **Open the notebook** in the last column to open a notebook in a new tab.

If the app's repository on GitHub is private, the Welcome notebook asks for a GitHub personal access token to check for new versions: see [Generate a GitHub personal access token](personal_access_token.md).

## Run a notebook

Notebooks made with LabConstrictor can hide their code. Run each cell with its play button, or select the cell and click the play button in the top toolbar. The **Show/Hide Code** button in the toolbar shows the code. See [Hide code cells in notebooks](code_hiding.md) for pictures. If you are new to JupyterLab, see [this JupyterLab guide](https://github.com/CellMigrationLab/docs-hub/blob/main/docs/JupyterLab/JupyterLab_Guide.md).

## Update a notebook

When a newer version of a notebook is published, the Welcome notebook shows **Update** next to it.

1. Close the notebook's tab first. Otherwise JupyterLab can save the old copy over the new one.
2. Click **Update**, then confirm with **Back up and update**.

Your current copy, with your changes and outputs, is kept as `<notebook>.backup-<date>.ipynb` in the same folder. If the download fails, your notebook is not changed.

## Close the app

Keep the terminal window open until you have finished working with your notebooks. When you have, stop the Jupyter Lab session with **File > Shut Down** in Jupyter Lab (or click on the terminal window and press `Ctrl+C`). The terminal then asks you to press `Enter`, and closes.

## If something does not work

See [Troubleshooting an installed app](troubleshooting_installed_app.md).


---


<div align="center">

[← Previous](download_executable.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Next →


</div>

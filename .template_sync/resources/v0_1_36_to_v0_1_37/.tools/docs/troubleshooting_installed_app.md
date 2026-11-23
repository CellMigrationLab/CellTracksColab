# Troubleshooting an installed app

Problems with an app on your computer, after you installed it. For problems with your repository on GitHub, see [Troubleshooting repository setup](troubleshooting.md).

## Where to find the logs

Two files in the app's install folder (the folder you chose when you installed it) record what happened:

- `menuinst_debug.log`: what the installer did.
- `launcher_debug.log`: what happened when you opened the app. If the app cannot start, a message names this file.

Send them along when you ask the app's developer for help.

## App still appears in the Windows Installed apps list after uninstalling

On Windows, after you uninstall the app from `Settings > Apps > Installed apps` as explained in [Uninstall an app](install.md#uninstall-an-app), the app can stay in the list and refuse to uninstall again, with a message like this:

![Uninstall error message](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/troubleshooting/Not_Uninstall.png)

Uninstall it from the Control Panel instead:

1. Open the Control Panel. You can search for it in the Windows search bar.
2. Go to `Programs > Programs and Features`.
3. Find the app in the list, right-click it, and select `Uninstall`.
4. A message may say something like "An error occurred while trying to uninstall xxx. It might have already been uninstalled." Click `Yes`.

The app is then removed from the list.

## Windows asks for administrator rights for python.exe while uninstalling

Uninstalling an app made with template 0.1.20 or earlier may show a User Account Control prompt: "Do you want to allow this app from an unknown publisher to make changes to your device? python.exe", with the program location `"<app folder>\python.exe" -m conda.core.initialize ...`.

1. Click **No**. The prompt is not needed.
2. The uninstaller then shows a "Failed to clean AutoRun" error with Abort, Retry and Ignore: click **Ignore**.

The uninstall then finishes normally, and nothing outside the app folder is changed. **Yes** and **Retry** do not help: the same error comes back.

Do not click **Abort**, even though it is the default button: it stops the uninstall halfway and leaves the app folder on disk. If that happened, run `Uninstall-<AppName>.exe` in the app folder again.

Installers built with template 0.1.21 or later do not have this problem ([details](migration_0_1_21.md)).

## The notebook fails with "No module named ..." although the installation finished

The first cell that imports the app fails with `ModuleNotFoundError: No module named '<app>'`. This happens with apps made with template 0.1.23 or earlier when someone once ran `python -m ipykernel install --user` on the computer (common with Anaconda or pip setups). Jupyter then has a personal "python3" kernel, and the app's notebooks run in that other Python instead of the app's own.

Install a release made with template 0.1.24 or later: it always uses the app's own Python. Until then, you can delete the personal kernel folder, but the other Jupyter installations on the computer that use it lose it too:

- Linux: `~/.local/share/jupyter/kernels/python3`
- macOS: `~/Library/Jupyter/kernels/python3`
- Windows: `%APPDATA%\jupyter\kernels\python3`

## My changes to a notebook are gone after an update

The **Update** button of the Welcome notebook keeps your copy before it replaces a notebook: look for `<notebook>.backup-<date>.ipynb` in the same folder and open it in JupyterLab. Apps made with template 0.1.30 or earlier replace the notebook without keeping a copy.

## JupyterLab does not start because Windows reports an ASN.1 certificate error

A broken certificate in the Windows certificate store can stop JupyterLab from starting. The app's launcher works around it: it first uses the system's certificates and, if one of them cannot be read, falls back to a verified certificate bundle. Certificate checks are never switched off. `launcher_debug.log` says which one was used.

If your organisation uses its own certificate authority, save its certificates as a PEM file and set one of these environment variables to the path of that file before you install or open the app:

- `LABCONSTRICTOR_CA_BUNDLE`, for every LabConstrictor app;
- `<APP_NAME>_CA_BUNDLE`, for one app: the app's name in capitals, with spaces and punctuation replaced by `_` (for the app "My Project", `MY_PROJECT_CA_BUNDLE`).

To test the launcher without opening JupyterLab, run this in a command prompt (replace `<install folder>` and `<app>`):

```text
<install folder>\python.exe <install folder>\<app>\launch_jupyter.py --self-test
```

# Install an app

Download the installer for your system from the app's page in the [App Centre](https://cellmigrationlab.github.io/LabConstrictor-Site/apps/) (or from the app's GitHub release), then follow the steps for your system below. You install an app once; it can take several minutes.

> **Check where it comes from.** The installers are not code-signed yet, so your system will warn you. Download only from the app's own GitHub release (the address starts with `https://github.com/` followed by the app's owner and repository, as shown on the app's page), and run it only if you trust that owner. A listing in the App Centre checks that the files come from the app's repository; it is not a security audit.

## Before you start

- **Internet connection.** The installer downloads the app's Python packages while it installs.
- **Install path.** The install path must contain no space: the installer refuses a path with a space. Use a path such as `C:\LabConstrictor\<app>` on Windows or `~/labconstrictor/<app>` on macOS and Linux.

## Install on Windows

1. Download the `.exe` file and double-click it.
2. If Windows shows "Windows protected your PC", click "More info" and then "Run anyway". The installers are not code-signed yet.

   ![Windows protected your PC: click More info, then Run anyway](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Windows_Protected_your_PC.png)

3. Follow the on-screen prompts. A command prompt window opens during the installation: this is normal, do not close it.
4. Launch the app from the Start Menu or your desktop.

## Install on a Mac

1. Choose your file: open the Apple menu, then About This Mac. If the Chip or Processor says Intel, download the Intel installer; if it says M1, M2 or similar, download the Apple Silicon one.
2. Double-click the `.pkg` file. If macOS refuses to open it, go to System Settings, Privacy & Security, scroll down to the message about the package and click "Open Anyway". The button appears only after you have tried to open the file once.
3. Follow the prompts (we recommend "Install only for me"). Launch the app from your Applications folder.

## Install on Linux

1. Download the `.sh` file.
2. Open a terminal in the download folder and run `bash` followed by the name of the file you downloaded, for example: `bash MyApp-0.1.0-Linux-x86_64.sh`
3. Follow the prompts, then launch the app from your applications menu.

## Check that it worked

Open the app. A terminal window opens, then JupyterLab opens in your web browser with the app's Welcome notebook. What to do next: [Open and use an app](notebook_usage.md).

## If something does not work

- The installer writes what it did to `menuinst_debug.log` in the install folder.
- See [Troubleshooting an installed app](troubleshooting_installed_app.md), or start from [Find your problem](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/help/).

## Update an installed app

- **macOS and Windows.** On macOS and Windows, update an app by [uninstalling the old version](#uninstall-an-app) first and then installing the new one: the installer does not install over an existing install. On macOS, if you already ran the new installer over the old version and it stopped or reported a failure at the end, move only the old app shortcut (for example `~/Applications/MyApp.app`) to the Bin, keep the app's folder in your home folder, and run the installer again.
- **Linux.** On Linux, running the installer with `-u` updates in place only if the previous install finished; otherwise delete `<prefix>/pkgs` or uninstall the app and install it again.

## Uninstall an app

Uninstalling deletes the app's install folder and everything in it: move anything you saved there to another folder first.

- **Windows.** Open Settings > Apps > Installed apps, find the app, click the three dots next to it and choose Uninstall. You can also double-click `Uninstall-<AppName>.exe` in the app's install folder. If the app is still listed afterwards, see [Troubleshooting an installed app](troubleshooting_installed_app.md).
- **macOS and Linux.** Open a terminal and run the `uninstall.sh` script of the app's install folder (the folder you chose when you installed it), for example `bash ~/labconstrictor/MyApp/uninstall.sh`. It removes the app's shortcut (from the Applications folder of your home folder on macOS, from your applications menu on Linux), takes the app out of the tools list of napari, Fiji and QuPath, and deletes the install folder.

## Where the app is listed

If the app ships tools for napari, Fiji or QuPath, the installer registers them with LabConstrictor Tools, with the `hosts:` line of the app's `labconstrictor-app.yaml`. A program lists the app only if that line names it: with `hosts: [notebook, napari]` the app is listed in napari, and not in Fiji or QuPath. Without a `hosts:` line, every program lists the app, and `menuinst_debug.log` in the install folder says so (`'<app>' declares no hosts, so it is listed in every host; ...`).

To run the tools, install the plugin for your image software once: see [Use apps inside napari, Fiji and QuPath](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/image-software/).

## Next

Open the app and start working: [Open and use an app](notebook_usage.md).

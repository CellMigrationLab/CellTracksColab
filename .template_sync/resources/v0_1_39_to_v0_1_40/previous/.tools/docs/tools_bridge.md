# Expose your app's tools to napari, Fiji and QuPath

Turn functions of your app's Python package into **tools** that people run in napari, Fiji and QuPath, and from the command line. Each tool gets a form made from its function signature. You declare the tools in your repository; the installer registers them on the user's computer. Apps that declare no tools are not affected.

To expose your app's tools:

1. Put your Python code in `src/` (see [Upload external code](external_code_upload.md)).
2. Declare the tools in a `<your_package>_lc_tools` module, as shown below.
3. Name the programs your tools work in under `hosts:` in `labconstrictor-app.yaml`.
4. Test the tools, then publish a release.

How to write a tool, with its inputs, outputs, form layout and tests, is in [Writing tools](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/writing-tools/). What the people who install your app do to run the tools is in [Use apps inside napari, Fiji and QuPath](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/image-software/).

## What you need

- An app repository made from this template, with your Python package in `src/<your_package>/`.
- Python functions with type annotations that you want to offer as tools.

## Declare your tools

1. Add a module named `<your_package>_lc_tools` next to your package under `src/` (so `src/mytool_lc_tools/__init__.py` for
   the package `mytool`). The installer registers tools only from a module with exactly this name. Declare tools in it with typed
   functions, and import your science code **inside** the functions:

   ```python
   from typing import Annotated
   from labconstrictor_tools import Image, ImageOut, Min, tool

   @tool("Gaussian blur")
   def blur(image: Image, sigma: Annotated[float, Min(0)] = 2.0) -> ImageOut:
       from mytool.filters import gaussian
       return gaussian(image, sigma)
   ```

   Keep the module light: it is imported to read the tool list, so heavy imports at the top slow every start.
2. If you can, import `labconstrictor_tools` only in `<your_package>_lc_tools`, or inside functions. The rest of your package then also
   works where LabConstrictor Tools is not installed: a notebook on Google Colab, a copy of the package used elsewhere, or an installation
   where Tools failed to install.
3. Name the programs your tools work in under `hosts:` in `labconstrictor-app.yaml`: see [Describe your app for the App Centre](app_manifest.md).
   napari, Fiji and QuPath list your app only if `hosts:` names them.
4. Test your tools with `labconstrictor-tools check` and `labconstrictor-tools test`, as
   [Writing tools](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/writing-tools/) shows.

## What the installer does

When someone installs your app, the installer:

1. installs your package;
2. if your package has a `<your_package>_lc_tools` module, installs LabConstrictor Tools;
3. checks that your package can be imported;
4. registers the app (`labconstrictor-tools register ...`) with the version from the `version:` line of `construct.yaml` and the hosts of
   `labconstrictor-app.yaml`.

When the app is uninstalled, it is unregistered.

If step 2 or 4 fails, the installer writes a `WARNING` to `menuinst_debug.log` in the install folder, skips the registration and finishes
the installation. The exception is a package that imports `labconstrictor_tools` at the top: then the import check of step 3 fails.

The installer installs one fixed release of LabConstrictor Tools: the tag in `LC_TOOLS_PIN` of
[app/bash_bat_scripts/post_install.sh](../../app/bash_bat_scripts/post_install.sh) and
[app/bash_bat_scripts/post_install.bat](../../app/bash_bat_scripts/post_install.bat). Template updates move it. To install Tools from
another source (a wheel, a git URL or a mirror), set the environment variable `LC_TOOLS_SPEC` when the installer runs; it replaces the pin.
How Tools releases are made and pinned is in
[Operations](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/tools-operations/#pinning-tools-in-app-installers).

## Check that it worked

After you install a release of your app:

- `menuinst_debug.log` in the install folder says `Tools registered`.
- `labconstrictor-tools list` shows your app and its tools.
- Your app appears in the programs you named under `hosts:`, once their plugin is installed.

## If something does not work

- `menuinst_debug.log` in the app's install folder says what the installer did with your tools.
- `labconstrictor-tools doctor` checks the Tools installation, and `labconstrictor-tools logs` shows its log.
- Other problems and their fixes: [Operations](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/tools-operations/).

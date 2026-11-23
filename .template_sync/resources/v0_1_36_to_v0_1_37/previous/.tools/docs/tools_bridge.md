# Expose your app's tools to napari, Fiji and QuPath (optional)

If your repository ships Python code (see [Upload external code](external_code_upload.md)), you can declare some of its functions
as **tools**. After installation, people can run them in napari, Fiji and QuPath and from the command line, with forms made from the
function signatures. Nothing changes for apps that do not declare tools.

This page covers what the template does: where you declare the tools and what the installer does with them. Two other guides cover the rest:

- [Writing tools](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/writing-tools/): the inputs
  and outputs a tool can have, how to lay out its form, what each program supports, and how to test a tool.
- [Use apps inside napari, Fiji and QuPath](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/image-software/): what the people
  who install your app do to run its tools, with the install steps of the
  [napari plugin](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/napari/), the
  [Fiji plugin](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/fiji/) and the
  [QuPath extension](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/qupath/).

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
2. Import `labconstrictor_tools` only in `<your_package>_lc_tools`, or inside functions, if you can. The rest of your package then also works
   where LabConstrictor Tools is not installed: a notebook run on Google Colab, a copy of the package used elsewhere, or an installation where
   Tools failed to install. An import at the top of your package does not stop the installation, because the installer installs Tools before
   it checks that your package can be imported.
3. Name the programs your tools work in under `hosts:` in `labconstrictor-app.yaml`: see [Describe your app for the App Centre](app_manifest.md).
4. Test your tools before you publish a release, as [Writing tools](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/writing-tools/)
   shows (`labconstrictor-tools check` and `labconstrictor-tools test`).

## What the installer does

The installer's `post_install` step:

1. installs your package;
2. if your package has a `<your_package>_lc_tools` module, installs LabConstrictor Tools;
3. checks that your package can be imported;
4. registers the app (`labconstrictor-tools register ...`) with the version from the `version:` line of `construct.yaml` and the hosts of
   `labconstrictor-app.yaml`.

When the app is uninstalled, `pre_uninstall` unregisters it.

A failure in step 2 or 4 is written as a `WARNING` to `menuinst_debug.log` in the install folder, the registration is skipped, and the
installation goes on. The one exception is a package that needs Tools to be imported: then the import check of step 3 fails, as it should.

The Windows step (`post_install.bat`) runs in the template's automated tests on a Windows machine, and people have installed apps on Windows
and macOS from a clean system.

### Which LabConstrictor Tools the installer installs

LabConstrictor Tools is not on PyPI yet. The installer downloads the GitHub archive of one release tag of LabConstrictor-Tools, a plain zip
file, so the user's computer does not need git. The tag is pinned: two installers built from the same app release install the same Tools,
and a change to Tools does not reach new installations until the pin is moved on purpose. Template updates move the pin in your app.

The tag is the value of `LC_TOOLS_PIN` in both scripts:
[app/bash_bat_scripts/post_install.sh](../../app/bash_bat_scripts/post_install.sh) (`LC_TOOLS_PIN="..."`) and
[app/bash_bat_scripts/post_install.bat](../../app/bash_bat_scripts/post_install.bat) (`SET "LC_TOOLS_PIN=..."`).

To install Tools from somewhere else, set the environment variable `LC_TOOLS_SPEC` when the installer runs: a wheel, a git URL, a mirror, or
the name `labconstrictor-tools` once it is on PyPI. It replaces the pin completely.

**Move the pin (template maintainers).** Create and test the tag in LabConstrictor-Tools (the procedure is in its `docs/OPERATIONS.md`).
Change `LC_TOOLS_PIN` in both scripts of the template, and in the copies under `.template_sync/resources/` of the latest migration if it has
them (the test `test_resources_are_identical_to_the_template_files` fails when they differ). Then write a migration that changes the string in
the apps' scripts, and bump `template_version`.

## If something does not work

- `menuinst_debug.log` in the app's install folder says what the installer did with your tools.
- `labconstrictor-tools doctor` checks the Tools installation, and `labconstrictor-tools logs` shows its log.

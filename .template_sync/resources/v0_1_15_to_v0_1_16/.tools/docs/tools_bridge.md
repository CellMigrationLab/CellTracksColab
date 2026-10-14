# Expose your app's tools to Napari and Fiji (optional, experimental)

If your repository ships Python code (see [Upload external code](external_code_upload.md)), you can declare some of its functions
as **tools**. After installation they appear in a generic **Napari** widget and a generic **Fiji** command, and work from the
command line, with forms generated automatically from the function signatures. Nothing changes for apps that do not declare tools.

1. Add a module named `<your_package>_lc_tools` next to your package under `src/` (so `src/mytool_lc_tools/__init__.py` for
   the package `mytool`). Declare tools in it with typed functions, importing your science code **inside** the functions:

   ```python
   from typing import Annotated
   from labconstrictor_tools import Image, ImageOut, Min, tool

   @tool("Gaussian blur")
   def blur(image: Image, sigma: Annotated[float, Min(0)] = 2.0) -> ImageOut:
       from mytool.filters import gaussian
       return gaussian(image, sigma)
   ```

   Keep the module light: it is imported to read the tool list, so heavy imports at the top slow every start.
2. The installer (`post_install`) detects the module, installs `labconstrictor-tools` and registers the app
   (`labconstrictor-tools register ...`). `pre_uninstall` unregisters it. A failure in this step is logged to
   `menuinst_debug.log` and never fails the installation.
   **Where `labconstrictor-tools` comes from.** It is not on PyPI yet, so by default the installer downloads the GitHub archive of a **pinned release tag**,
   `https://github.com/CellMigrationLab/LabConstrictor-Tools/archive/refs/tags/v0.2.0-rc.1.zip` (a plain zip: git is not needed on the user's computer).
   Set the environment variable `LC_TOOLS_SPEC` when the installer runs to install it from somewhere else (a wheel, a git URL, a mirror, or the
   name `labconstrictor-tools` once it is published on PyPI); it overrides the pin completely.
   **Pinned (important).** Two installers built from the same app release get the same version of the tools package, and a change to Tools `main`
   does not reach new installations until the pin is moved on purpose.
   **How to move the pin.** The tag name is written in one place per script: the variable `LC_TOOLS_PIN` (`LC_TOOLS_PIN="v0.2.0-rc.1"` in
   `app/bash_bat_scripts/post_install.sh`, `SET "LC_TOOLS_PIN=v0.2.0-rc.1"` in `post_install.bat`). To use a new release: create and test the tag in
   LabConstrictor-Tools (procedure in its `docs/OPERATIONS.md`), change that string in both scripts and in the copies under
   `.template_sync/resources/` of the latest migration (the test `test_resources_are_identical_to_the_template_files` fails when they differ),
   update the tag in this paragraph, and bump `template_version` so apps receive the change. Apps already migrated need a new migration to change the
   string in their own scripts.
3. Test your tools before releasing: `labconstrictor-tools check --module mytool_lc_tools` and
   `labconstrictor-tools test --module mytool_lc_tools --cases lc_tests/cases.json`.
4. Install the Napari plugin (`napari-labconstrictor`) or the Fiji plugin (`labconstrictor-fiji`) once per machine. Neither is on PyPI or a Fiji update site yet: see the READMEs of
   [napari-labconstrictor](https://github.com/CellMigrationLab/napari-labconstrictor) and [LabConstrictor-Fiji](https://github.com/CellMigrationLab/LabConstrictor-Fiji) for the install commands.
5. Optional: make a long form readable and rich with markers (the authoring guide in the Tools repository lists them all):
   - layout: `Group("...")`, `Advanced()`, `Collapsed()`, `EnabledWhen(...)`;
   - input widgets: `Widget("slider")` for a bounded number and `Widget("radio")` for a short list of choices;
   - `PickChannel()` to let the person choose one channel of a multi-channel image (your function receives that channel as a 2D array);
   - `ChoicesFrom("other_tool")` to fill a dropdown from another tool of yours (for example the available devices);
   - `RegionOf("image")` (optional labels input) to run on the selection in the host (Napari Shapes layer, Fiji ROI, QuPath annotations);
   - outputs: `ImageOut`, `LabelsOut`, `TableOut`, `Scalars`, `MessageOut`, `PointsOut` and `ShapesOut` (polygons: Napari shapes layer, Fiji ROIs, QuPath annotations);
   - say "nothing found" with `ToolError("no_match", ...)`.
   Napari, Fiji and QuPath do not support every feature in the same way: the table "What each host does with each feature" in `docs/HOST_FEATURES.md` of the Tools repository is the reference.
Documentation and source: https://github.com/CellMigrationLab/LabConstrictor-Tools.
The registered version is read from the `version:` line of `construct.yaml`. The Windows hook (`post_install.bat`) has not yet been run on a real Windows computer; if you can help, follow the
[human test protocol](https://github.com/CellMigrationLab/LabConstrictor-Tools/blob/main/docs/HUMAN_TEST_PROTOCOL.md).
Troubleshooting: `labconstrictor-tools doctor`, `labconstrictor-tools logs`, and `~/.labconstrictor/logs/labconstrictor.log`.

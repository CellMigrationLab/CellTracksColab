#!/bin/bash
set -e
echo "Running pre_uninstall" 
# best effort: a failure must not stop the uninstall, but it is reported (a stale entry can stay in the Napari/Fiji list)
if ! "BASE_PATH/bin/python" -m labconstrictor_tools unregister --name "CellTracksColab" --prefix "BASE_PATH" > /dev/null 2>&1; then
    echo "WARNING: could not remove CellTracksColab from the LabConstrictor tools registry; Napari/Fiji may still list it (run: \"BASE_PATH/bin/python\" -m labconstrictor_tools unregister --name CellTracksColab)." >&2
fi
"BASE_PATH/bin/python" -c "from menuinst.api import remove; import os; remove(os.path.join(r'BASE_PATH', 'CellTracksColab', 'notebook_launcher.json'))"

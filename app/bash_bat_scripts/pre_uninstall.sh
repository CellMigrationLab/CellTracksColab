#!/bin/bash
set -e
echo "Running pre_uninstall" 
"BASE_PATH/bin/python" -m labconstrictor_tools unregister --name "CellTracksColab" > /dev/null 2>&1 || true
"BASE_PATH/bin/python" -c "from menuinst.api import remove; import os; remove(os.path.join(r'BASE_PATH', 'CellTracksColab', 'notebook_launcher.json'))"

#!/bin/bash
set -e
echo "Running pre_uninstall" 
"BASE_PATH/bin/python" -c "from menuinst.api import remove; import os; remove(os.path.join(r'BASE_PATH', 'CellTracksColab', 'notebook_launcher.json'))"

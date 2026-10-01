#!/bin/bash
set -euo pipefail

LOG_FILE="$PREFIX/menuinst_debug.log"
PYTHON_EXE="$PREFIX/bin/python"
PROJECT_ROOT="$PREFIX/PROJECT_NAME"
BASE_REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
GPU_REQUIREMENTS="$PROJECT_ROOT/requirements_gpu.txt"
SELECTED_REQUIREMENTS="$BASE_REQUIREMENTS"

echo "Running post_install" > "$LOG_FILE"

fail() {
    echo "ERROR: $1" | tee -a "$LOG_FILE" >&2
    exit 1
}

[ -x "$PYTHON_EXE" ] || fail "Bundled Python executable was not found at $PYTHON_EXE."
[ -f "$PROJECT_ROOT/launch_jupyter.py" ] || fail "TLS-resilient launcher was not found at $PROJECT_ROOT/launch_jupyter.py."
[ -f "$BASE_REQUIREMENTS" ] || fail "Base requirements file was not found at $BASE_REQUIREMENTS."

# pip 24.2+ uses system certificates by default. Opt out for installer pip
# operations and use an explicit verified CA bundle instead.
export PIP_USE_DEPRECATED=legacy-certs
CA_BUNDLE="$("$PYTHON_EXE" "$PROJECT_ROOT/launch_jupyter.py" --print-ca-bundle 2>> "$LOG_FILE")"
[ -s "$CA_BUNDLE" ] || fail "No valid verified CA bundle could be selected: $CA_BUNDLE"

export PIP_CERT="$CA_BUNDLE"
export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
export CURL_CA_BUNDLE="$CA_BUNDLE"
echo "Using verified CA bundle for installer network operations: $CA_BUNDLE" >> "$LOG_FILE"

if [ -f "$GPU_REQUIREMENTS" ]; then
    if [[ "${OSTYPE:-}" == "darwin"* ]]; then
        echo "macOS detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$LOG_FILE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected, installing GPU requirements from $GPU_REQUIREMENTS" >> "$LOG_FILE"
        SELECTED_REQUIREMENTS="$GPU_REQUIREMENTS"
    else
        echo "NVIDIA GPU not detected, installing CPU requirements from $BASE_REQUIREMENTS" >> "$LOG_FILE"
    fi
else
    echo "GPU requirements file not found, installing CPU requirements from $BASE_REQUIREMENTS" >> "$LOG_FILE"
fi

echo "Installing requirements from $SELECTED_REQUIREMENTS" >> "$LOG_FILE"
"$PYTHON_EXE" -m pip install -r "$SELECTED_REQUIREMENTS" >> "$LOG_FILE" 2>&1

if [[ "${OSTYPE:-}" == "darwin"* ]]; then
    echo "Detected macOS platform" >> "$LOG_FILE"
    if [ -f "$PROJECT_ROOT/requirements-macos.txt" ]; then
        "$PYTHON_EXE" -m pip install -r "$PROJECT_ROOT/requirements-macos.txt" >> "$LOG_FILE" 2>&1
    fi
elif [[ "${OSTYPE:-}" == "linux-gnu"* ]]; then
    echo "Detected Linux platform" >> "$LOG_FILE"
    if [ -f "$PROJECT_ROOT/requirements-linux.txt" ]; then
        "$PYTHON_EXE" -m pip install -r "$PROJECT_ROOT/requirements-linux.txt" >> "$LOG_FILE" 2>&1
    fi
else
    echo "Unknown platform: ${OSTYPE:-unset}" >> "$LOG_FILE"
fi

# External Python code is optional. Verify the generated package only when the
# constructor bundled setup.py and src/.
if [ -f "$PROJECT_ROOT/setup.py" ]; then
    echo "Found setup.py, installing PROJECT_NAME package locally without build isolation" >> "$LOG_FILE"
    "$PYTHON_EXE" -m pip install --no-deps --no-build-isolation "$PROJECT_ROOT" >> "$LOG_FILE" 2>&1
    "$PYTHON_EXE" -c "import PYTHON_PROJ_NAME; print('PROJECT_NAME import successful:', PYTHON_PROJ_NAME.__file__)" >> "$LOG_FILE" 2>&1
else
    echo "No setup.py detected; this project does not bundle an optional Python package." >> "$LOG_FILE"
fi

"$PYTHON_EXE" "$PROJECT_ROOT/include_path.py" --path "$PREFIX" --files "$PROJECT_ROOT/notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "$LOG_FILE" 2>&1
"$PYTHON_EXE" "$PROJECT_ROOT/include_path.py" --path "$PREFIX" --files "$PREFIX/pre_uninstall.sh" --keyword "BASE_PATH" >> "$LOG_FILE" 2>&1
"$PYTHON_EXE" "$PROJECT_ROOT/include_path.py" --path "$PREFIX" --files "$PREFIX/uninstall.sh" --keyword "BASE_PATH" >> "$LOG_FILE" 2>&1
"$PYTHON_EXE" "$PROJECT_ROOT/hide_code_cells.py" "$PROJECT_ROOT" >> "$LOG_FILE" 2>&1

# Keep conda verification enabled while avoiding the system certificate store
# that failed. Conda accepts a CA bundle path for ssl_verify.
if ! "$PYTHON_EXE" -m conda config --system --set ssl_verify "$CA_BUNDLE" >> "$LOG_FILE" 2>&1; then
    echo "WARNING: Could not configure conda with the selected CA bundle. The launcher still has its own verified fallback." >> "$LOG_FILE"
fi

# Do not create the application shortcut unless launcher preflight succeeds.
"$PYTHON_EXE" "$PROJECT_ROOT/launch_jupyter.py" --self-test >> "$LOG_FILE" 2>&1
echo "Launcher preflight completed successfully." >> "$LOG_FILE"

"$PYTHON_EXE" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "$LOG_FILE" 2>&1
"$PYTHON_EXE" -c "from menuinst.api import install; import os; print(install(os.path.join('$PREFIX', 'PROJECT_NAME', 'notebook_launcher.json')))" >> "$LOG_FILE" 2>&1

echo "Post-install completed successfully." >> "$LOG_FILE"

if [ -t 0 ]; then
    echo
    read -rp "Press Enter to close the installer..." _
fi

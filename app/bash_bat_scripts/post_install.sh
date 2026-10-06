#!/bin/bash
set -euo pipefail

LOG_FILE="$PREFIX/menuinst_debug.log"
PYTHON_EXE="$PREFIX/bin/python"
PROJECT_ROOT="$PREFIX/CellTracksColab"
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
    echo "Found setup.py, installing CellTracksColab package locally without build isolation" >> "$LOG_FILE"
    "$PYTHON_EXE" -m pip install --no-deps --no-build-isolation "$PROJECT_ROOT" >> "$LOG_FILE" 2>&1
    "$PYTHON_EXE" -c "import celltracks; print('CellTracksColab import successful:', celltracks.__file__)" >> "$LOG_FILE" 2>&1
else
    echo "No setup.py detected; this project does not bundle an optional Python package." >> "$LOG_FILE"
fi


# --- Optional: expose the app's tools to Napari, Fiji and the command line (LabConstrictor tools bridge) -------------------
# If the app's package ships a module named <package>_lc_tools, install labconstrictor-tools and register that module.
# This step must never fail the installation. LC_TOOLS_SPEC can point to another source (wheel, git URL, mirror).
# Default source: the GitHub archive of labconstrictor-tools (a plain zip: no git needed on the user's computer), because the package
# is not on PyPI yet. Once it is, use "labconstrictor-tools" here.
LC_TOOLS_MODULE="celltracks_lc_tools"
if [ -f "$PROJECT_ROOT/setup.py" ] && "$PYTHON_EXE" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$LC_TOOLS_MODULE') else 1)" >> "$LOG_FILE" 2>&1; then
    lc_note() { echo "$*" >> "$LOG_FILE" || :; }  # a full or unwritable log must not fail the installation
    lc_note "Found $LC_TOOLS_MODULE: registering the tools of CellTracksColab for Napari and Fiji"
    LC_APP_VERSION="$(sed -n 's/^version: *//p' "$PROJECT_ROOT/construct.yaml" 2>/dev/null | head -1 | tr -d "\"'\r" || true)"
    if [ -z "$LC_APP_VERSION" ]; then
        lc_note "WARNING: no version: line at the start of a line in construct.yaml - registering the tools with version 0."
        LC_APP_VERSION=0
    fi
    if "$PYTHON_EXE" -m pip install "${LC_TOOLS_SPEC:-https://github.com/CellMigrationLab/LabConstrictor-Tools/archive/refs/heads/main.zip}" >> "$LOG_FILE" 2>&1 \
        && "$PYTHON_EXE" -m labconstrictor_tools register --name "CellTracksColab" --prefix "$PREFIX" \
            --module "$LC_TOOLS_MODULE" --version "$LC_APP_VERSION" --display-name "CellTracksColab" >> "$LOG_FILE" 2>&1; then
        lc_note "Tools registered (labconstrictor-tools list shows them)."
    else
        lc_note "WARNING: tool registration failed - see the pip and register output above in this file; CellTracksColab itself is installed."
    fi
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
"$PYTHON_EXE" -c "from menuinst.api import install; import os; print(install(os.path.join('$PREFIX', 'CellTracksColab', 'notebook_launcher.json')))" >> "$LOG_FILE" 2>&1

echo "Post-install completed successfully." >> "$LOG_FILE"

if [ -t 0 ]; then
    echo
    read -rp "Press Enter to close the installer..." _
fi

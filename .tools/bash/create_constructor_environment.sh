#!/usr/bin/env bash
# Create the conda environment used for constructor builds.
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  if [ -n "${CONDA:-}" ] && [ -f "${CONDA}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA}/etc/profile.d/conda.sh"
  elif [ -n "${MAMBA_ROOT_PREFIX:-}" ] && [ -f "${MAMBA_ROOT_PREFIX}/etc/profile.d/micromamba.sh" ]; then
    # shellcheck disable=SC1090
    source "${MAMBA_ROOT_PREFIX}/etc/profile.d/micromamba.sh"
  else
    echo "conda command is not available in PATH and CONDA is unset." >&2
    exit 1
  fi
fi

# conda-standalone (the installer's _conda.exe) older than 26.9.3 writes package
# records without paths_data when constructor disables conda's safety checks.
# conda 26.9 and later in the installed app then cannot find the Windows
# launcher stub, and `conda init --reverse` in the uninstaller asks for admin
# rights (conda/conda#16856, fixed in conda 26.9.3).
conda create -y -n "${CONSTRUCTOR_ENV:?}" -c conda-forge -c defaults python=3.11 constructor "conda-standalone>=26.9.3"

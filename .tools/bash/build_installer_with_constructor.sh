#!/usr/bin/env bash
# Build the platform-specific installer using the prepared constructor env.
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

mkdir -p "${OUTPUT_DIR:?}"
conda run -n "${CONSTRUCTOR_ENV:?}" constructor . --output-dir "${OUTPUT_DIR}"

#!/bin/bash
# Test script to validate conda environment creation and pip requirements installation
# This script creates a test conda environment from environment.yaml and installs pip requirements

set -e  # Exit on any error
set -u  # Exit on undefined variables

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if is_truthy "${SKIP_CONDA_ENV_TEST:-}"; then
    echo "SKIP_CONDA_ENV_TEST is enabled; skipping conda environment validation."
    exit 0
fi

echo "=========================================="
echo "Testing Conda Environment Creation"
echo "=========================================="

# Define test environment name
TEST_ENV_NAME="potential-potato-test-env"

# Clean up any existing test environment
echo "Cleaning up any existing test environment..."
mamba env remove -n "$TEST_ENV_NAME" -y 2>/dev/null || true

# Create conda environment from environment.yaml
echo ""
echo "Creating conda environment from environment.yaml..."
if [ ! -f "environment.yaml" ]; then
    echo "ERROR: environment.yaml not found!"
    exit 1
fi

mamba env create -n "$TEST_ENV_NAME" -f environment.yaml -y

# Activate the environment
echo ""
echo "Activating test environment..."
# Note: In CI, conda/mamba environments need to be activated via conda run
# or by sourcing the activation script

# Test pip installation of requirements.txt
echo ""
echo "Installing requirements from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    mamba env remove -n "$TEST_ENV_NAME" -y
    exit 1
fi

# Check if requirements.txt is not empty
if [ ! -s "requirements.txt" ]; then
    echo "WARNING: requirements.txt is empty, skipping pip install"
else
    echo "Installing packages with pip..."
    mamba run -n "$TEST_ENV_NAME" pip install --prefer-binary -r requirements.txt
fi

# Verify installation by checking Python and key packages
echo ""
echo "Verifying installation..."
echo "Python version:"
mamba run -n "$TEST_ENV_NAME" python --version

echo ""
echo "Installed pip packages (sample):"
mamba run -n "$TEST_ENV_NAME" pip list | head -n 20

# Clean up test environment
echo ""
echo "Cleaning up test environment..."
mamba env remove -n "$TEST_ENV_NAME" -y

echo ""
echo "=========================================="
echo "✓ Conda environment test completed successfully!"
echo "=========================================="

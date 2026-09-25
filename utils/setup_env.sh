#!/usr/bin/env bash
# Environment setup helper script for Linux and macOS

set -e

echo "======================================================================"
echo "  Multimodal Uncertainty Quantification - Environment Setup"
echo "======================================================================"

OS_TYPE=$(uname -s)
echo "Detected OS: $OS_TYPE"

# 1. Install virtualenv & dev dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment in .venv..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing package in editable mode with development dependencies..."
pip install --upgrade pip
pip install -e ".[dev,all]"

# 2. Pre-generate synthetic benchmark image
echo "Generating local benchmark image (utils/cat_on_sofa.jpg)..."
python utils/generate_test_image.py

# 3. Check VLM Server Readiness
echo "Checking local VLM server availability..."
python utils/setup_local_vlm.py --check-only || true

echo "======================================================================"
echo "  Setup Complete! You can now run tests with: pytest"
echo "======================================================================"

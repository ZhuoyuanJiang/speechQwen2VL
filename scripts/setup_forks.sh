#!/bin/bash
# =============================================================================
# Setup forked packages for Speech-Qwen2VL
# =============================================================================
# This script clones the forked transformers and qwen-vl-utils repos
# and installs them in editable mode.
#
# IMPORTANT: Run this AFTER `pip install -r requirements.txt`
# These must be installed last to prevent other packages from overwriting them.
#
# Usage:
#   bash scripts/setup_forks.sh
# =============================================================================

set -e  # Exit on error

FORKS_DIR="$(cd "$(dirname "$0")/.." && pwd)/forks"
mkdir -p "$FORKS_DIR"

echo "=== Setting up forked packages in $FORKS_DIR ==="

# --- Fork 1: transformers ---
# Base: huggingface/transformers at commit 0f9c9088 (=4.56.0.dev0)
# Branch: speech-qwen2vl (contains audio modifications)
TRANSFORMERS_REPO="https://github.com/ZhuoyuanJiang/transformers.git"
TRANSFORMERS_BRANCH="speech-qwen2vl"

if [ -d "$FORKS_DIR/transformers" ]; then
    echo "transformers fork already cloned, pulling latest..."
    cd "$FORKS_DIR/transformers"
    git fetch origin
    git checkout "$TRANSFORMERS_BRANCH"
    git pull origin "$TRANSFORMERS_BRANCH"
else
    echo "Cloning transformers fork..."
    cd "$FORKS_DIR"
    git clone "$TRANSFORMERS_REPO"
    cd transformers
    git checkout -b "$TRANSFORMERS_BRANCH" 2>/dev/null || git checkout "$TRANSFORMERS_BRANCH"
fi

echo "Installing transformers from fork (editable)..."
pip install -e "$FORKS_DIR/transformers"

# --- Fork 2: qwen-vl-utils (inside Qwen2-VL repo) ---
# Base: QwenLM/Qwen2-VL
# Branch: speech-qwen2vl (contains fetch_audio modifications)
QWEN_VL_REPO="https://github.com/ZhuoyuanJiang/Qwen2-VL.git"
QWEN_VL_BRANCH="speech-qwen2vl"

if [ -d "$FORKS_DIR/Qwen2-VL" ]; then
    echo "qwen-vl-utils fork already cloned, pulling latest..."
    cd "$FORKS_DIR/Qwen2-VL"
    git fetch origin
    git checkout "$QWEN_VL_BRANCH"
    git pull origin "$QWEN_VL_BRANCH"
else
    echo "Cloning qwen-vl-utils fork..."
    cd "$FORKS_DIR"
    git clone "$QWEN_VL_REPO"
    cd Qwen2-VL
    git checkout -b "$QWEN_VL_BRANCH" 2>/dev/null || git checkout "$QWEN_VL_BRANCH"
fi

echo "Installing qwen-vl-utils from fork (editable)..."
pip install -e "$FORKS_DIR/Qwen2-VL"

# --- Verify installations ---
echo ""
echo "=== Verification ==="
python -c "import transformers; print(f'transformers: {transformers.__version__} from {transformers.__file__}')"
python -c "import qwen_vl_utils; print(f'qwen_vl_utils from {qwen_vl_utils.__file__}')"
echo ""
echo "=== Setup complete ==="
echo "Forks installed in: $FORKS_DIR"
echo "Both packages are in editable mode — changes to fork files take effect immediately."

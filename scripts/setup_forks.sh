#!/bin/bash
# =============================================================================
# Setup forked packages for Speech-Qwen2VL
# =============================================================================
# This script clones the forked transformers and qwen-vl-utils repos
# and installs them in editable mode.
#
# This script assumes the following forked repos already exist on GitHub:
#   - ZhuoyuanJiang/transformers
#   - ZhuoyuanJiang/Qwen3-VL
# (These repos already contain all audio-related modifications on their
#  speech-qwen2vl branches.)
#
# Prerequisites (must be completed before running this script):
#   1. conda env create -f environment.yml && conda activate speech_qwen2vl
#   2. pip install -r requirements.txt
#      (forks must be installed LAST to prevent other packages from overwriting them)
#
# What this script does:
#   1. Clones the forked repos into the forks/ directory
#   2. Checks out the speech-qwen2vl branch (which contains audio modifications)
#   3. Installs both packages in editable mode (pip install -e)
#
# Usage:
#   bash scripts/setup_forks.sh
#
# After running, verify with:
#   python -c "import transformers; print(transformers.__version__)"
#   # Expected: 4.56.0.dev0
# =============================================================================

set -e  # Exit on error

FORKS_DIR="$(cd "$(dirname "$0")/.." && pwd)/forks"
mkdir -p "$FORKS_DIR"

echo "=== Setting up forked packages in $FORKS_DIR ==="

# --- Fork 1: transformers ---
# Base/Original repo: huggingface/transformers (the HuggingFace transformers library)
#   at commit 0f9c9088 (= version 4.56.0.dev0)
# Our fork: ZhuoyuanJiang/transformers
#   The fork copies the entire repo (including the latest v5.x on main).
#   We don't use main — instead we created a speech-qwen2vl branch from
#   an older commit (0f9c9088) so our branch stays at 4.56.0.dev0.
# Branch: speech-qwen2vl, created from huggingface/transformers commit 0f9c9088
#   Why this commit: the original repo has moved to v5.x with breaking changes.
#   Commit 0f9c9088 (4.56.0.dev0) is proven compatible with our stack
#   (torch 2.4.1, peft 0.17.1, trl 0.22.0.dev0).
# Modified files on this branch:
#   - src/transformers/models/qwen2_vl/processing_qwen2_vl.py (audio token expansion)
#
# REPRODUCIBILITY: We pin to a specific COMMIT, not just the branch, so future
# pushes to speech-qwen2vl don't silently change what gets installed. To bump,
# update TRANSFORMERS_COMMIT below and the corresponding line in README.md.
TRANSFORMERS_REPO="https://github.com/ZhuoyuanJiang/transformers.git"
TRANSFORMERS_BRANCH="speech-qwen2vl"
TRANSFORMERS_COMMIT="934129b7701e7607facb39f286afc6bc4cc657df"

if [ -d "$FORKS_DIR/transformers" ]; then
    echo "transformers fork already cloned, fetching..."
    cd "$FORKS_DIR/transformers"
    git fetch origin
else
    echo "Cloning transformers fork..."
    cd "$FORKS_DIR"
    git clone "$TRANSFORMERS_REPO"
    cd transformers
fi
# Detached-HEAD checkout to the pinned commit (reproducible).
echo "Checking out transformers commit $TRANSFORMERS_COMMIT (branch tip: $TRANSFORMERS_BRANCH)"
git checkout --detach "$TRANSFORMERS_COMMIT"

echo "Installing transformers from fork (editable)..."
pip install -e "$FORKS_DIR/transformers"

# --- Fork 2: qwen-vl-utils (inside Qwen2-VL repo) ---
# Base/Original repo: QwenLM/Qwen2-VL (Qwen's vision-language model utilities)
#   at commit 9658872 (qwen-vl-utils version 0.0.14)
#   Note: the repo was renamed to QwenLM/Qwen3-VL on GitHub, but the qwen-vl-utils
#   code inside is still Qwen2-VL compatible (no Qwen3 changes as of this commit).
# Our fork: ZhuoyuanJiang/Qwen3-VL (inherits the renamed repo name)
# Branch: speech-qwen2vl, created from QwenLM/Qwen3-VL commit 9658872
# Modified files on this branch:
#   - qwen-vl-utils/src/qwen_vl_utils/vision_process.py (fetch_audio, process_vision_info)
#
# REPRODUCIBILITY: pin to a specific COMMIT (see transformers section above).
QWEN_VL_REPO="https://github.com/ZhuoyuanJiang/Qwen3-VL.git"
QWEN_VL_BRANCH="speech-qwen2vl"
QWEN_VL_COMMIT="56b0756a768cc3b01cba45b01c1bc3c8cb74ea3f"

if [ -d "$FORKS_DIR/Qwen2-VL" ]; then
    echo "qwen-vl-utils fork already cloned, fetching..."
    cd "$FORKS_DIR/Qwen2-VL"
    git fetch origin
else
    echo "Cloning qwen-vl-utils fork..."
    cd "$FORKS_DIR"
    # Syntax: git clone <repo_url> <local_folder_name>
    # The repo is named Qwen3-VL on GitHub, so git clone would create a Qwen3-VL/ folder
    # by default. We pass "Qwen2-VL" as the second argument to name it Qwen2-VL/ instead,
    # since the code inside is still Qwen2-VL and we want our project references consistent.
    git clone "$QWEN_VL_REPO" Qwen2-VL
    cd Qwen2-VL
fi
echo "Checking out qwen-vl-utils commit $QWEN_VL_COMMIT (branch tip: $QWEN_VL_BRANCH)"
git checkout --detach "$QWEN_VL_COMMIT"

echo "Installing qwen-vl-utils from fork (editable)..."
# The installable package (pyproject.toml) is in the qwen-vl-utils/ subdirectory,
# not at the repo root. That's why we need the /qwen-vl-utils suffix here.
pip install -e "$FORKS_DIR/Qwen2-VL/qwen-vl-utils"

# --- Verify installations ---
echo ""
echo "=== Verification ==="
python -c "import transformers; print(f'transformers: {transformers.__version__} from {transformers.__file__}')"
python -c "import qwen_vl_utils; print(f'qwen_vl_utils from {qwen_vl_utils.__file__}')"
echo ""
echo "=== Setup complete ==="
echo "Forks installed in: $FORKS_DIR"
echo "Both packages are in editable mode — changes to fork files take effect immediately."

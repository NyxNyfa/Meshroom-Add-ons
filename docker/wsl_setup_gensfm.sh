#!/usr/bin/env bash
# ============================================================================
# Automated WSL2 Ubuntu Setup Script for dotb1ue/GenSfm + Low-VRAM Offloading
# Run inside WSL2 Ubuntu:
#   bash /mnt/d/Meshroom/mrGenSfM/docker/wsl_setup_gensfm.sh
# ============================================================================
set -euo pipefail

GENSFM_DIR="${1:-/mnt/d/Meshroom/GenSfm}"

echo "[mrGenSfM] Setting up GenSfM environment in WSL2 Ubuntu (Target repo: ${GENSFM_DIR})..."

if ! command -v conda >/dev/null 2>&1; then
    echo "[mrGenSfM] Installing Miniforge3 in WSL2..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
    export PATH="$HOME/miniforge3/bin:$PATH"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^gensfm "; then
    conda create -y -n gensfm python=3.12
fi

conda activate gensfm
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

cd "$GENSFM_DIR"
git submodule update --init --recursive
. ./setup.sh --basic --spconv --mipgaussian --nvdiffrast

echo "[mrGenSfM] WSL2 GenSfM environment 'gensfm' is ready!"

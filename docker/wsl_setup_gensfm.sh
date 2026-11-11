#!/usr/bin/env bash
# ============================================================================
# Automated WSL2 Ubuntu Setup Script for dotb1ue/GenSfm + Low-VRAM Offloading
# Run inside WSL2 Ubuntu:
#   bash /mnt/d/Meshroom/mrGenSfM/docker/wsl_setup_gensfm.sh
# ============================================================================
set -euo pipefail

GENSFM_DIR="${1:-/mnt/d/Meshroom/GenSfm}"

echo "[mrGenSfM] Setting up GenSfM environment in WSL2 Ubuntu (Target repo: ${GENSFM_DIR})..."

# Use physical disk storage (~/tmp with 900+ GB) instead of WSL2 default RAM tmpfs (/tmp with only 3.9 GB)
# to prevent "OSError: [Errno 28] No space left on device" during large PyTorch wheel downloads
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
if ! grep -q "TMPDIR=" "$HOME/.bashrc" 2>/dev/null; then
    echo "export TMPDIR=\"\$HOME/tmp\"" >> "$HOME/.bashrc"
fi

# Detect existing conda in standard locations if not currently in PATH
if ! command -v conda >/dev/null 2>&1; then
    for candidate in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -x "$candidate/bin/conda" ]; then
            export PATH="$candidate/bin:$PATH"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[mrGenSfM] Installing Miniforge3 in WSL2..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -u -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
    export PATH="$HOME/miniforge3/bin:$PATH"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Ensure conda is automatically available in future bash sessions
if ! grep -q "etc/profile.d/conda.sh" "$HOME/.bashrc" 2>/dev/null; then
    echo "source \"$CONDA_BASE/etc/profile.d/conda.sh\"" >> "$HOME/.bashrc"
fi

if ! conda env list | grep -q "^gensfm "; then
    conda create -y -n gensfm python=3.12
fi

# Configure CUDA Toolkit paths if available on WSL2
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    if ! grep -q "CUDA_HOME=" "$HOME/.bashrc" 2>/dev/null; then
        echo 'export CUDA_HOME="/usr/local/cuda"' >> "$HOME/.bashrc"
        echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> "$HOME/.bashrc"
        echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"' >> "$HOME/.bashrc"
    fi
fi

conda activate gensfm
pip install --no-cache-dir ninja
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

cd "$GENSFM_DIR"
# Strip Windows CRLF line endings from shell scripts if cloned on Windows filesystem
sed -i 's/\r$//' ./setup.sh 2>/dev/null || true
sed -i 's/\r$//' ./dataset_toolkits/setup.sh 2>/dev/null || true

# Configure compatible GCC toolchain and single-job parallelism to prevent compiler mismatch & OOM crashes
export CUDAHOSTCXX="/home/nyx/miniforge3/envs/gensfm/bin/g++"
export CC="/home/nyx/miniforge3/envs/gensfm/bin/gcc"
export CXX="/home/nyx/miniforge3/envs/gensfm/bin/g++"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=1

git submodule update --init --recursive
. ./setup.sh --basic --spconv --mipgaussian --nvdiffrast

echo "[mrGenSfM] WSL2 GenSfM environment 'gensfm' is ready!"

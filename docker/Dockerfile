# ============================================================================
# mrGenSfM Dockerfile (Linux + CUDA 12.8 + PyTorch 2.9.1 + GenSfM Extensions)
# Compatible with Windows Docker Desktop (WSL2 backend) & Linux NVIDIA Container Toolkit
# ============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ATTN_BACKEND=sdpa \
    ATTN_BACKEND_WITH_ATTN_BIAS=sdpa \
    SPCONV_ALGO=native \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Python 3.12)
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/dotb1ue/GenSfm.git /workspace/GenSfm

WORKDIR /workspace/GenSfm
RUN conda install -y python=3.12 && \
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir spconv-cu126==2.3.8

RUN bash -c ". ./setup.sh --nvdiffrast --mipgaussian"

ENTRYPOINT ["python3"]

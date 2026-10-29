#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Speech Enhancement Benchmark Setup ==="

# 1. System tool checks
echo "Checking system prerequisites..."
command -v /usr/bin/python3.11 >/dev/null 2>&1 || { echo "ERROR: /usr/bin/python3.11 not found."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found."; exit 1; }
command -v git-lfs >/dev/null 2>&1 || { echo "ERROR: git-lfs not found."; exit 1; }

# 2. Check / install uv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-./.uv-cache}"
mkdir -p "$UV_CACHE_DIR"

# 3. Create directories
mkdir -p src/bench adapters config scripts tests logs results \
         models/hf models/torch models/cache models/asteroid \
         third_party inputs/pairs references

# 4. Virtual environments setup
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

# envs/eval
if [ ! -f "envs/eval/bin/python" ]; then
    echo "Creating envs/eval..."
    uv venv --python /usr/bin/python3.11 envs/eval
    uv pip install --python envs/eval/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/eval/bin/python numpy==1.26.4 pesq soxr soundfile scipy pyyaml tqdm pytest
fi

# envs/speechbrain
if [ ! -f "envs/speechbrain/bin/python" ]; then
    echo "Creating envs/speechbrain..."
    uv venv --python /usr/bin/python3.11 envs/speechbrain
    uv pip install --python envs/speechbrain/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/speechbrain/bin/python speechbrain soundfile soxr
fi

# envs/dfn
if [ ! -f "envs/dfn/bin/python" ]; then
    echo "Creating envs/dfn..."
    uv venv --python /usr/bin/python3.11 envs/dfn
    uv pip install --python envs/dfn/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/dfn/bin/python deepfilternet soundfile soxr
fi

# envs/denoiser
if [ ! -f "envs/denoiser/bin/python" ]; then
    echo "Creating envs/denoiser..."
    uv venv --python /usr/bin/python3.11 envs/denoiser
    uv pip install --python envs/denoiser/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/denoiser/bin/python denoiser soundfile soxr
fi

# envs/asteroid
if [ ! -f "envs/asteroid/bin/python" ]; then
    echo "Creating envs/asteroid..."
    uv venv --python /usr/bin/python3.11 envs/asteroid
    uv pip install --python envs/asteroid/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/asteroid/bin/python "asteroid @ git+https://github.com/asteroid-team/asteroid.git" soundfile soxr
fi

# envs/cleanunet
if [ ! -f "envs/cleanunet/bin/python" ]; then
    echo "Creating envs/cleanunet..."
    uv venv --python /usr/bin/python3.11 envs/cleanunet
    uv pip install --python envs/cleanunet/bin/python torch==2.11.0+cu128 torchvision torchaudio==2.11.0+cu128 --index-url "$TORCH_INDEX"
    uv pip install --python envs/cleanunet/bin/python soundfile soxr tqdm pyyaml
fi

# 5. Check CleanUNet repository and weights
if [ ! -f "third_party/CleanUNet/exp/DNS-large-full/checkpoint/pretrained.pkl" ]; then
    echo "Setting up CleanUNet repository and weights..."
    mkdir -p third_party
    if [ ! -d "third_party/CleanUNet" ]; then
        git clone https://github.com/NVIDIA/CleanUNet.git third_party/CleanUNet
    fi
    (cd third_party/CleanUNet && git lfs install && git lfs pull --include="exp/DNS-large-full/checkpoint/pretrained.pkl")
fi

# 6. GPU Self-Test in envs/eval
echo "Verifying GPU in envs/eval..."
envs/eval/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ GPU OK:', torch.cuda.get_device_name(0), '| archs:', torch.cuda.get_arch_list())"

echo "=== Setup complete! All 6 environments and dependencies ready. ==="


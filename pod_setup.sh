#!/bin/bash
# RunPod GPU Pod Setup Script for StoryGen - ROCm/AMD Version
# For AMD MI300X and other ROCm-compatible GPUs
# Run this once when you create the pod

set -e

echo "=== StoryGen Pod Setup (ROCm/AMD) ==="

# Check for ROCm
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not found. This script is for AMD GPUs with ROCm."
    echo "For NVIDIA GPUs, use the CUDA version instead."
    exit 1
fi

echo "Detected GPU:"
rocm-smi --showproductname

# Create model directory (use /workspace for persistence)
MODEL_DIR="/workspace/models"
mkdir -p $MODEL_DIR

# Install system dependencies
apt-get update && apt-get install -y ffmpeg libsndfile1 git-lfs aria2

# Install Python dependencies
pip install --upgrade pip

# Install PyTorch with ROCm 6.0 support
echo "Installing PyTorch with ROCm 6.0..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify ROCm PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Install other dependencies (no bitsandbytes - not ROCm compatible)
pip install \
    fastapi \
    uvicorn[standard] \
    transformers>=4.45.0 \
    accelerate \
    diffusers>=0.32.0 \
    sentencepiece \
    protobuf \
    scipy \
    soundfile \
    librosa \
    huggingface_hub \
    safetensors \
    omegaconf \
    einops \
    toml \
    peft \
    aiofiles \
    aiohttp \
    python-multipart \
    pillow

# Install flash-attention for ROCm (if available)
echo "Attempting to install flash-attention for ROCm..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available for ROCm, continuing without it..."

# Parler-TTS
pip install git+https://github.com/huggingface/parler-tts.git

# Fish Speech
pip install fish-speech

# Kohya sd-scripts for LoRA training
# Note: Kohya has limited ROCm support, may need modifications
if [ ! -d "/workspace/sd-scripts" ]; then
    echo "Cloning Kohya sd-scripts (ROCm compatibility may vary)..."
    git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts
    cd /workspace/sd-scripts
    # Install without bitsandbytes
    pip install -r requirements.txt --ignore-installed bitsandbytes || true
fi

echo "=== Downloading Models ==="

# Download models (this takes a while first time)
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/workspace/models"

from huggingface_hub import snapshot_download

print("Downloading Qwen-Image-2512...")
snapshot_download("Qwen/Qwen-Image-2512", local_dir="/workspace/models/qwen-image")

print("Downloading Wan 2.1 FLF2V (First-Last-Frame to Video)...")
snapshot_download("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers", local_dir="/workspace/models/wan-flf2v")

print("Downloading Parler-TTS...")
snapshot_download("parler-tts/parler-tts-mini-v1", local_dir="/workspace/models/parler-tts")

print("Downloading Fish Speech...")
snapshot_download("fishaudio/fish-speech-1.4", local_dir="/workspace/models/fish-speech")

print("=== All models downloaded! ===")
EOF

echo ""
echo "=== Setup Complete (ROCm/AMD) ==="
echo "GPU Info:"
rocm-smi --showmeminfo vram
echo ""
echo "Models are in: /workspace/models"
echo ""
echo "To start the server, run:"
echo "  python /workspace/pod_server.py"
echo ""

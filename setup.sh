#!/bin/bash
# ==========================================
# Setup Script for InternVideo2.5
#
# Use runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 as base image
# ==========================================

echo "🔧 Creating workspace..."

cd ..
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
# source ~/.bashrc
source $HOME/miniconda/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name=mobile_videogpt python=3.10
conda activate internvideo

pip install --upgrade pip

# apt-get update

# python3 -m venv venv
# source venv/bin/activate

# --------------------------------------------------
# 1️⃣ Install base dependencies
# --------------------------------------------------
echo "🧱 Installing base Python packages..."

cd internvideo2.5-adaptation/

pip install -r xtuner-train_internvideo2_5/requirements_main.txt

export PYTHONPATH="./:$PYTHONPATH"

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

which nvcc

conda activate internvideo

echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "❌ nvcc not found"
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ PyTorch cannot see CUDA')
"

echo ""
echo "=== Flash Attention Check ==="
python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('❌ Flash Attention not installed')
"

pip install openpyxl scikit-learn sentence-transformers rouge_score scikit-image

pip install git+https://github.com/okankop/vidaug

apt-get update
apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

echo "✅ Setup complete!"
echo "🚀 InternVideo2.5 environment is ready."

# Initialize WandB
echo "🔑 Logging into WandB..."
wandb login

# Initialize HuggingFace Hub
echo "🤗 Logging into HuggingFace Hub..."
hf auth login

source ~/.bashrc

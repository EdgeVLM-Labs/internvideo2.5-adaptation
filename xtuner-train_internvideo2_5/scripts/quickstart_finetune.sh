#!/bin/bash

# InternVideo2.5 QEVD-Fit-300k Finetuning - Quick Start
# This script runs all necessary steps to start finetuning

set -e  # Exit on error

# Setup logging
mkdir -p work_dirs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="work_dirs/finetune_${TIMESTAMP}.log"
echo "Logging all output to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "InternVideo2.5 QEVD-Fit-300k Finetuning"
echo "Quick Start Script"
echo "========================================="

# Step 1: Check environment
echo -e "\n[Step 1/5] Checking environment..."
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Step 2: Verify dataset
echo -e "\n[Step 2/5] Verifying dataset..."
if [ ! -f "data/diy_ft_data.json" ]; then
    echo "Error: data/diy_ft_data.json not found!"
    echo "Please run scripts/initialize_dataset.sh first."
    exit 1
fi

if [ ! -f "data/annotaions/qevd_fit_300k_train.jsonl" ]; then
    echo "Error: Training data not found!"
    echo "Please run scripts/initialize_dataset.sh to prepare the dataset."
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < data/annotaions/qevd_fit_300k_train.jsonl)
echo "✓ Found training data: $TRAIN_SAMPLES samples"

if [ ! -d "dataset" ]; then
    echo "Warning: dataset/ directory not found. Videos should be in dataset/ folder."
fi

# Step 3: Check model
echo -e "\n[Step 3/5] Checking model..."
MODEL_PATH="OpenGVLab/InternVideo2_5-Chat-8B"
echo "Model: $MODEL_PATH"
echo "Note: Model will be downloaded from HuggingFace if not cached locally."

# Step 4: Confirm to proceed
echo -e "\n========================================="
echo "Ready to start finetuning!"
echo "This will:"
echo "  - Finetune InternVideo2.5-8B on QEVD-Fit-300k"
echo "  - Use single GPU with DeepSpeed ZeRO-2"
echo "  - Use QLoRA (4-bit) for memory efficiency"
echo "  - Save checkpoints to work_dirs/qevd_fit_300k_internvideo2_5_r16/"
echo ""
echo -n "Continue? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 5: Start finetuning
echo -e "\n[Step 5/5] Starting finetuning..."
echo "========================================="

# Run finetuning
bash scripts/finetune_qved.sh

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "\n========================================="
    echo "✓ Finetuning Complete!"
    echo "========================================="
    echo "Model saved to: work_dirs/qevd_fit_300k_internvideo2_5_r16/"
    echo ""

    # Find latest checkpoint
    LATEST_CKPT=$(ls -d work_dirs/qevd_fit_300k_internvideo2_5_r16/step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Latest checkpoint: $LATEST_CKPT"
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Test the model: bash scripts/run_inference.sh"
    echo "  2. Upload to HuggingFace: python utils/hf_upload.py --model_path work_dirs/qevd_fit_300k_internvideo2_5_r16/"
else
    echo -e "\n========================================="
    echo "✗ Finetuning Failed!"
    echo "========================================="
    echo "Check the log file for errors: $LOG_FILE"
    exit 1
fi

echo -e "\n========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "To use the finetuned model:"
echo "  python utils/infer_qved.py \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --video_path sample_videos/00000340.mp4"
echo ""
echo "Adjustable parameters in utils/infer_qved.py:"
echo "  --model_path       Path to model checkpoint (default: Amshaker/Mobile-VideoGPT-0.5B)"
echo "  --video_path       Path to video file (default: sample_videos/00000340.mp4)"
echo "  --prompt           Custom prompt (default: physiotherapy evaluation prompt)"
echo "  --device           Device to use (default: cuda, options: cuda/cpu)"
echo "  --max_new_tokens   Max tokens to generate (default: 512)"
echo ""
echo "To run the inference script:"
echo ""
echo "Using local checkpoint:"
echo "  bash scripts/run_inference.sh --model_path $MODEL_PATH"
echo ""
echo "Using HuggingFace model:"
echo "  bash scripts/run_inference.sh --hf_repo EdgeVLM-Labs/${HF_REPO_NAME}"
echo ""
echo "========================================="

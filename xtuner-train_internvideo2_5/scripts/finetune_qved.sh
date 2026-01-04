#!/bin/bash

# InternVideo2.5 Finetuning Script for QEVD-Fit-300k Dataset
# Single GPU training with DeepSpeed and QLoRA (4-bit quantization)

# Environment setup
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # Single GPU

# Suppress warnings
export TOKENIZERS_PARALLELISM=false

# Model Configuration
# Choose one of these models:
# - OpenGVLab/InternVideo2_5-Chat-8B (Recommended for video understanding)
# - OpenGVLab/InternVL-2-5-HiCo-R16 (Alternative with high-quality features)
MODEL_NAME="OpenGVLab/InternVideo2_5-Chat-8B"
WORK_DIR="work_dirs/qevd_fit_300k_internvideo2_5_r16"

# Create output directory
mkdir -p "$WORK_DIR"

# Training Hyperparameters for Single GPU
MICRO_BATCH_SIZE=1          # Batch size per forward pass (reduce if OOM)
GLOBAL_BATCH_SIZE=8         # Total batch size (adjust based on GPU memory)
EPOCHS=3                     # Training epochs
LR=1e-5                      # Learning rate (conservative for finetuning)
VIT_LR=5e-6                  # Vision tower learning rate (lower)
CONNECTOR_LR=1e-5            # Connector learning rate
MAX_LENGTH=8192              # Maximum sequence length
MIN_FRAMES=8                 # Minimum frames per video
MAX_FRAMES=8                 # Maximum frames per video

echo "========================================="
echo "InternVideo2.5 QEVD-Fit-300k Finetuning"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Output: $WORK_DIR"
echo "Epochs: $EPOCHS"
echo "Micro Batch: $MICRO_BATCH_SIZE"
echo "Global Batch: $GLOBAL_BATCH_SIZE"
echo "Learning Rates: LR=$LR, VIT_LR=$VIT_LR, CONNECTOR_LR=$CONNECTOR_LR"
echo "Frames: MIN=$MIN_FRAMES, MAX=$MAX_FRAMES"
echo "Max Length: $MAX_LENGTH"
echo "========================================="

# Save configuration
cat <<EOF > "$WORK_DIR/training_config.json"
{
  "model": "$MODEL_NAME",
  "dataset": "QEVD-Fit-300k",
  "micro_batch_size": $MICRO_BATCH_SIZE,
  "global_batch_size": $GLOBAL_BATCH_SIZE,
  "epochs": $EPOCHS,
  "lr": $LR,
  "vit_lr": $VIT_LR,
  "connector_lr": $CONNECTOR_LR,
  "max_length": $MAX_LENGTH,
  "min_frames": $MIN_FRAMES,
  "max_frames": $MAX_FRAMES,
  "quantization": "QLoRA-4bit",
  "gpu_count": 1
}
EOF

# Start Training with DeepSpeed ZeRO-2
# Note: xtuner will handle the training through unify_internvl2_train_r16.py
echo ""
echo "Starting training..."
echo ""

NPROC_PER_NODE=1 xtuner train unify_internvl2_train_r16.py \
  --model "$MODEL_NAME" \
  --datasets data/diy_ft_data.json \
  --work-dir "$WORK_DIR" \
  --deepspeed scripts/zero2.json \
  --mirco-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --vit_lr $VIT_LR \
  --connector_lr $CONNECTOR_LR \
  --max-length $MAX_LENGTH \
  --min_num_frames $MIN_FRAMES \
  --max_num_frames $MAX_FRAMES \
  --shard-strategy zero2 \
  --freeze-vit \
  --checkpoint-interval 0.25 \
  --log-interval 10 \
  --seed 42

echo ""
echo "========================================="
echo "✓ Finetuning completed!"
echo "========================================="
echo "Model saved to: $WORK_DIR"
echo ""
echo "Next steps:"
echo "  1. Test inference: bash scripts/run_inference.sh"
echo "  2. Upload to HuggingFace Hub"
echo "========================================="

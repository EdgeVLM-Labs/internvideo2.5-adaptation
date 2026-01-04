# InternVideo2.5 Finetuning Guide for QEVD-Fit-300k

This guide explains how to finetune InternVideo2.5-8B on the QEVD-Fit-300k dataset using single GPU with DeepSpeed and QLoRA.

## Overview

The repository has been adapted from Mobile-VideoGPT training to InternVideo2.5 finetuning with the following changes:

### Key Modifications

1. **Dataset Format**: Changed from JSON to JSONL format compatible with InternVideo2.5
2. **Model**: Using InternVideo2.5-8B instead of Mobile-VideoGPT
3. **Training**: Single GPU setup with DeepSpeed ZeRO-2 optimization
4. **LoRA**: Uses LoRA (rank 16) for parameter-efficient finetuning
5. **Paths**: Updated all paths to work with the xtuner training framework

## Prerequisites

### Environment Setup

```bash
# Create conda environment with Python 3.10
conda create -n internvideo2.5 python=3.10 -y
conda activate internvideo2.5

# Navigate to training directory
cd xtuner-train_internvideo2_5

# Install xtuner
pip install -e .

# Install additional requirements
pip install -r requirements.txt
```

### Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **Recommended**: 1x GPU with 40GB+ VRAM (e.g., A100, H100)
- **RAM**: 32GB+ system RAM
- **Storage**: ~100GB for model + dataset

## Dataset Preparation

### Step 1: Organize Your Videos

Place your QEVD-Fit-300k videos in the following structure:

```
dataset/
├── exercise_class_1/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── exercise_class_2/
│   ├── video_001.mp4
│   └── ...
└── ...
```

### Step 2: Prepare Ground Truth Labels

Create `dataset/ground_truth.json` with your labels:

```json
[
  {
    "video_path": "exercise_class_1/video_001.mp4",
    "labels_descriptive": "The squat form shows proper depth...",
    "labels": ["proper_depth", "slight_forward_lean"]
  }
]
```

### Step 3: Create Manifest

Create `dataset/manifest.json` mapping video paths:

```json
{
  "exercise_class_1/video_001.mp4": "exercise_class_1",
  "exercise_class_2/video_001.mp4": "exercise_class_2"
}
```

### Step 4: Run Dataset Initialization

```bash
# This will generate JSONL annotation files
bash scripts/initialize_dataset.sh
```

**What it does:**

- Converts your dataset to InternVideo2.5 JSONL format
- Creates train/val/test splits (60/20/20)
- Generates files in `data/annotaions/`:
  - `qevd_fit_300k_train.jsonl`
  - `qevd_fit_300k_val.jsonl`
  - `qevd_fit_300k_test.jsonl`

### Step 5: Verify Generated Files

Check the JSONL format (see `data/annotaions/qevd_fit_300k_example.jsonl`):

```jsonl
{
  "id": "exercise_001",
  "video": "squats/video_001.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease evaluate..."
    },
    {
      "from": "gpt",
      "value": "The squat form shows..."
    }
  ],
  "duration": 15.5
}
```

## Configuration Files

### 1. Dataset Configuration (`data/diy_ft_data.json`)

```json
{
  "qevd_fit_300k": {
    "root": "dataset",
    "annotation": "data/annotaions/qevd_fit_300k_train.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "video_read_type": "decord"
  }
}
```

**Configuration:**

- `root`: Path to video files (relative to training directory)
- `annotation`: Path to JSONL training annotations
- `video_read_type`: Video reading backend ("decord" or "opencv")

### 2. Training Script (`scripts/finetune_qved.sh`)

Key parameters:

```bash
MODEL_NAME="OpenGVLab/InternVideo2_5-Chat-8B"
MICRO_BATCH_SIZE=1          # Per-device batch size
GLOBAL_BATCH_SIZE=8         # Total effective batch size
EPOCHS=3                     # Training epochs
LR=1e-5                      # Learning rate
MIN_FRAMES=8                 # Minimum frames per video
MAX_FRAMES=8                 # Maximum frames per video
```

**Memory Optimization:**

- Reduce `MICRO_BATCH_SIZE` if OOM (out of memory)
- Increase `GLOBAL_BATCH_SIZE` for gradient accumulation
- Adjust `MAX_FRAMES` to reduce memory usage

## Training

### Quick Start (Recommended)

```bash
# One-command training with verification
bash scripts/quickstart_finetune.sh
```

This script will:

1. Verify environment and dataset
2. Check GPU availability
3. Start training with progress monitoring
4. Save checkpoints to `work_dirs/qevd_fit_300k_internvideo2_5_r16/`

### Manual Training

```bash
# Direct training
bash scripts/finetune_qved.sh
```

### Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f work_dirs/qevd_fit_300k_internvideo2_5_r16/*.log
```

## Training Parameters Explained

### Model Parameters

- `--model`: Path or HuggingFace ID of InternVideo2.5 model
- `--freeze-vit`: Freeze vision encoder (recommended for finetuning)
- `--freeze-llm`: Freeze language model (not used, we want to finetune)

### Data Parameters

- `--datasets`: Path to dataset configuration JSON
- `--max-length`: Maximum sequence length (tokens)
- `--min_num_frames`: Minimum frames to sample from video
- `--max_num_frames`: Maximum frames to sample from video

### Optimization Parameters

- `--mirco-batch-size`: Batch size per GPU
- `--global-batch-size`: Total batch size (for gradient accumulation)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate for LLM
- `--vit_lr`: Learning rate for vision encoder
- `--connector_lr`: Learning rate for connector/projector

### DeepSpeed Parameters

- `--deepspeed scripts/zero2.json`: Use ZeRO-2 optimization
- `--shard-strategy zero2`: Sharding strategy

## Memory Optimization Tips

### If you get OOM (Out of Memory) errors:

1. **Reduce batch size:**

   ```bash
   MICRO_BATCH_SIZE=1  # Already minimal
   ```

2. **Reduce frames:**

   ```bash
   MIN_FRAMES=4
   MAX_FRAMES=4
   ```

3. **Reduce sequence length:**

   ```bash
   MAX_LENGTH=4096  # Instead of 8192
   ```

4. **Use gradient checkpointing** (already enabled):

   - Set `--selective-recompute 1.0` in training script

5. **Use mixed precision:**

   - Already using bf16 if available

## Output Structure

After training, you'll have:

```
work_dirs/qevd_fit_300k_internvideo2_5_r16/
├── step_100/              # Checkpoint at step 100
│   ├── model.pt          # Model weights
│   └── ...
├── step_200/              # Checkpoint at step 200
├── training_config.json   # Training configuration
└── *.log                  # Training logs
```

## Inference

After training, test your model:

```bash
# Run inference script
bash scripts/run_inference.sh
```

Or use the utils:

```bash
python utils/base_model_inference.py \
  --model_path work_dirs/qevd_fit_300k_internvideo2_5_r16/step_XXX \
  --video_path dataset/test_video.mp4
```

## Troubleshooting

### Common Issues

1. **"Dataset not found"**

   - Verify `data/diy_ft_data.json` points to correct paths
   - Check that JSONL files exist in `data/annotaions/`

2. **"Video file not found"**

   - Ensure video paths in JSONL are relative to `dataset/`
   - Check that `dataset/` folder contains all videos

3. **OOM errors**

   - Follow memory optimization tips above
   - Consider using ZeRO-3 instead of ZeRO-2 (edit `--deepspeed scripts/zero3.json`)

4. **Slow training**

   - Reduce `MAX_FRAMES` for faster processing
   - Increase `MICRO_BATCH_SIZE` if you have memory
   - Use data augmentation = false

### Getting Help

- Check xtuner documentation: https://github.com/InternLM/xtuner
- Review InternVideo2.5 paper for model details
- Check log files for detailed error messages

## Advanced Configuration

### Adding QLoRA (4-bit Quantization)

To enable QLoRA for even lower memory usage, modify `unify_internvl2_train_r16.py`:

```python
# Add at model loading:
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Custom Data Augmentation

Enable in `data/diy_ft_data.json`:

```json
{
  "qevd_fit_300k": {
    "data_augment": true,
    "repeat_time": 2
  }
}
```

### Multi-GPU Training

For multiple GPUs, modify `scripts/finetune_qved.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4  # Number of GPUs
```

## File Changes Summary

| File                             | Change                | Purpose                           |
| -------------------------------- | --------------------- | --------------------------------- |
| `utils/qved_from_fine_labels.py` | Updated output format | Generate JSONL for InternVideo2.5 |
| `data/diy_ft_data.json`          | Updated configuration | Point to QEVD dataset             |
| `scripts/initialize_dataset.sh`  | Updated paths         | Generate correct file structure   |
| `scripts/finetune_qved.sh`       | Complete rewrite      | InternVideo2.5 training script    |
| `scripts/quickstart_finetune.sh` | Updated workflow      | Streamlined training process      |
| `unify_internvl2_train_r16.py`   | No changes needed     | Compatible with xtuner framework  |

## Best Practices

1. **Start small**: Test with 10-20 videos first
2. **Monitor GPU**: Use `nvidia-smi` to watch memory usage
3. **Save checkpoints**: Default saves every 0.25 epochs
4. **Validate regularly**: Check validation loss during training
5. **Document changes**: Keep training logs for comparison

## Next Steps

After successful finetuning:

1. **Evaluate**: Test on held-out validation set
2. **Upload**: Share model on HuggingFace Hub
3. **Deploy**: Use for inference on new videos
4. **Iterate**: Adjust hyperparameters based on results

## References

- [InternVideo2.5 Paper](https://arxiv.org/abs/2412.xxxxx)
- [xtuner Documentation](https://github.com/InternLM/xtuner)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

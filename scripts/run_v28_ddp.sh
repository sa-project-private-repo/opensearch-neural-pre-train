#!/bin/bash
# V28 Multi-GPU Training Script (8x A100)
#
# Key fixes from previous failed training:
# - non_korean_penalty: 5.0 (was 100.0)
# - lambda_language: 0.3 (was 0.5)
#
# Usage:
#   ./scripts/run_v28_ddp.sh
#   ./scripts/run_v28_ddp.sh --resume

set -e

# Configuration
NUM_GPUS=8
BATCH_SIZE=32          # Per-GPU batch size
GRAD_ACCUM=2           # Gradient accumulation steps
# Effective batch size = 32 * 8 * 2 = 512

EPOCHS=30
LR=3e-5
OUTPUT_DIR="outputs/train_v28_ddp"

# V28 FIXED parameters (prevent collapse)
NON_KOREAN_PENALTY=5.0    # Reduced from 100.0
LAMBDA_LANGUAGE=0.3       # Reduced from 0.5

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "V28 Multi-GPU Training (8x A100)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""
echo "V28 FIXED Parameters:"
echo "  non_korean_penalty: $NON_KOREAN_PENALTY (was 100.0)"
echo "  lambda_language: $LAMBDA_LANGUAGE (was 0.5)"
echo ""
echo "Output: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Run training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m src.train.cli.train_v28_ddp \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --output-dir "$OUTPUT_DIR" \
    --non-korean-penalty $NON_KOREAN_PENALTY \
    --lambda-language $LAMBDA_LANGUAGE \
    "$@"

echo ""
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"

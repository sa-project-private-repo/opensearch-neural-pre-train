#!/bin/bash
# V34 Training Launch Script for B200 x8
#
# V34: V33 + expanded Korean data + extended training + relaxed query sparsity
# Pure SPLADE v2 recipe: InfoNCE + FLOPS (quadratic warmup)
# Architecture identical to V33; only config differs.
#
# Effective batch size: 64 * 4 * 8 = 2048
# Estimated time: ~24h on B200 x8

set -euo pipefail

# Configuration
NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-"configs/train_v34.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/train_v34"}
MASTER_PORT=${MASTER_PORT:-29500}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo "V34 SPLADE-max Training (ModernBERT)"
echo "============================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "============================================="

# Activate virtual environment
source .venv/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Launch with torchrun (reuses V33 CLI; config drives all V34 differences)
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m src.train.cli.train_v33_ddp \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "Training complete. Output at: $OUTPUT_DIR"

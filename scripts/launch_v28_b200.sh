#!/bin/bash
# Launch V28 training on B200 x8 with DDP
#
# Usage:
#   bash scripts/launch_v28_b200.sh
#   bash scripts/launch_v28_b200.sh --resume
#   bash scripts/launch_v28_b200.sh --config configs/my_config.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: python -m venv .venv"
    exit 1
fi

# DDP settings
export NPROC_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# NCCL settings for B200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training settings
CONFIG="configs/train_v28_b200.yaml"
OUTPUT_DIR="outputs/train_v28_ddp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR"

echo "=== V28 DDP Training on B200 x8 ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo "Log: $LOG_FILE"
echo "==================================="

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.train.cli.train_v28_ddp \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo "Training complete. Log: $LOG_FILE"

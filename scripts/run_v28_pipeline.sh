#!/bin/bash
# V28 Full Pipeline: Data Collection -> Preprocessing -> Training
#
# Usage:
#   nohup bash scripts/run_v28_pipeline.sh > pipeline.log 2>&1 &
#   tail -f pipeline.log
#
# Skip data collection (already done):
#   nohup bash scripts/run_v28_pipeline.sh --skip-collect > pipeline.log 2>&1 &
#
# Skip all preprocessing (data ready):
#   nohup bash scripts/run_v28_pipeline.sh --skip-preprocess > pipeline.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse flags
SKIP_COLLECT=false
SKIP_PREPROCESS=false
for arg in "$@"; do
    case $arg in
        --skip-collect) SKIP_COLLECT=true ;;
        --skip-preprocess) SKIP_PREPROCESS=true ;;
    esac
done

# Activate venv
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "============================================"
echo "  V28 Full Training Pipeline"
echo "  Started: $(date)"
echo "  PID: $$"
echo "============================================"

# ===== Phase 1: Data Collection =====
if [[ "$SKIP_PREPROCESS" == "false" && "$SKIP_COLLECT" == "false" ]]; then
    echo ""
    echo "[Phase 1/3] Collecting Korean datasets from HuggingFace..."
    echo "  Start: $(date)"

    python scripts/collect_korean_datasets.py \
        --output-dir data/v29.0/raw \
        --max-samples 500000

    echo "  Done: $(date)"
else
    echo "[Phase 1/3] SKIPPED: Data collection"
fi

# ===== Phase 2: Data Preprocessing =====
if [[ "$SKIP_PREPROCESS" == "false" ]]; then
    echo ""
    echo "[Phase 2/3] Building V29 training data (merge, dedup, shard)..."
    echo "  Start: $(date)"

    python scripts/build_v29_data.py \
        --output-dir data/v29.0 \
        --shard-size 100000 \
        --val-ratio 0.05 \
        --seed 42 \
        --dedup-threshold 0.8

    echo "  Done: $(date)"

    # Print data stats
    echo ""
    echo "  Data summary:"
    wc -l data/v29.0/train_*.jsonl | tail -1
    wc -l data/v29.0/val.jsonl
    echo ""
else
    echo "[Phase 2/3] SKIPPED: Data preprocessing"
fi

# ===== Phase 3: Training =====
echo ""
echo "[Phase 3/3] Starting V28 DDP training..."
echo "  Start: $(date)"

# Check GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "  Detected GPUs: $GPU_COUNT"

if [[ "$GPU_COUNT" -ge 2 ]]; then
    # Multi-GPU DDP training
    echo "  Mode: DDP (torchrun --nproc_per_node=$GPU_COUNT)"

    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=5
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    OUTPUT_DIR="outputs/train_v28_ddp"
    mkdir -p "$OUTPUT_DIR"

    torchrun \
        --nproc_per_node="$GPU_COUNT" \
        --master_addr=localhost \
        --master_port=29500 \
        -m src.train.cli.train_v28_ddp \
        --config configs/train_v28_b200.yaml \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}/training_${TIMESTAMP}.log"
else
    # Single GPU training
    echo "  Mode: Single GPU"

    OUTPUT_DIR="outputs/train_v28"
    mkdir -p "$OUTPUT_DIR"

    python -m src.train v28 \
        --config configs/train_v28.yaml \
        --output-dir "$OUTPUT_DIR" \
        --train-files "data/v29.0/train_*.jsonl" \
        --val-files "data/v29.0/val.jsonl" \
        2>&1 | tee "${OUTPUT_DIR}/training_${TIMESTAMP}.log"
fi

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "  Finished: $(date)"
echo "============================================"

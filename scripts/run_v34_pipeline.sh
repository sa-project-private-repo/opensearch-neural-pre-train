#!/bin/bash
set -e

cd /home/ec2-user/workspace/opensearch-workspace/opensearch-neural-pre-train
source .venv/bin/activate

echo "============================================"
echo "V34 Training Pipeline (P0 KD + P1 Multi-Neg)"
echo "============================================"
echo "Start: $(date)"
echo ""

# Step 1: Pre-compute teacher scores
echo "=== Step 1/3: Pre-compute BGE-M3 teacher scores ==="
if [ -f data/v29.0_kd/teacher_embeddings.npy ]; then
    echo "Cached embeddings found, skipping encoding."
else
    python scripts/precompute_teacher_scores.py \
        --input-pattern "data/v29.0/train_*.jsonl" \
        --val-pattern "data/v29.0/val.jsonl" \
        --output-dir data/v29.0_kd \
        --batch-size 256 \
        --max-length 256 \
        --device cuda \
        --save-embeddings \
        --num-gpus 8
fi
echo "Step 1 complete: $(date)"
echo ""

# Step 2: Mine multi-hard-negatives
echo "=== Step 2/3: Mine k=7 hard negatives with FAISS ==="
if [ -d data/v29.0_multi_neg ] && [ "$(ls data/v29.0_multi_neg/train_*.jsonl 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Multi-neg data found, skipping mining."
else
    python scripts/mine_multi_negatives.py \
        --input-pattern "data/v29.0_kd/train_*.jsonl" \
        --val-pattern "data/v29.0_kd/val.jsonl" \
        --embeddings data/v29.0_kd/teacher_embeddings.npy \
        --text-index data/v29.0_kd/text_to_idx.json \
        --output-dir data/v29.0_multi_neg \
        --k 7 \
        --rank-start 10 \
        --rank-end 50 \
        --search-k 100
fi
echo "Step 2 complete: $(date)"
echo ""

# Step 3: Train V34 with multi-neg + KD
echo "=== Step 3/3: Train V34 (multi-neg + MarginMSE KD) ==="
echo "Config: configs/train_v34_multi_neg.yaml"
echo "Checkpoint: outputs/train_v33/final_model/model.pt"
echo "GPUs: 8x B200"
torchrun --nproc_per_node=8 \
    -m src.train.cli.train_v33_ddp \
    --config configs/train_v34_multi_neg.yaml \
    --checkpoint outputs/train_v33/final_model

echo ""
echo "============================================"
echo "Pipeline complete: $(date)"
echo "============================================"

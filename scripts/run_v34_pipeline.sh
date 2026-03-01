#!/bin/bash
# V34 Pipeline: build dataset → launch training
# Run AFTER collect_v34_datasets.py completes.
set -e

cd /home/ec2-user/workspace/opensearch-workspace/opensearch-neural-pre-train
source .venv/bin/activate

LOG="outputs/train_v34/pipeline.log"
mkdir -p outputs/train_v34

echo "$(date) | V34 Pipeline started" | tee -a "$LOG"

# Step 1: Build v34.0 dataset (merge v29.0 + new, dedup, shard)
echo "$(date) | Building v34.0 dataset..." | tee -a "$LOG"
python scripts/build_v34_data.py \
    --output-dir data/v34.0 \
    --shard-size 100000 \
    --val-ratio 0.05 \
    --seed 42 \
    2>&1 | tee -a "$LOG"

echo "$(date) | Dataset build complete" | tee -a "$LOG"
wc -l data/v34.0/train_shard_*.jsonl | tail -1 | tee -a "$LOG"
wc -l data/v34.0/val.jsonl | tee -a "$LOG"

# Step 2: Launch V34 training on B200 x8
echo "$(date) | Starting V34 training..." | tee -a "$LOG"
nohup bash scripts/launch_v34_b200.sh >> "$LOG" 2>&1 &
TRAIN_PID=$!
echo "$(date) | Training launched (PID: $TRAIN_PID)" | tee -a "$LOG"

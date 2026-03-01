#!/bin/bash
# Wait for V34 training to complete, then run benchmarks on all 3 datasets.
set -e

cd /home/ec2-user/workspace/opensearch-workspace/opensearch-neural-pre-train
source .venv/bin/activate

LOG="outputs/train_v34/benchmark_runner.log"
CHECKPOINT="outputs/train_v34/final_model/model.pt"

echo "$(date) | Waiting for V34 training to complete..." | tee -a "$LOG"

# Wait for final checkpoint to appear
while [ ! -f "$CHECKPOINT" ]; do
    sleep 60
done

# Wait a bit more for file write to finish
sleep 30
echo "$(date) | Training complete. Final checkpoint found: $CHECKPOINT" | tee -a "$LOG"

# Run benchmarks sequentially
for DATASET in ko-strategyqa miracl-ko mrtydi-ko; do
    echo "$(date) | Starting benchmark: $DATASET" | tee -a "$LOG"
    python -m benchmark.hf_runner \
        --dataset "$DATASET" \
        --checkpoint "$CHECKPOINT" \
        --index-suffix v34 \
        --output-dir "outputs/benchmarks/v34/$DATASET" \
        --cleanup \
        2>&1 | tee -a "$LOG"
    echo "$(date) | Completed benchmark: $DATASET" | tee -a "$LOG"
done

echo "$(date) | All V34 benchmarks complete!" | tee -a "$LOG"

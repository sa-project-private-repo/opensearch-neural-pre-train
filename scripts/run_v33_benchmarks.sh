#!/bin/bash
# Wait for V33 training to complete, then run benchmarks on all 3 datasets.
set -e

cd /home/ec2-user/workspace/opensearch-workspace/opensearch-neural-pre-train
source .venv/bin/activate

LOG="outputs/train_v33/benchmark_runner.log"
CHECKPOINT="outputs/train_v33/final_model/model.pt"

echo "$(date) | Waiting for V33 training to complete..." | tee -a "$LOG"

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
        --index-suffix v33 \
        --output-dir "outputs/benchmarks/v33/$DATASET" \
        --cleanup \
        2>&1 | tee -a "$LOG"
    echo "$(date) | Completed benchmark: $DATASET" | tee -a "$LOG"
done

echo "$(date) | All V33 benchmarks complete!" | tee -a "$LOG"

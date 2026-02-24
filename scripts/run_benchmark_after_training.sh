#!/bin/bash
# Monitor V28 v3 training (SPLADELossV29) and run all 3 benchmarks upon completion.
# Usage: nohup bash scripts/run_benchmark_after_training.sh > outputs/benchmark_monitor_v3.log 2>&1 &
set -euo pipefail

TRAIN_DIR="outputs/train_v28_ddp_v3"
BENCH_DIR="outputs/benchmark_v28_v3"
PY=".venv/bin/python"

echo "[$(date)] Waiting for training to complete..."

# Poll every 5 min
while pgrep -f "train_v28_ddp.*train_v28_ddp_v3" > /dev/null 2>&1; do
    STEP=$(grep -oP 'step=\K\d+' "$TRAIN_DIR/training.log" 2>/dev/null | tail -1 || echo "?")
    echo "[$(date)] Training in progress... step=$STEP/56071"
    sleep 300
done

echo "[$(date)] Training process finished."

# Find latest checkpoint
CKPT=$(ls -td "$TRAIN_DIR"/checkpoint_epoch*/model.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[$(date)] ERROR: No checkpoint found"; exit 1
fi
echo "[$(date)] Checkpoint: $CKPT"

# Run 3 benchmarks sequentially
for DS in ko-strategyqa miracl-ko mrtydi-ko; do
    SUFFIX="v28v3-${DS}"
    OUT="$BENCH_DIR/$DS"
    mkdir -p "$OUT"

    echo ""
    echo "[$(date)] === Benchmark: $DS ==="
    $PY -m benchmark.hf_runner \
        --dataset "$DS" \
        --checkpoint "$CKPT" \
        --index-suffix "$SUFFIX" \
        --output-dir "$OUT" \
        --cleanup \
        2>&1 | tee "$OUT/run.log"

    echo "[$(date)] $DS done. Report: $OUT/report.md"
done

echo ""
echo "[$(date)] All benchmarks complete! Results in $BENCH_DIR/"

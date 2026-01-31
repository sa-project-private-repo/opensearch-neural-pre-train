#!/bin/bash
# V28 Auto-Start Script
# Waits for V27 training to complete, then starts V28 training
#
# Usage:
#   ./scripts/run_v28_after_v27.sh
#   nohup ./scripts/run_v28_after_v27.sh > outputs/v28_auto.log 2>&1 &

set -e

# Configuration
V27_OUTPUT_DIR="outputs/train_v27"
V27_FLAG_FILE="${V27_OUTPUT_DIR}/training_complete.flag"
V28_CONFIG="configs/train_v28.yaml"
V28_OUTPUT_DIR="outputs/train_v28"
CHECK_INTERVAL=300  # Check every 5 minutes

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}V28 Auto-Start Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Waiting for V27 training to complete..."
echo "  V27 flag file: ${V27_FLAG_FILE}"
echo "  Check interval: ${CHECK_INTERVAL} seconds"
echo ""

# Wait for V27 completion
wait_count=0
while [ ! -f "${V27_FLAG_FILE}" ]; do
    wait_count=$((wait_count + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check #${wait_count}: V27 still running..."

    # Check if V27 process is still alive
    if ! pgrep -f 'src.train v27' > /dev/null; then
        # No V27 process, check if flag exists (maybe it just finished)
        if [ ! -f "${V27_FLAG_FILE}" ]; then
            echo -e "${YELLOW}Warning: V27 process not found and no completion flag.${NC}"
            echo "V27 may have crashed. Checking for checkpoints..."

            if [ -d "${V27_OUTPUT_DIR}" ] && ls "${V27_OUTPUT_DIR}"/checkpoint_* 1> /dev/null 2>&1; then
                echo "Found V27 checkpoints. Proceeding with V28 training..."
                break
            else
                echo -e "${YELLOW}No V27 checkpoints found. Waiting for V27 to start...${NC}"
            fi
        fi
    fi

    sleep ${CHECK_INTERVAL}
done

echo ""
echo -e "${GREEN}V27 training complete!${NC}"
if [ -f "${V27_FLAG_FILE}" ]; then
    echo "Flag file contents:"
    cat "${V27_FLAG_FILE}"
fi
echo ""

# Brief pause before starting V28
echo "Starting V28 training in 10 seconds..."
sleep 10

# Start V28 training
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting V28 Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Config: ${V28_CONFIG}"
echo "Output: ${V28_OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${V28_OUTPUT_DIR}"

# Run V28 training
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
fi

# Run training
python -m src.train v28 --config "${V28_CONFIG}" 2>&1 | tee "${V28_OUTPUT_DIR}/training.log"

# Check exit status
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}V28 training completed successfully!${NC}"
else
    echo ""
    echo -e "${YELLOW}V28 training exited with code: ${exit_code}${NC}"
fi

exit $exit_code

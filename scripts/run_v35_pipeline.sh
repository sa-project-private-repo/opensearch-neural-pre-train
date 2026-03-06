#!/bin/bash
# V35 Two-Phase Training Pipeline
#
# Phase 1: Representation learning (5 epochs, no FLOPS)
# Phase 2: Sparsity compression (10 epochs, FLOPS only)
#
# Total: ~15 epochs, ~42 hours on 8x B200

set -e

VENV_DIR=".venv"
V33_CHECKPOINT="outputs/train_v33/final_model"
PHASE1_CONFIG="configs/train_v35_phase1.yaml"
PHASE2_CONFIG="configs/train_v35_phase2.yaml"
PHASE1_OUTPUT="outputs/train_v35_phase1"
PHASE2_OUTPUT="outputs/train_v35_phase2"

source "${VENV_DIR}/bin/activate"

# ============================================================
# Phase 1: Representation learning (no FLOPS)
# ============================================================
if [ -d "${PHASE1_OUTPUT}/final_model" ]; then
    echo "Phase 1 already completed. Skipping."
else
    echo "=========================================="
    echo "V35 Phase 1: Representation Learning"
    echo "  - Multi-neg (k=7) + MarginMSE"
    echo "  - NO FLOPS regularization"
    echo "  - 5 epochs from V33 checkpoint"
    echo "=========================================="

    mkdir -p "${PHASE1_OUTPUT}"
    torchrun --nproc_per_node=8 \
        -m src.train.cli.train_v33_ddp \
        --config "${PHASE1_CONFIG}" \
        --checkpoint "${V33_CHECKPOINT}"
fi

# ============================================================
# Phase 2: Sparsity compression (FLOPS only)
# ============================================================
if [ -d "${PHASE2_OUTPUT}/final_model" ]; then
    echo "Phase 2 already completed. Skipping."
else
    echo "=========================================="
    echo "V35 Phase 2: Sparsity Compression"
    echo "  - InfoNCE + FLOPS (V33 levels)"
    echo "  - NO MarginMSE"
    echo "  - 10 epochs from Phase 1 checkpoint"
    echo "=========================================="

    mkdir -p "${PHASE2_OUTPUT}"
    torchrun --nproc_per_node=8 \
        -m src.train.cli.train_v33_ddp \
        --config "${PHASE2_CONFIG}" \
        --checkpoint "${PHASE1_OUTPUT}/final_model"
fi

echo "=========================================="
echo "V35 Training Complete!"
echo "Final model: ${PHASE2_OUTPUT}/final_model"
echo "=========================================="

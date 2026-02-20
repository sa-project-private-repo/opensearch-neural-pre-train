# Makefile for SPLADE-doc Training on DGX Spark (ARM + GB10)
# Optimized for Nvidia DGX Spark with GB10 GPU and ARM64 architecture

.PHONY: help setup test prepare-baseline train-baseline train-pretrain train-finetune clean clean-outputs monitor logs \
	train-v22 train-v22-bg train-v22-resume logs-v22 tensorboard-v22 \
	train-v24 train-v24-bg train-v24-resume eval-v24 logs-v24 tensorboard-v24 \
	prepare-v25-idf prepare-v25-data train-v25 train-v25-bg train-v25-resume train-v25-quick train-v25-verify \
	eval-v25 eval-v25-sparsity convert-v25-hf logs-v25 tensorboard-v25 v25-pipeline \
	prepare-v26-idf prepare-v26-data train-v26 train-v26-bg train-v26-resume \
	eval-v26 eval-v26-sparsity convert-v26-hf logs-v26 tensorboard-v26 v26-pipeline \
	validate-semantic-ratio \
	collect-travel prepare-v27-data train-v27 train-v27-bg train-v27-resume \
	eval-v27 eval-v27-travel convert-v27-hf logs-v27 tensorboard-v27 v27-pipeline \
	build-korean-tokens train-v28 train-v28-bg train-v28-resume train-v28-after-v27 train-v28a \
	eval-v28 eval-v28-language eval-v28-context convert-v28-hf logs-v28 tensorboard-v28 v28-pipeline \
	collect-v29-data build-v29-data v29-data-stats \
	train-v28-ddp train-v28-ddp-bg train-v28-ddp-resume logs-v28-ddp tensorboard-v28-ddp \
	v29-pipeline v29-pipeline-bg

# Default target
.DEFAULT_GOAL := help

# Virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Configuration files
CONFIG_BASELINE := configs/baseline_dgx.yaml
CONFIG_PRETRAIN := configs/pretrain_korean_dgx.yaml
CONFIG_FINETUNE := configs/finetune_msmarco.yaml
CONFIG_V22 := configs/train_v22.yaml
CONFIG_V24 := configs/train_v24.yaml
CONFIG_V25 := configs/train_v25.yaml
CONFIG_V26 := configs/train_v26.yaml
CONFIG_V27 := configs/train_v27.yaml
CONFIG_V28_B200 := configs/train_v28_b200.yaml

# Output directories
OUTPUT_BASELINE := outputs/baseline_dgx
OUTPUT_PRETRAIN := outputs/pretrain_korean_dgx
OUTPUT_FINETUNE := outputs/finetune_msmarco
OUTPUT_V22 := outputs/train_v22
OUTPUT_V24 := outputs/train_v24
OUTPUT_V25 := outputs/train_v25
OUTPUT_V26 := outputs/train_v26
OUTPUT_V27 := outputs/train_v27

# V25 specific directories
IDF_WEIGHTS_DIR := outputs/idf_weights
V25_DATA_DIR := data/v25.0
V25_CHECKPOINT_DIR := checkpoints/v25.0
V25_HF_DIR := huggingface/v25

# V26 specific directories
V26_CHECKPOINT_DIR := checkpoints/v26.0
V26_HF_DIR := huggingface/v26

# V27 specific directories
V27_DATA_DIR := data/v27.0
V27_CHECKPOINT_DIR := checkpoints/v27.0
V27_HF_DIR := huggingface/v27

# V29 data directories
V29_DATA_DIR := data/v29.0
V29_RAW_DIR := data/v29.0/raw

# V28 DDP output
OUTPUT_V28_DDP := outputs/train_v28_ddp

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)SPLADE-doc Training - DGX Spark$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)GPU:$(NC) NVIDIA GB10 (119GB VRAM)"
	@echo "$(YELLOW)Arch:$(NC) ARM64"
	@echo "$(YELLOW)Optimization:$(NC) BF16 mixed precision"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(BLUE)<target>$(NC)\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

setup: ## Setup and test the environment
	@echo "$(BLUE)[1/2]$(NC) Activating virtual environment..."
	@test -d $(VENV) || (echo "$(RED)Error: venv not found. Run: python3 -m venv .venv$(NC)" && exit 1)
	@echo "$(GREEN)✓$(NC) Virtual environment found"
	@echo ""
	@echo "$(BLUE)[2/2]$(NC) Running GPU environment test..."
	@$(PYTHON) test_dgx_setup.py

test: ## Test GPU and model setup
	@echo "$(BLUE)Testing DGX Spark environment...$(NC)"
	@$(PYTHON) test_dgx_setup.py

##@ Data Preparation

prepare-baseline: ## Prepare baseline training data (10K samples)
	@echo "$(BLUE)Preparing baseline data (10K samples)...$(NC)"
	@echo "  - 5,000 from Korean Wikipedia"
	@echo "  - 5,000 from NamuWiki"
	@$(PYTHON) scripts/prepare_baseline_data.py
	@echo "$(GREEN)✓ Baseline data ready$(NC)"

##@ Training

train-baseline: ## Train baseline model (10K samples, ~10 minutes)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting Baseline Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_BASELINE)"
	@echo "Output: $(OUTPUT_BASELINE)"
	@echo "Dataset: 10K samples (Korean Wikipedia + NamuWiki)"
	@echo "Epochs: 3"
	@echo "Batch size: 16 (effective: 32 with gradient accumulation)"
	@echo "Mixed precision: BF16"
	@echo "Expected time: ~10 minutes on GB10"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) train.py --config $(CONFIG_BASELINE)

train-pretrain: ## Train full pre-training model (all Korean data)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting Pre-training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_PRETRAIN)"
	@echo "Output: $(OUTPUT_PRETRAIN)"
	@echo "Dataset: Full Korean + English Wikipedia"
	@echo "  - Korean Wikipedia: ~600K articles"
	@echo "  - NamuWiki: ~1.5M articles"
	@echo "  - English Wikipedia: ~6M articles"
	@echo "Epochs: 10"
	@echo "Batch size: 32 (effective: 64 with gradient accumulation)"
	@echo "Mixed precision: BF16"
	@echo "Expected time: Several hours to 1 day"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) train.py --config $(CONFIG_PRETRAIN)

train-finetune: ## Fine-tune on MS MARCO
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting Fine-tuning (MS MARCO)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_FINETUNE)"
	@echo "Output: $(OUTPUT_FINETUNE)"
	@echo "Dataset: MS MARCO triples"
	@echo "Epochs: 3"
	@echo "Batch size: 8 (effective: 64 with gradient accumulation)"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) train.py --config $(CONFIG_FINETUNE)

##@ V22 Curriculum Training

train-v22: ## V22 curriculum training (30 epochs, 3 phases)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V22 Curriculum Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V22)"
	@echo "Output: $(OUTPUT_V22)"
	@echo ""
	@echo "$(YELLOW)Curriculum Phases:$(NC)"
	@echo "  Phase 1 (1-10):  Single-term focus, temp=0.07"
	@echo "  Phase 2 (11-20): Balanced training, temp=0.05"
	@echo "  Phase 3 (21-30): Hard negatives, temp=0.03"
	@echo ""
	@echo "$(YELLOW)Loss Components:$(NC)"
	@echo "  InfoNCE + Self + Positive + Margin + FLOPS + MinAct"
	@echo ""
	@echo "Mixed precision: BF16"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) -m src.train v22 --config $(CONFIG_V22)

train-v22-bg: ## V22 training in background (nohup)
	@echo "$(BLUE)Starting V22 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V22)
	@nohup $(PYTHON) -m src.train v22 --config $(CONFIG_V22) > $(OUTPUT_V22)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@echo "PID: $$(pgrep -f 'src.train v22' | tail -1)"
	@echo "Log: $(OUTPUT_V22)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v22"
	@echo "  make tensorboard-v22"

train-v22-resume: ## Resume V22 training from checkpoint
	@echo "$(BLUE)Resuming V22 training from checkpoint...$(NC)"
	@$(PYTHON) -m src.train v22 --config $(CONFIG_V22) --resume

logs-v22: ## Show V22 training logs (real-time)
	@echo "$(BLUE)V22 Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V22)/training.log ]; then \
		tail -f $(OUTPUT_V22)/training.log; \
	elif [ -f $(OUTPUT_V22)/nohup.out ]; then \
		tail -f $(OUTPUT_V22)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v22$(NC)"; \
	fi

tensorboard-v22: ## Start TensorBoard for V22 training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V22)/tensorboard --port 6006

##@ V24 XLM-RoBERTa Training (Baseline)

train-v24: ## V24 XLM-RoBERTa training (25 epochs, BGE-M3 teacher)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V24 XLM-RoBERTa Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V24)"
	@echo "Output: $(OUTPUT_V24)"
	@echo ""
	@echo "$(YELLOW)Key Features:$(NC)"
	@echo "  - Base model: xlm-roberta-base (250K vocab)"
	@echo "  - Teacher: BAAI/bge-m3"
	@echo "  - Standard FLOPS (no IDF weighting)"
	@echo ""
	@$(PYTHON) -m src.train v24 --config $(CONFIG_V24)

train-v24-bg: ## V24 training in background
	@echo "$(BLUE)Starting V24 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V24)
	@nohup $(PYTHON) -m src.train v24 --config $(CONFIG_V24) > $(OUTPUT_V24)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@echo "PID: $$(pgrep -f 'src.train v24' | tail -1)"
	@echo "Log: $(OUTPUT_V24)/nohup.out"

train-v24-resume: ## Resume V24 training from checkpoint
	@echo "$(BLUE)Resuming V24 training...$(NC)"
	@$(PYTHON) -m src.train v24 --config $(CONFIG_V24) --resume

eval-v24: ## Evaluate V24 model on validation set
	@echo "$(BLUE)Evaluating V24 model...$(NC)"
	@$(PYTHON) scripts/quick_eval.py --model huggingface/v24_best --num-samples 100

logs-v24: ## Show V24 training logs
	@if [ -f $(OUTPUT_V24)/training.log ]; then \
		tail -f $(OUTPUT_V24)/training.log; \
	elif [ -f $(OUTPUT_V24)/nohup.out ]; then \
		tail -f $(OUTPUT_V24)/nohup.out; \
	else \
		echo "$(RED)No logs found$(NC)"; \
	fi

tensorboard-v24: ## Start TensorBoard for V24
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V24)/tensorboard --port 6006

##@ V25 IDF-Aware Training Pipeline

prepare-v25-idf: ## Compute IDF weights from training corpus
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Computing IDF Weights for V25$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p $(IDF_WEIGHTS_DIR)
	@echo "$(YELLOW)Corpus:$(NC) data/v24.0/train_*.jsonl"
	@echo "$(YELLOW)Output:$(NC) $(IDF_WEIGHTS_DIR)/xlmr_v25_idf.pt"
	@echo ""
	@$(PYTHON) scripts/compute_idf_weights.py
	@echo "$(GREEN)✓ IDF weights computed$(NC)"

prepare-v25-stopwords: ## Generate Korean stopword mask
	@echo "$(BLUE)Generating Korean stopword mask...$(NC)"
	@mkdir -p $(IDF_WEIGHTS_DIR)
	@$(PYTHON) scripts/generate_stopword_mask.py
	@echo "$(GREEN)✓ Stopword mask generated$(NC)"

prepare-v25-data: prepare-v25-idf prepare-v25-stopwords ## Prepare all V25 data (IDF + stopwords)
	@echo "$(GREEN)✓ V25 data preparation complete$(NC)"

train-v25: ## V25 IDF-aware FLOPS training (XLM-RoBERTa 250K vocab)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V25 IDF-Aware Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V25)"
	@echo "Output: $(OUTPUT_V25)"
	@echo ""
	@echo "$(YELLOW)Key Features:$(NC)"
	@echo "  - Base model: xlm-roberta-base (250K vocab)"
	@echo "  - Teacher: BAAI/bge-m3"
	@echo "  - IDF-aware FLOPS: BM25 IDF weighting"
	@echo "  - Stopword masking: Korean particles/endings"
	@echo ""
	@echo "$(YELLOW)Loss Components:$(NC)"
	@echo "  InfoNCE + Self + Positive + IDF-FLOPS + MinAct + KD"
	@echo ""
	@echo "Mixed precision: BF16"
	@echo "$(BLUE)========================================$(NC)"
	@test -f $(IDF_WEIGHTS_DIR)/xlmr_v25_idf.pt || (echo "$(RED)Error: IDF weights not found. Run: make prepare-v25-idf$(NC)" && exit 1)
	@$(PYTHON) -m src.train v25 --config $(CONFIG_V25)

train-v25-quick: ## Quick V25 training validation (500 samples, 2 epochs)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V25 Quick Training Validation$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Samples: 500"
	@echo "Epochs: 2"
	@echo "Purpose: Verify IDF and stopword masking work correctly"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) scripts/quick_train_v25.py --samples 500 --epochs 2

train-v25-bg: ## V25 training in background (nohup)
	@echo "$(BLUE)Starting V25 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V25)
	@test -f $(IDF_WEIGHTS_DIR)/xlmr_v25_idf.pt || (echo "$(RED)Error: IDF weights not found. Run: make prepare-v25-idf$(NC)" && exit 1)
	@nohup $(PYTHON) -m src.train v25 --config $(CONFIG_V25) > $(OUTPUT_V25)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@sleep 1
	@echo "PID: $$(pgrep -f 'src.train v25' | tail -1)"
	@echo "Log: $(OUTPUT_V25)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v25"
	@echo "  make tensorboard-v25"

train-v25-resume: ## Resume V25 training from checkpoint
	@echo "$(BLUE)Resuming V25 training from checkpoint...$(NC)"
	@$(PYTHON) -m src.train v25 --config $(CONFIG_V25) --resume

train-v25-verify: ## Verify V25 IDF setup (no training, just validation)
	@echo "$(BLUE)Verifying V25 IDF setup...$(NC)"
	@$(PYTHON) scripts/quick_train_v25.py --verify-only

eval-v25: ## Evaluate V25 model on validation set
	@echo "$(BLUE)Evaluating V25 model...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V25_HF_DIR)/best \
		--num-samples 100

eval-v25-sparsity: ## Analyze V25 sparsity (semantic vs stopword tokens)
	@echo "$(BLUE)V25 Sparsity Analysis...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V25_HF_DIR)/best \
		--sparsity-only

eval-v25-compare: ## Compare V24 vs V25 models
	@echo "$(BLUE)Comparing V24 vs V25...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V25_HF_DIR)/best \
		--compare huggingface/v24_best \
		--num-samples 100

convert-v25-hf: ## Convert V25 checkpoint to HuggingFace format
	@echo "$(BLUE)Converting V25 checkpoint to HuggingFace format...$(NC)"
	@mkdir -p $(V25_HF_DIR)
	@$(PYTHON) scripts/convert_checkpoint_to_hf.py \
		--checkpoint $(V25_CHECKPOINT_DIR)/best_model.pt \
		--output $(V25_HF_DIR)/best \
		--model-name xlm-roberta-base
	@echo "$(GREEN)✓ Converted to $(V25_HF_DIR)/best$(NC)"

logs-v25: ## Show V25 training logs (real-time)
	@echo "$(BLUE)V25 Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V25)/training.log ]; then \
		tail -f $(OUTPUT_V25)/training.log; \
	elif [ -f $(OUTPUT_V25)/nohup.out ]; then \
		tail -f $(OUTPUT_V25)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v25$(NC)"; \
	fi

tensorboard-v25: ## Start TensorBoard for V25 training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V25)/tensorboard --port 6006

v25-pipeline: ## Run full V25 pipeline (data -> train -> eval)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V25 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Steps:$(NC)"
	@echo "  1. Compute IDF weights"
	@echo "  2. Generate stopword mask"
	@echo "  3. Run quick validation (500 samples)"
	@echo "  4. Full training (if validation passes)"
	@echo "  5. Evaluation"
	@echo ""
	@echo "$(BLUE)[1/5]$(NC) Computing IDF weights..."
	@$(MAKE) prepare-v25-idf
	@echo ""
	@echo "$(BLUE)[2/5]$(NC) Generating stopword mask..."
	@$(MAKE) prepare-v25-stopwords
	@echo ""
	@echo "$(BLUE)[3/5]$(NC) Running quick validation..."
	@$(MAKE) train-v25-quick
	@echo ""
	@echo "$(YELLOW)Quick validation complete.$(NC)"
	@echo "Review the results above. If semantic tokens are dominant, proceed with:"
	@echo "  make train-v25-bg"
	@echo ""
	@echo "$(GREEN)✓ Pipeline steps 1-3 complete$(NC)"

##@ V26 Enhanced IDF Training Pipeline

prepare-v26-idf: ## Compute IDF weights for V26 (uses same corpus as V25)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Computing IDF Weights for V26$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p $(IDF_WEIGHTS_DIR)
	@echo "$(YELLOW)Corpus:$(NC) data/v24.0/train_*.jsonl"
	@echo "$(YELLOW)Output:$(NC) $(OUTPUT_V26)/idf_weights/xlmr_v26_idf.pt"
	@echo ""
	@$(PYTHON) scripts/compute_idf_weights.py --output-dir $(OUTPUT_V26)/idf_weights --output-name xlmr_v26_idf
	@echo "$(GREEN)✓ IDF weights computed$(NC)"

prepare-v26-data: prepare-v26-idf ## Prepare all V26 data
	@echo "$(GREEN)✓ V26 data preparation complete$(NC)"

train-v26: ## V26 Enhanced IDF training (XLM-RoBERTa with special token fix)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V26 Enhanced IDF Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V26)"
	@echo "Output: $(OUTPUT_V26)"
	@echo ""
	@echo "$(YELLOW)V26 Key Improvements over V25:$(NC)"
	@echo "  - Special tokens excluded from IDF normalization"
	@echo "  - Fixed penalty (100.0) for special tokens"
	@echo "  - lambda_flops: 0.010 (5x increase)"
	@echo "  - stopword_penalty: 15.0 (3x increase)"
	@echo "  - idf_alpha: 4.0 (sharper penalty curve)"
	@echo "  - Extended Korean stopword list"
	@echo ""
	@echo "$(YELLOW)Expected Results:$(NC)"
	@echo "  - semantic_ratio > 1.0 (semantic tokens dominating)"
	@echo "  - Top-10 semantic tokens: 80%+"
	@echo "  - Recall@1 parity with BM25"
	@echo ""
	@echo "Mixed precision: BF16"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) -m src.train v26 --config $(CONFIG_V26)

train-v26-bg: ## V26 training in background (nohup)
	@echo "$(BLUE)Starting V26 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V26)
	@nohup $(PYTHON) -m src.train v26 --config $(CONFIG_V26) > $(OUTPUT_V26)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@sleep 1
	@echo "PID: $$(pgrep -f 'src.train v26' | tail -1)"
	@echo "Log: $(OUTPUT_V26)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v26"
	@echo "  make tensorboard-v26"

train-v26-resume: ## Resume V26 training from checkpoint
	@echo "$(BLUE)Resuming V26 training from checkpoint...$(NC)"
	@$(PYTHON) -m src.train v26 --config $(CONFIG_V26) --resume

eval-v26: ## Evaluate V26 model on validation set
	@echo "$(BLUE)Evaluating V26 model...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V26_HF_DIR)/best \
		--num-samples 100

eval-v26-sparsity: ## Analyze V26 sparsity (semantic vs stopword tokens)
	@echo "$(BLUE)V26 Sparsity Analysis...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V26_HF_DIR)/best \
		--sparsity-only

convert-v26-hf: ## Convert V26 checkpoint to HuggingFace format
	@echo "$(BLUE)Converting V26 checkpoint to HuggingFace format...$(NC)"
	@mkdir -p $(V26_HF_DIR)
	@$(PYTHON) scripts/export_v25_to_huggingface.py \
		--checkpoint $(OUTPUT_V26)/best_model \
		--output $(V26_HF_DIR)
	@echo "$(GREEN)✓ Converted to $(V26_HF_DIR)$(NC)"

logs-v26: ## Show V26 training logs (real-time)
	@echo "$(BLUE)V26 Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V26)/training.log ]; then \
		tail -f $(OUTPUT_V26)/training.log; \
	elif [ -f $(OUTPUT_V26)/nohup.out ]; then \
		tail -f $(OUTPUT_V26)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v26$(NC)"; \
	fi

tensorboard-v26: ## Start TensorBoard for V26 training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V26)/tensorboard --port 6006

v26-pipeline: ## Run full V26 pipeline (data -> train -> eval)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V26 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)V26 addresses V25's stopword dominance:$(NC)"
	@echo "  Root cause: Special tokens compressed IDF range"
	@echo "  Fix: Exclude special tokens + increase penalties"
	@echo ""
	@echo "$(YELLOW)Steps:$(NC)"
	@echo "  1. Compute IDF weights"
	@echo "  2. Full training (25 epochs)"
	@echo "  3. Evaluation and benchmark"
	@echo ""
	@echo "$(BLUE)[1/3]$(NC) Computing IDF weights..."
	@$(MAKE) prepare-v26-idf
	@echo ""
	@echo "$(YELLOW)IDF weights ready.$(NC)"
	@echo "To start training:"
	@echo "  make train-v26-bg    (background)"
	@echo "  make train-v26       (foreground)"
	@echo ""
	@echo "$(GREEN)✓ Pipeline step 1 complete$(NC)"

validate-semantic-ratio: ## Validate model semantic token ratio
	@echo "$(BLUE)Validating semantic token ratio...$(NC)"
	@$(PYTHON) scripts/validate_semantic_ratio.py \
		--model $(V26_HF_DIR) \
		--queries "당뇨병 치료 방법" "서울 맛집 추천" "파이썬 프로그래밍 배우기"

##@ V27 Travel Domain Training Pipeline

collect-travel: ## Collect travel/tourism domain data for V27
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Collecting Travel Domain Data$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)Sources:$(NC)"
	@echo "  - Korean Wikipedia travel categories"
	@echo "  - Template-generated triplets"
	@echo "$(YELLOW)Output:$(NC) $(V27_DATA_DIR)/raw"
	@echo ""
	@mkdir -p $(V27_DATA_DIR)/raw
	@$(PYTHON) scripts/collect_travel_data.py \
		--output $(V27_DATA_DIR)/raw \
		--sources wikipedia,template
	@echo "$(GREEN)✓ Travel data collected$(NC)"

collect-travel-full: ## Collect travel data from all sources (requires Namuwiki dump)
	@echo "$(BLUE)Collecting travel data from all sources...$(NC)"
	@mkdir -p $(V27_DATA_DIR)/raw
	@$(PYTHON) scripts/collect_travel_data.py \
		--output $(V27_DATA_DIR)/raw \
		--sources wikipedia,namuwiki,korpora,template
	@echo "$(GREEN)✓ Full travel data collected$(NC)"

prepare-v27-triplets: ## Generate triplets from collected travel data
	@echo "$(BLUE)Generating travel domain triplets...$(NC)"
	@$(PYTHON) scripts/travel_triplet_generator.py \
		--input $(V27_DATA_DIR)/raw \
		--output $(V27_DATA_DIR) \
		--include-templates
	@echo "$(GREEN)✓ Travel triplets generated$(NC)"

prepare-v27-data: collect-travel prepare-v27-triplets ## Prepare all V27 data (collect + generate triplets)
	@echo "$(GREEN)✓ V27 data preparation complete$(NC)"
	@echo ""
	@$(PYTHON) scripts/collect_travel_data.py --stats

train-v27: ## V27 training with travel domain data
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V27 Travel Domain Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V27)"
	@echo "Output: $(OUTPUT_V27)"
	@echo ""
	@echo "$(YELLOW)V27 Key Improvements over V26:$(NC)"
	@echo "  - Travel/tourism domain data (~110K samples)"
	@echo "  - Location-aware hard negatives"
	@echo "  - Better coverage for travel queries"
	@echo ""
	@echo "$(YELLOW)Expected Results:$(NC)"
	@echo "  - Location tokens (서울, 부산) in top-5"
	@echo "  - Travel domain Recall@1 > 40%"
	@echo ""
	@echo "Mixed precision: BF16"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) -m src.train v27 --config $(CONFIG_V27)

train-v27-bg: ## V27 training in background (nohup)
	@echo "$(BLUE)Starting V27 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V27)
	@nohup $(PYTHON) -m src.train v27 --config $(CONFIG_V27) > $(OUTPUT_V27)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@sleep 1
	@echo "PID: $$(pgrep -f 'src.train v27' | tail -1)"
	@echo "Log: $(OUTPUT_V27)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v27"
	@echo "  make tensorboard-v27"

train-v27-resume: ## Resume V27 training from checkpoint
	@echo "$(BLUE)Resuming V27 training from checkpoint...$(NC)"
	@$(PYTHON) -m src.train v27 --config $(CONFIG_V27) --resume

eval-v27: ## Evaluate V27 model on validation set
	@echo "$(BLUE)Evaluating V27 model...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V27_HF_DIR)/best \
		--num-samples 100

eval-v27-travel: ## Evaluate V27 on travel domain queries
	@echo "$(BLUE)V27 Travel Domain Evaluation...$(NC)"
	@$(PYTHON) scripts/validate_semantic_ratio.py \
		--model $(V27_HF_DIR) \
		--queries "서울 여행 추천" "부산 맛집" "제주도 관광" "강원도 숙소" "경주 여행 코스"

convert-v27-hf: ## Convert V27 checkpoint to HuggingFace format
	@echo "$(BLUE)Converting V27 checkpoint to HuggingFace format...$(NC)"
	@mkdir -p $(V27_HF_DIR)
	@$(PYTHON) scripts/export_v25_to_huggingface.py \
		--checkpoint $(OUTPUT_V27)/best_model \
		--output $(V27_HF_DIR)
	@echo "$(GREEN)✓ Converted to $(V27_HF_DIR)$(NC)"

logs-v27: ## Show V27 training logs (real-time)
	@echo "$(BLUE)V27 Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V27)/training.log ]; then \
		tail -f $(OUTPUT_V27)/training.log; \
	elif [ -f $(OUTPUT_V27)/nohup.out ]; then \
		tail -f $(OUTPUT_V27)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v27$(NC)"; \
	fi

tensorboard-v27: ## Start TensorBoard for V27 training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V27)/tensorboard --port 6006

v27-pipeline: ## Run full V27 pipeline (collect -> prepare -> train -> eval)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V27 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)V27 addresses travel domain coverage:$(NC)"
	@echo "  Root cause: V26 lacks travel/tourism training data"
	@echo "  Fix: Add 110K travel triplets with location negatives"
	@echo ""
	@echo "$(YELLOW)Steps:$(NC)"
	@echo "  1. Collect travel data"
	@echo "  2. Generate triplets"
	@echo "  3. Full training (25 epochs)"
	@echo "  4. Evaluation"
	@echo ""
	@echo "$(BLUE)[1/4]$(NC) Collecting travel data..."
	@$(MAKE) collect-travel
	@echo ""
	@echo "$(BLUE)[2/4]$(NC) Generating triplets..."
	@$(MAKE) prepare-v27-triplets
	@echo ""
	@echo "$(YELLOW)Data preparation complete.$(NC)"
	@echo "To start training:"
	@echo "  make train-v27-bg    (background)"
	@echo "  make train-v27       (foreground)"
	@echo ""
	@echo "$(GREEN)✓ Pipeline steps 1-2 complete$(NC)"

##@ V28 Context-Gated Training Pipeline

# V28 specific directories
V28_DATA_DIR := data/v28.0
V28_CHECKPOINT_DIR := checkpoints/v28.0
V28_HF_DIR := huggingface/v28
CONFIG_V28 := configs/train_v28.yaml
OUTPUT_V28 := outputs/train_v28

build-korean-tokens: ## Build Korean token ID set for V28 language filtering
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Building Korean Token IDs$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p $(OUTPUT_V28)
	@$(PYTHON) -c "from src.train.idf.korean_tokens import build_korean_token_ids, save_korean_token_ids, analyze_token_language_distribution; \
		from transformers import AutoTokenizer; \
		tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base'); \
		analyze_token_language_distribution(tokenizer); \
		ids = build_korean_token_ids(tokenizer); \
		save_korean_token_ids(ids, '$(OUTPUT_V28)/korean_token_ids.json')"
	@echo "$(GREEN)✓ Korean token IDs saved to $(OUTPUT_V28)/korean_token_ids.json$(NC)"

train-v28: ## V28 training with language filtering and context gate
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V28 Context-Gated Training$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V28)"
	@echo "Output: $(OUTPUT_V28)"
	@echo ""
	@echo "$(YELLOW)V28 Key Features:$(NC)"
	@echo "  - V28a: Korean language filtering"
	@echo "  - V28b: Context-gated sparse expansion"
	@echo "  - Inherits V26 IDF-aware FLOPS"
	@echo ""
	@echo "$(YELLOW)Expected Results:$(NC)"
	@echo "  - Korean token ratio > 85%"
	@echo "  - Context discrimination rate > 60%"
	@echo "  - Recall@1 > 40%"
	@echo ""
	@echo "Mixed precision: BF16"
	@echo "$(BLUE)========================================$(NC)"
	@$(PYTHON) -m src.train v28 --config $(CONFIG_V28)

train-v28-bg: ## V28 training in background (nohup)
	@echo "$(BLUE)Starting V28 training in background...$(NC)"
	@mkdir -p $(OUTPUT_V28)
	@nohup $(PYTHON) -m src.train v28 --config $(CONFIG_V28) > $(OUTPUT_V28)/nohup.out 2>&1 &
	@echo "$(GREEN)Training started in background$(NC)"
	@sleep 1
	@echo "PID: $$(pgrep -f 'src.train v28' | tail -1)"
	@echo "Log: $(OUTPUT_V28)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v28"
	@echo "  make tensorboard-v28"

train-v28-resume: ## Resume V28 training from checkpoint
	@echo "$(BLUE)Resuming V28 training from checkpoint...$(NC)"
	@$(PYTHON) -m src.train v28 --config $(CONFIG_V28) --resume

train-v28-after-v27: ## Wait for V27 completion and auto-start V28
	@echo "$(BLUE)Starting V28 auto-start after V27...$(NC)"
	@./scripts/run_v28_after_v27.sh

train-v28a: ## V28a only (language filtering, no context gate)
	@echo "$(BLUE)Starting V28a (Language Filtering Only)...$(NC)"
	@$(PYTHON) -m src.train v28 --config $(CONFIG_V28) --no-context-gate

eval-v28: ## Evaluate V28 model on validation set
	@echo "$(BLUE)Evaluating V28 model...$(NC)"
	@$(PYTHON) scripts/quick_eval.py \
		--model $(V28_HF_DIR)/best \
		--num-samples 100

eval-v28-language: ## Analyze V28 Korean vs non-Korean token ratio
	@echo "$(BLUE)V28 Language Distribution Analysis...$(NC)"
	@$(PYTHON) scripts/validate_semantic_ratio.py \
		--model $(V28_HF_DIR) \
		--queries "토니 베넷의 중간 이름은?" "당뇨병 치료 방법" "서울 맛집 추천"

eval-v28-context: ## Analyze V28 context discrimination
	@echo "$(BLUE)V28 Context Discrimination Analysis...$(NC)"
	@$(PYTHON) -c "from benchmark.encoders import NeuralSparseEncoder; \
		encoder = NeuralSparseEncoder('$(V28_HF_DIR)'); \
		q1 = encoder.encode(['출근했는데 점심 메뉴 추천해줘'])[0]; \
		q2 = encoder.encode(['학교를 갔는데 점심 메뉴 추천해줘'])[0]; \
		overlap = set(q1.keys()) & set(q2.keys()); \
		print(f'Context overlap: {len(overlap)} / {len(set(q1.keys()) | set(q2.keys()))} tokens'); \
		print(f'Overlap ratio: {100*len(overlap)/max(len(q1), len(q2)):.1f}%')"

convert-v28-hf: ## Convert V28 checkpoint to HuggingFace format
	@echo "$(BLUE)Converting V28 checkpoint to HuggingFace format...$(NC)"
	@mkdir -p $(V28_HF_DIR)
	@$(PYTHON) scripts/export_v25_to_huggingface.py \
		--checkpoint $(OUTPUT_V28)/best_model \
		--output $(V28_HF_DIR)
	@echo "$(GREEN)✓ Converted to $(V28_HF_DIR)$(NC)"

logs-v28: ## Show V28 training logs (real-time)
	@echo "$(BLUE)V28 Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V28)/training.log ]; then \
		tail -f $(OUTPUT_V28)/training.log; \
	elif [ -f $(OUTPUT_V28)/nohup.out ]; then \
		tail -f $(OUTPUT_V28)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v28$(NC)"; \
	fi

tensorboard-v28: ## Start TensorBoard for V28 training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V28)/tensorboard --port 6006

v28-pipeline: ## Run full V28 pipeline (build tokens -> train -> eval)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V28 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)V28 addresses multilingual token leakage:$(NC)"
	@echo "  V28a: Korean language filtering"
	@echo "  V28b: Context-gated sparse expansion"
	@echo ""
	@echo "$(YELLOW)Steps:$(NC)"
	@echo "  1. Build Korean token IDs"
	@echo "  2. Full training (25 epochs)"
	@echo "  3. Evaluation"
	@echo ""
	@echo "$(BLUE)[1/3]$(NC) Building Korean token IDs..."
	@$(MAKE) build-korean-tokens
	@echo ""
	@echo "$(YELLOW)Token IDs ready.$(NC)"
	@echo "To start training:"
	@echo "  make train-v28-bg    (background)"
	@echo "  make train-v28       (foreground)"
	@echo ""
	@echo "$(GREEN)✓ Pipeline step 1 complete$(NC)"

##@ V29 Data Pipeline (Expanded Korean Data)

collect-v29-data: ## Collect Korean datasets from HuggingFace (KorQuAD, KLUE, mC4, etc.)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Collecting Korean Datasets$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)Sources:$(NC)"
	@echo "  - KorQuAD 2.0, KLUE-MRC/STS/NLI"
	@echo "  - Ko-StrategyQA, KoAlpaca, Open-Orca-Ko"
	@echo "  - mC4 Korean, Korean Wikipedia, OPUS en-ko"
	@echo "$(YELLOW)Output:$(NC) $(V29_RAW_DIR)"
	@echo ""
	@mkdir -p $(V29_RAW_DIR)
	@$(PYTHON) scripts/collect_korean_datasets.py \
		--output-dir $(V29_RAW_DIR) \
		--max-samples 500000
	@echo "$(GREEN)✓ Korean datasets collected$(NC)"

build-v29-data: ## Merge, deduplicate, and shard all data (fast bash, ~1-2 min)
	@bash scripts/build_v29_data_fast.sh $(V29_DATA_DIR) 100000 0.05

build-v29-data-slow: ## Merge, deduplicate with MinHash LSH (Python, slow)
	@$(PYTHON) scripts/build_v29_data.py \
		--output-dir $(V29_DATA_DIR) \
		--shard-size 100000 \
		--val-ratio 0.05 \
		--seed 42 \
		--dedup-threshold 0.8

v29-data-stats: ## Show V29 data statistics
	@echo "$(BLUE)V29 Data Statistics:$(NC)"
	@if ls $(V29_DATA_DIR)/train_*.jsonl 1>/dev/null 2>&1; then \
		echo "$(YELLOW)Train shards:$(NC)"; \
		wc -l $(V29_DATA_DIR)/train_*.jsonl | tail -1; \
		echo "$(YELLOW)Validation:$(NC)"; \
		wc -l $(V29_DATA_DIR)/val.jsonl; \
		echo "$(YELLOW)Shard count:$(NC)"; \
		ls $(V29_DATA_DIR)/train_*.jsonl | wc -l; \
	else \
		echo "$(RED)No V29 data found. Run: make build-v29-data$(NC)"; \
	fi

##@ V28 DDP Training (Multi-GPU B200 x8)

train-v28-ddp: ## V28 DDP training on multi-GPU (torchrun)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V28 DDP Training (Multi-GPU)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V28_B200)"
	@echo "Output: $(OUTPUT_V28_DDP)"
	@echo ""
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	echo "$(YELLOW)Detected GPUs:$(NC) $$GPU_COUNT"; \
	echo "$(YELLOW)Per-GPU batch:$(NC) 64"; \
	echo "$(YELLOW)Grad accum:$(NC) 4"; \
	echo "$(YELLOW)Effective batch:$(NC) $$(( 64 * 4 * $$GPU_COUNT ))"; \
	echo ""; \
	mkdir -p $(OUTPUT_V28_DDP); \
	export NCCL_DEBUG=INFO; \
	export NCCL_IB_DISABLE=0; \
	export NCCL_NET_GDR_LEVEL=5; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		-m src.train.cli.train_v28_ddp \
		--config $(CONFIG_V28_B200) \
		--output-dir $(OUTPUT_V28_DDP)

train-v28-ddp-bg: ## V28 DDP training in background (nohup)
	@echo "$(BLUE)Starting V28 DDP training in background...$(NC)"
	@mkdir -p $(OUTPUT_V28_DDP)
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5 \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	nohup torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		-m src.train.cli.train_v28_ddp \
		--config $(CONFIG_V28_B200) \
		--output-dir $(OUTPUT_V28_DDP) \
		> $(OUTPUT_V28_DDP)/nohup.out 2>&1 &
	@echo "$(GREEN)DDP training started in background$(NC)"
	@sleep 2
	@echo "PID: $$(pgrep -f 'train_v28_ddp' | head -1)"
	@echo "Log: $(OUTPUT_V28_DDP)/nohup.out"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  make logs-v28-ddp"
	@echo "  make tensorboard-v28-ddp"

train-v28-ddp-resume: ## Resume V28 DDP training from checkpoint
	@echo "$(BLUE)Resuming V28 DDP training from checkpoint...$(NC)"
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	export NCCL_DEBUG=INFO; \
	export NCCL_IB_DISABLE=0; \
	export NCCL_NET_GDR_LEVEL=5; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		-m src.train.cli.train_v28_ddp \
		--config $(CONFIG_V28_B200) \
		--output-dir $(OUTPUT_V28_DDP) \
		--resume

logs-v28-ddp: ## Show V28 DDP training logs (real-time)
	@echo "$(BLUE)V28 DDP Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V28_DDP)/training.log ]; then \
		tail -f $(OUTPUT_V28_DDP)/training.log; \
	elif [ -f $(OUTPUT_V28_DDP)/nohup.out ]; then \
		tail -f $(OUTPUT_V28_DDP)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v28-ddp$(NC)"; \
	fi

tensorboard-v28-ddp: ## Start TensorBoard for V28 DDP training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V28_DDP)/tensorboard --port 6006

##@ V29 Full Pipeline (Data + DDP Training)

v29-pipeline: ## Run full V29 pipeline (collect -> build -> DDP train)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V29 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Pipeline:$(NC)"
	@echo "  1. Collect Korean datasets from HuggingFace"
	@echo "  2. Merge, dedup, shard into V29 training set"
	@echo "  3. DDP training on all available GPUs"
	@echo ""
	@echo "$(BLUE)[1/3]$(NC) Collecting Korean datasets..."
	@$(MAKE) collect-v29-data
	@echo ""
	@echo "$(BLUE)[2/3]$(NC) Building V29 training data..."
	@$(MAKE) build-v29-data
	@echo ""
	@echo "$(BLUE)[3/3]$(NC) Starting DDP training..."
	@$(MAKE) train-v28-ddp
	@echo ""
	@echo "$(GREEN)✓ V29 pipeline complete$(NC)"

v29-pipeline-bg: ## Run V29 pipeline in background (nohup)
	@echo "$(BLUE)Starting V29 pipeline in background...$(NC)"
	@mkdir -p $(OUTPUT_V28_DDP)
	@nohup bash scripts/run_v28_pipeline.sh > $(OUTPUT_V28_DDP)/pipeline.log 2>&1 &
	@echo "$(GREEN)Pipeline started in background$(NC)"
	@sleep 1
	@echo "PID: $$!"
	@echo "Log: $(OUTPUT_V28_DDP)/pipeline.log"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  tail -f $(OUTPUT_V28_DDP)/pipeline.log"
	@echo "  make logs-v28-ddp"

##@ Monitoring

monitor: ## Monitor GPU usage (real-time)
	@echo "$(YELLOW)Monitoring GPU usage (Ctrl+C to exit)...$(NC)"
	@watch -n 1 nvidia-smi

logs-baseline: ## Show baseline training logs
	@echo "$(BLUE)Baseline Training Logs:$(NC)"
	@if [ -f $(OUTPUT_BASELINE)/training_log.jsonl ]; then \
		tail -f $(OUTPUT_BASELINE)/training_log.jsonl; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-baseline$(NC)"; \
	fi

logs-pretrain: ## Show pre-training logs
	@echo "$(BLUE)Pre-training Logs:$(NC)"
	@if [ -f $(OUTPUT_PRETRAIN)/training_log.jsonl ]; then \
		tail -f $(OUTPUT_PRETRAIN)/training_log.jsonl; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-pretrain$(NC)"; \
	fi

logs-finetune: ## Show fine-tuning logs
	@echo "$(BLUE)Fine-tuning Logs:$(NC)"
	@if [ -f $(OUTPUT_FINETUNE)/training_log.jsonl ]; then \
		tail -f $(OUTPUT_FINETUNE)/training_log.jsonl; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-finetune$(NC)"; \
	fi

##@ Cleanup

clean-outputs: ## Remove all training outputs
	@echo "$(YELLOW)Removing training outputs...$(NC)"
	@rm -rf outputs/baseline_dgx
	@rm -rf outputs/pretrain_korean_dgx
	@rm -rf outputs/finetune_msmarco
	@echo "$(GREEN)✓ Training outputs cleaned$(NC)"

clean-data: ## Remove baseline sample data
	@echo "$(YELLOW)Removing baseline sample data...$(NC)"
	@rm -rf dataset/baseline_samples
	@echo "$(GREEN)✓ Baseline data cleaned$(NC)"

clean-cache: ## Clean Python cache files
	@echo "$(YELLOW)Cleaning Python cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

clean: clean-outputs clean-cache ## Clean all generated files (outputs + cache)
	@echo "$(GREEN)✓ All cleaned$(NC)"

##@ Quick Start

quickstart: setup prepare-baseline train-baseline ## Quick start: setup + prepare + train baseline
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Check training logs: make logs-baseline"
	@echo "  2. Run full pre-training: make train-pretrain"
	@echo "  3. Monitor GPU usage: make monitor"

##@ Information

info: ## Show system and training information
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)System Information$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)GPU:$(NC)"
	@$(PYTHON) -c "import torch; print(f'  Name: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB'); print(f'  CUDA: {torch.version.cuda}'); print(f'  BF16: {torch.cuda.is_bf16_supported()}')"
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(YELLOW)PyTorch:$(NC)"
	@$(PYTHON) -c "import torch; print(f'  Version: {torch.__version__}')"
	@echo ""
	@echo "$(YELLOW)Training Configs:$(NC)"
	@echo "  Baseline: $(CONFIG_BASELINE)"
	@echo "  Pre-train: $(CONFIG_PRETRAIN)"
	@echo "  Fine-tune: $(CONFIG_FINETUNE)"
	@echo ""
	@echo "$(YELLOW)Output Directories:$(NC)"
	@echo "  Baseline: $(OUTPUT_BASELINE)"
	@echo "  Pre-train: $(OUTPUT_PRETRAIN)"
	@echo "  Fine-tune: $(OUTPUT_FINETUNE)"
	@echo "$(BLUE)========================================$(NC)"

##@ HuggingFace Benchmark

benchmark-ko-strategyqa: ## Run benchmark on Ko-StrategyQA dataset
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Ko-StrategyQA Benchmark$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Dataset: mteb/Ko-StrategyQA"
	@echo "  - 592 queries"
	@echo "  - 9,251 documents"
	@echo ""
	@$(PYTHON) -m benchmark.hf_runner \
		--dataset ko-strategyqa \
		--output-dir outputs/benchmark_ko_strategyqa \
		--index-suffix kostrategyqa \
		--cleanup
	@echo "$(GREEN)✓ Benchmark complete$(NC)"

benchmark-ko-strategyqa-skip: ## Run Ko-StrategyQA benchmark (skip setup)
	@echo "$(BLUE)Running Ko-StrategyQA benchmark (skip-setup)...$(NC)"
	@$(PYTHON) -m benchmark.hf_runner \
		--dataset ko-strategyqa \
		--output-dir outputs/benchmark_ko_strategyqa \
		--index-suffix kostrategyqa \
		--skip-setup

##@ Development

lint: ## Run code quality checks
	@echo "$(BLUE)Running code quality checks...$(NC)"
	@$(PYTHON) -m black --check src/ || echo "$(YELLOW)Run: make format$(NC)"
	@$(PYTHON) -m flake8 src/ || echo "$(YELLOW)Linting warnings found$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(PYTHON) -m black src/
	@echo "$(GREEN)✓ Code formatted$(NC)"

##@ Notebooks

notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	@$(VENV)/bin/jupyter notebook notebooks/pretraining-neural-sparse-model/

##@ Git

commit: ## Commit changes (prompts for message)
	@echo "$(BLUE)Git status:$(NC)"
	@git status -s
	@echo ""
	@read -p "Commit message: " msg; \
	git add -A && git commit -m "$$msg"

push: ## Push to remote repository
	@echo "$(BLUE)Pushing to remote...$(NC)"
	@git push
	@echo "$(GREEN)✓ Pushed to remote$(NC)"

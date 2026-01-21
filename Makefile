# Makefile for SPLADE-doc Training on DGX Spark (ARM + GB10)
# Optimized for Nvidia DGX Spark with GB10 GPU and ARM64 architecture

.PHONY: help setup test prepare-baseline train-baseline train-pretrain train-finetune clean clean-outputs monitor logs \
	train-v22 train-v22-bg train-v22-resume logs-v22 tensorboard-v22 \
	train-v24 train-v24-bg train-v24-resume eval-v24 logs-v24 tensorboard-v24 \
	prepare-v25-idf prepare-v25-data train-v25 train-v25-bg train-v25-resume train-v25-quick train-v25-verify \
	eval-v25 eval-v25-sparsity convert-v25-hf logs-v25 tensorboard-v25 v25-pipeline

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

# Output directories
OUTPUT_BASELINE := outputs/baseline_dgx
OUTPUT_PRETRAIN := outputs/pretrain_korean_dgx
OUTPUT_FINETUNE := outputs/finetune_msmarco
OUTPUT_V22 := outputs/train_v22
OUTPUT_V24 := outputs/train_v24
OUTPUT_V25 := outputs/train_v25

# V25 specific directories
IDF_WEIGHTS_DIR := outputs/idf_weights
V25_DATA_DIR := data/v25.0
V25_CHECKPOINT_DIR := checkpoints/v25.0
V25_HF_DIR := huggingface/v25

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
	@$(PYTHON) -c " \
from src.train.idf import IDFComputer; \
from transformers import AutoTokenizer; \
import glob; \
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base'); \
computer = IDFComputer(tokenizer); \
files = sorted(glob.glob('data/v24.0/train_*.jsonl')); \
print(f'Loading from {len(files)} files...'); \
computer.compute_from_training_files(files, sample_size=100000); \
computer.save('$(IDF_WEIGHTS_DIR)/xlmr_v25_idf.pt'); \
print('Done!'); \
"
	@echo "$(GREEN)✓ IDF weights computed$(NC)"

prepare-v25-stopwords: ## Generate Korean stopword mask
	@echo "$(BLUE)Generating Korean stopword mask...$(NC)"
	@mkdir -p $(IDF_WEIGHTS_DIR)
	@$(PYTHON) -c " \
from src.train.idf import create_stopword_mask, get_korean_stopword_ids; \
from transformers import AutoTokenizer; \
import torch; \
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base'); \
stopword_ids = get_korean_stopword_ids(tokenizer); \
print(f'Found {len(stopword_ids)} Korean stopword tokens'); \
mask = create_stopword_mask(tokenizer); \
torch.save(mask, '$(IDF_WEIGHTS_DIR)/xlmr_stopword_mask.pt'); \
print('Saved to $(IDF_WEIGHTS_DIR)/xlmr_stopword_mask.pt'); \
"
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

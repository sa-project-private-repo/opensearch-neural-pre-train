# Makefile for SPLADE-doc Training on B200 x8 DDP

.PHONY: help setup info monitor \
	compute-idf-rust collect-v29-data build-v29-data v29-data-stats mine-hard-negatives \
	train-v28-ddp train-v28-ddp-bg train-v28-ddp-resume logs-v28-ddp tensorboard-v28-ddp \
	v29-pipeline v29-pipeline-bg \
	benchmark-ko-strategyqa lint format clean-cache

# Default target
.DEFAULT_GOAL := help

# Virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Configuration
CONFIG_V28_B200 := configs/train_v28_b200.yaml

# Output directories
OUTPUT_V28_DDP := outputs/train_v28_ddp
IDF_WEIGHTS_DIR := outputs/idf_weights

# V29 data directories
V29_DATA_DIR := data/v29.0
V29_RAW_DIR := data/v29.0/raw

# Rust IDF tool
IDF_COMPUTE_BIN := tools/idf-compute/target/release/idf-compute

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

##@ General

help: ## Display this help message
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)SPLADE-doc Training - B200 x8 DDP$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)GPU:$(NC) NVIDIA B200 x8 (183GB VRAM each)"
	@echo "$(YELLOW)Optimization:$(NC) BF16 mixed precision + DDP"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(BLUE)<target>$(NC)\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  $(BLUE)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup: ## Setup and test the environment
	@echo "$(BLUE)[1/2]$(NC) Activating virtual environment..."
	@test -d $(VENV) || (echo "$(RED)Error: venv not found. Run: python3 -m venv .venv$(NC)" && exit 1)
	@echo "$(GREEN)OK$(NC) Virtual environment found"
	@echo ""
	@echo "$(BLUE)[2/2]$(NC) Running GPU environment test..."
	@$(PYTHON) test_dgx_setup.py

info: ## Show system and training information
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)System Information$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(YELLOW)GPU:$(NC)"
	@$(PYTHON) -c "import torch; \
		n = torch.cuda.device_count(); \
		print(f'  Count: {n}'); \
		print(f'  Name: {torch.cuda.get_device_name(0)}'); \
		print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB each'); \
		print(f'  CUDA: {torch.version.cuda}'); \
		print(f'  BF16: {torch.cuda.is_bf16_supported()}')" 2>/dev/null || \
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
	@echo ""
	@echo "$(YELLOW)Config:$(NC) $(CONFIG_V28_B200)"
	@echo "$(YELLOW)Output:$(NC) $(OUTPUT_V28_DDP)"
	@echo "$(BLUE)========================================$(NC)"

monitor: ## Monitor GPU usage (real-time)
	@watch -n 1 nvidia-smi

##@ V29 Data Pipeline

$(IDF_COMPUTE_BIN):
	@echo "$(BLUE)Building Rust IDF compute tool...$(NC)"
	@source "$$HOME/.cargo/env" && cd tools/idf-compute && cargo build --release
	@echo "$(GREEN)OK idf-compute built$(NC)"

compute-idf-rust: $(IDF_COMPUTE_BIN) ## Compute IDF weights with Rust (~30s)
	@echo "$(BLUE)Computing IDF weights (Rust parallel)...$(NC)"
	@$(IDF_COMPUTE_BIN) \
		--tokenizer xlm-roberta-base \
		--input "data/v29.0/train_shard_*.jsonl" \
		--output $(IDF_WEIGHTS_DIR)/xlmr_v29_idf \
		--smoothing bm25
	@$(PYTHON) tools/idf-compute/load_idf.py $(IDF_WEIGHTS_DIR)/xlmr_v29_idf
	@echo "$(GREEN)OK IDF weights ready: $(IDF_WEIGHTS_DIR)/xlmr_v29_idf.pt$(NC)"

collect-v29-data: ## Collect Korean datasets from HuggingFace
	@echo "$(BLUE)Collecting Korean Datasets...$(NC)"
	@mkdir -p $(V29_RAW_DIR)
	@$(PYTHON) scripts/collect_korean_datasets.py \
		--output-dir $(V29_RAW_DIR) \
		--max-samples 500000
	@echo "$(GREEN)OK Korean datasets collected$(NC)"

build-v29-data: ## Merge, deduplicate, and shard data (bash, ~1-2 min)
	@bash scripts/build_v29_data_fast.sh $(V29_DATA_DIR) 100000 0.05

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

mine-hard-negatives: ## Mine BM25 hard negatives for V29 data
	@echo "Mining BM25 hard negatives..."
	source .venv/bin/activate && python scripts/mine_hard_negatives.py \
		--data-dir data/v29.0 \
		--max-corpus 500000

##@ V28 DDP Training (B200 x8)

train-v28-ddp: ## DDP training on multi-GPU (foreground)
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)Starting V28 DDP Training (Multi-GPU)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo "Config: $(CONFIG_V28_B200)"
	@echo "Output: $(OUTPUT_V28_DDP)"
	@echo ""
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	echo "$(YELLOW)Detected GPUs:$(NC) $$GPU_COUNT"; \
	echo ""; \
	mkdir -p $(OUTPUT_V28_DDP); \
	export NCCL_DEBUG=INFO; \
	export NCCL_IB_DISABLE=0; \
	export NCCL_NET_GDR_LEVEL=5; \
	export NCCL_NVLS_ENABLE=0; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	$(VENV)/bin/torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		-m src.train.cli.train_v28_ddp \
		--config $(CONFIG_V28_B200) \
		--output-dir $(OUTPUT_V28_DDP)

train-v28-ddp-bg: ## DDP training in background (nohup)
	@echo "$(BLUE)Starting V28 DDP training in background...$(NC)"
	@mkdir -p $(OUTPUT_V28_DDP)
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	export NCCL_DEBUG=INFO; \
	export NCCL_IB_DISABLE=0; \
	export NCCL_NET_GDR_LEVEL=5; \
	export NCCL_NVLS_ENABLE=0; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	nohup $(VENV)/bin/torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		--redirects 3 \
		--log-dir $(OUTPUT_V28_DDP)/worker_logs \
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

train-v28-ddp-resume: ## Resume DDP training from checkpoint
	@echo "$(BLUE)Resuming V28 DDP training from checkpoint...$(NC)"
	@GPU_COUNT=$$(nvidia-smi -L 2>/dev/null | wc -l || echo "0"); \
	export NCCL_DEBUG=INFO; \
	export NCCL_IB_DISABLE=0; \
	export NCCL_NET_GDR_LEVEL=5; \
	export NCCL_NVLS_ENABLE=0; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	$(VENV)/bin/torchrun \
		--nproc_per_node=$$GPU_COUNT \
		--master_addr=localhost \
		--master_port=29500 \
		-m src.train.cli.train_v28_ddp \
		--config $(CONFIG_V28_B200) \
		--output-dir $(OUTPUT_V28_DDP) \
		--resume

logs-v28-ddp: ## Show DDP training logs (real-time)
	@echo "$(BLUE)V28 DDP Training Logs:$(NC)"
	@if [ -f $(OUTPUT_V28_DDP)/training.log ]; then \
		tail -f $(OUTPUT_V28_DDP)/training.log; \
	elif [ -f $(OUTPUT_V28_DDP)/nohup.out ]; then \
		tail -f $(OUTPUT_V28_DDP)/nohup.out; \
	else \
		echo "$(RED)No logs found. Start training first with: make train-v28-ddp$(NC)"; \
	fi

tensorboard-v28-ddp: ## Start TensorBoard for DDP training
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "URL: http://localhost:6006"
	@$(VENV)/bin/tensorboard --logdir $(OUTPUT_V28_DDP)/tensorboard --port 6006

##@ Full Pipeline

v29-pipeline: ## Full pipeline: collect -> build -> DDP train
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(GREEN)V29 Full Training Pipeline$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)[1/3]$(NC) Collecting Korean datasets..."
	@$(MAKE) collect-v29-data
	@echo ""
	@echo "$(BLUE)[2/3]$(NC) Building V29 training data..."
	@$(MAKE) build-v29-data
	@echo ""
	@echo "$(BLUE)[3/3]$(NC) Starting DDP training..."
	@$(MAKE) train-v28-ddp

v29-pipeline-bg: ## Full pipeline in background
	@echo "$(BLUE)Starting V29 pipeline in background...$(NC)"
	@mkdir -p $(OUTPUT_V28_DDP)
	@nohup bash scripts/run_v28_pipeline.sh > $(OUTPUT_V28_DDP)/pipeline.log 2>&1 &
	@echo "$(GREEN)Pipeline started in background$(NC)"
	@sleep 1
	@echo "Log: $(OUTPUT_V28_DDP)/pipeline.log"
	@echo ""
	@echo "$(YELLOW)Monitor with:$(NC)"
	@echo "  tail -f $(OUTPUT_V28_DDP)/pipeline.log"
	@echo "  make logs-v28-ddp"

##@ Benchmark

benchmark-ko-strategyqa: ## Run Ko-StrategyQA benchmark (592 queries, 9251 docs)
	@$(PYTHON) -m benchmark.hf_runner \
		--dataset ko-strategyqa \
		--output-dir outputs/benchmark_ko_strategyqa \
		--index-suffix kostrategyqa \
		--cleanup

##@ Development

lint: ## Run code quality checks
	@$(PYTHON) -m black --check src/ || echo "$(YELLOW)Run: make format$(NC)"
	@$(PYTHON) -m flake8 src/ || echo "$(YELLOW)Linting warnings found$(NC)"

format: ## Format code with black
	@$(PYTHON) -m black src/
	@echo "$(GREEN)OK Code formatted$(NC)"

clean-cache: ## Clean Python cache files
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)OK Cache cleaned$(NC)"

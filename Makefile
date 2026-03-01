# Makefile for V33 SPLADEModernBERT Training (B200 x8 DDP)

.PHONY: help setup test train train-bg train-resume \
	benchmark benchmark-ko-strategyqa benchmark-miracl benchmark-mrtydi \
	export-hf tensorboard logs monitor \
	lint format clean clean-outputs clean-cache \
	upload-data download-data upload-outputs download-outputs info

# ============================================================================
# Variables
# ============================================================================

PYTHON ?= python
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate
V33_OUTPUT := outputs/train_v33
V33_CONFIG := configs/train_v33.yaml
V33_CHECKPOINT := $(V33_OUTPUT)/final_model/model.pt
V33_LAUNCH := scripts/launch_v33_b200.sh
HF_DIR := huggingface/v33
BENCHMARK_OUTPUT := outputs/benchmarks/v33

# S3 paths
S3_BUCKET := s3://sewoong-ml-assets/opensearch-neural-pre-train
S3_DATA := $(S3_BUCKET)/data.tar.gz
S3_OUTPUTS := $(S3_BUCKET)/outputs.tar.gz

# ============================================================================
# Help
# ============================================================================

help: ## Display this help message
	@echo "V33 SPLADEModernBERT Training Pipeline"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Setup
# ============================================================================

setup: ## Setup Python venv and install dependencies
	@echo "Setting up environment..."
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	@echo "Setup complete. Run: source $(VENV)/bin/activate"

test: ## Test GPU and model setup
	@echo "=== GPU Test ==="
	$(ACTIVATE) && $(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
	@echo "=== Model Import Test ==="
	$(ACTIVATE) && $(PYTHON) -c "from src.model import SPLADEModernBERT, SPLADELossV33; print('OK')"
	@echo "=== Config Import Test ==="
	$(ACTIVATE) && $(PYTHON) -c "from src.train.config.v33 import V33Config; print('OK')"
	@echo "=== Encoder Import Test ==="
	$(ACTIVATE) && $(PYTHON) -c "from benchmark.encoders import NeuralSparseEncoderV33, BgeM3Encoder; print('OK')"

# ============================================================================
# Training
# ============================================================================

train: ## Start V33 DDP training (foreground)
	$(ACTIVATE) && bash $(V33_LAUNCH)

train-bg: ## Start V33 DDP training (background)
	nohup bash -c '$(ACTIVATE) && bash $(V33_LAUNCH)' > $(V33_OUTPUT)/nohup.log 2>&1 &
	@echo "Training started in background. Logs: $(V33_OUTPUT)/nohup.log"
	@echo "Monitor: make logs"

train-resume: ## Resume V33 DDP training from checkpoint
	$(ACTIVATE) && RESUME=1 bash $(V33_LAUNCH)

# ============================================================================
# Benchmark
# ============================================================================

benchmark: benchmark-ko-strategyqa benchmark-miracl benchmark-mrtydi ## Run all benchmarks

benchmark-ko-strategyqa: ## Benchmark on Ko-StrategyQA
	$(ACTIVATE) && $(PYTHON) -m benchmark.hf_runner \
		--dataset ko-strategyqa \
		--checkpoint $(V33_CHECKPOINT) \
		--output-dir $(BENCHMARK_OUTPUT)/ko-strategyqa \
		--cleanup

benchmark-miracl: ## Benchmark on MIRACL-ko
	$(ACTIVATE) && $(PYTHON) -m benchmark.hf_runner \
		--dataset miracl-ko \
		--checkpoint $(V33_CHECKPOINT) \
		--output-dir $(BENCHMARK_OUTPUT)/miracl-ko \
		--cleanup

benchmark-mrtydi: ## Benchmark on Mr.TyDi-ko
	$(ACTIVATE) && $(PYTHON) -m benchmark.hf_runner \
		--dataset mrtydi-ko \
		--checkpoint $(V33_CHECKPOINT) \
		--output-dir $(BENCHMARK_OUTPUT)/mrtydi-ko \
		--cleanup

# ============================================================================
# Export & Serve
# ============================================================================

export-hf: ## Export V33 model to HuggingFace format
	$(ACTIVATE) && $(PYTHON) scripts/export_v33_hf.py \
		--checkpoint $(V33_CHECKPOINT) \
		--output-dir $(HF_DIR)

# ============================================================================
# Monitoring
# ============================================================================

tensorboard: ## Start TensorBoard for V33 training
	$(ACTIVATE) && tensorboard --logdir=$(V33_OUTPUT)/tensorboard --port=6006 --bind_all

logs: ## Tail V33 training logs
	tail -f $(V33_OUTPUT)/training.log

monitor: ## Monitor GPU usage
	watch -n 1 nvidia-smi

info: ## Show system and training information
	@echo "=== System ==="
	@uname -a
	@echo ""
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
	@echo ""
	@echo "=== Training Output ==="
	@ls -lh $(V33_OUTPUT)/*.log 2>/dev/null || echo "No training logs"
	@echo ""
	@echo "=== Model Checkpoint ==="
	@ls -lh $(V33_CHECKPOINT) 2>/dev/null || echo "No checkpoint"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run code quality checks
	$(ACTIVATE) && ruff check src/ benchmark/ scripts/

format: ## Format code with ruff
	$(ACTIVATE) && ruff format src/ benchmark/ scripts/
	$(ACTIVATE) && ruff check --fix src/ benchmark/ scripts/

# ============================================================================
# Data Sync (S3)
# ============================================================================

upload-data: ## Compress and upload data/ to S3
	@echo "Compressing data/..."
	tar czf /tmp/data.tar.gz data/
	@echo "Uploading to $(S3_DATA)..."
	aws s3 cp /tmp/data.tar.gz $(S3_DATA)
	rm /tmp/data.tar.gz

download-data: ## Download and extract data/ from S3
	@echo "Downloading from $(S3_DATA)..."
	aws s3 cp $(S3_DATA) /tmp/data.tar.gz
	@echo "Extracting..."
	tar xzf /tmp/data.tar.gz
	rm /tmp/data.tar.gz

upload-outputs: ## Compress and upload outputs/ to S3
	@echo "Compressing outputs/..."
	tar czf /tmp/outputs.tar.gz outputs/
	@echo "Uploading to $(S3_OUTPUTS)..."
	aws s3 cp /tmp/outputs.tar.gz $(S3_OUTPUTS)
	rm /tmp/outputs.tar.gz

download-outputs: ## Download and extract outputs/ from S3
	@echo "Downloading from $(S3_OUTPUTS)..."
	aws s3 cp $(S3_OUTPUTS) /tmp/outputs.tar.gz
	@echo "Extracting..."
	tar xzf /tmp/outputs.tar.gz
	rm /tmp/outputs.tar.gz

# ============================================================================
# Cleanup
# ============================================================================

clean-outputs: ## Remove all training outputs
	rm -rf outputs/

clean-cache: ## Clean Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean: clean-outputs clean-cache ## Clean all generated files

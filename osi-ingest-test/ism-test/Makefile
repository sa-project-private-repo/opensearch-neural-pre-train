.PHONY: all build clean run test deps

BINARY := ism-log-sender
GO := /usr/local/go/bin/go

all: build

deps:
	$(GO) mod tidy

build: deps
	$(GO) build -o $(BINARY) .

clean:
	rm -f $(BINARY)

run: build
	./$(BINARY)

# Send logs with custom options
# Usage: make send INTERVAL=2s BATCH=5 COUNT=100
INTERVAL ?= 1s
BATCH ?= 3
COUNT ?= 0

send: build
	./$(BINARY) -interval=$(INTERVAL) -batch=$(BATCH) -count=$(COUNT)

# Quick test: send 10 documents
test: build
	./$(BINARY) -interval=500ms -batch=5 -count=10

# Long running test for ISM
# Usage: make ism-test INTERVAL=1s BATCH=50 COUNT=0
ISM_INTERVAL ?= 1s
ISM_BATCH ?= 50
ISM_COUNT ?= 0

ism-test: build
	./$(BINARY) -interval=$(ISM_INTERVAL) -batch=$(ISM_BATCH) -count=$(ISM_COUNT)

# Help
help:
	@echo "Usage:"
	@echo "  make build     - Build binary"
	@echo "  make run       - Build and run with defaults"
	@echo "  make send      - Run with custom options (INTERVAL, BATCH, COUNT)"
	@echo "  make test      - Quick test (10 docs)"
	@echo "  make ism-test  - Long running ISM test (default: 50/sec)"
	@echo "  make clean     - Remove binary"
	@echo ""
	@echo "Examples:"
	@echo "  make send INTERVAL=2s BATCH=5"
	@echo "  make send INTERVAL=1s BATCH=10 COUNT=1000"
	@echo "  make ism-test ISM_BATCH=100 ISM_INTERVAL=500ms"
	@echo "  make ism-test ISM_COUNT=10000"

#!/bin/bash
# AI Hub Dataset Downloader for V29 Training
# Downloads approved datasets: 624 (web corpus), 86 (emotion), 71828 (AI instructor)

set -e

APIKEY="$AIHUB_KEY"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/aihub"
AIHUB_SHELL="$PROJECT_ROOT/aihubshell"

# Check if aihubshell exists
if [ ! -f "$AIHUB_SHELL" ]; then
    echo "Error: aihubshell not found at $AIHUB_SHELL"
    echo "Please download from AI Hub and place in project root"
    exit 1
fi

# Check if aihubshell is executable
if [ ! -x "$AIHUB_SHELL" ]; then
    chmod +x "$AIHUB_SHELL"
fi

echo "============================================"
echo "AI Hub Dataset Downloader"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo ""

# Create data directories
mkdir -p "$DATA_DIR/624"
mkdir -p "$DATA_DIR/86"
mkdir -p "$DATA_DIR/71828"

# Dataset 624: 대규모 웹데이터 기반 한국어 말뭉치 (9.6GB)
echo "[1/3] Downloading Dataset 624: 대규모 웹데이터 기반 한국어 말뭉치"
echo "      Size: ~9.6GB (news articles, 1B+ words)"
echo "      Use: Document corpus for training"
cd "$DATA_DIR/624"
"$AIHUB_SHELL" -mode d -datasetkey 624 -aihubapikey "$APIKEY"

# Dataset 86: 감성 대화 말뭉치 (20.3MB)
echo ""
echo "[2/3] Downloading Dataset 86: 감성 대화 말뭉치"
echo "      Size: ~20.3MB (270K emotional dialogues)"
echo "      Use: Query-like dialog pairs"
cd "$DATA_DIR/86"
"$AIHUB_SHELL" -mode d -datasetkey 86 -aihubapikey "$APIKEY"

# Dataset 71828: AI 교관 데이터 (5.66GB)
echo ""
echo "[3/3] Downloading Dataset 71828: AI 교관 데이터"
echo "      Size: ~5.66GB (12K Q&A pairs)"
echo "      Use: Q&A triplets for training"
cd "$DATA_DIR/71828"
"$AIHUB_SHELL" -mode d -datasetkey 71828 -aihubapikey "$APIKEY"

cd "$PROJECT_ROOT"

echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "Downloaded to: $DATA_DIR/"
echo "  - 624/   : Web corpus (9.6GB)"
echo "  - 86/    : Emotion dialogs (20.3MB)"
echo "  - 71828/ : AI instructor (5.66GB)"
echo ""
echo "Next steps:"
echo "  1. Run preprocessing: python -m src.preprocessing.convert_aihub"
echo "  2. Mine hard negatives: python -m src.preprocessing.mine_negatives"
echo "  3. Start training: make train-v29-ddp"

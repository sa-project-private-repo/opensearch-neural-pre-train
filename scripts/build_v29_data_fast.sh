#!/usr/bin/env bash
# Fast V29 data build: streaming hash dedup + bash sharding
# Replaces slow Python MinHash LSH with exact query-based dedup
# Expected time: ~1-2 minutes for 4.4M records
set -euo pipefail

OUTPUT_DIR="${1:-data/v29.0}"
SHARD_SIZE="${2:-100000}"
VAL_RATIO="${3:-0.05}"
TMP_DIR="/tmp/v29_build_$$"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Fast V29 Data Build (hash dedup)${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Merge all sources
echo -e "${YELLOW}[1/5]${NC} Merging all sources..."
cat \
    data/v29.0/raw/*.jsonl \
    data/v24.0/train_shard_*.jsonl \
    data/v27.0/train_*.jsonl \
    data/aihub/processed/*.jsonl \
    > "$TMP_DIR/merged.jsonl" 2>/dev/null

TOTAL_BEFORE=$(wc -l < "$TMP_DIR/merged.jsonl")
echo "  Total records: $TOTAL_BEFORE"

# Step 2: Streaming hash dedup on query field
echo -e "${YELLOW}[2/5]${NC} Deduplicating (exact query hash)..."
python3 -c "
import json, sys, hashlib

seen = set()
kept = 0
for i, line in enumerate(sys.stdin, 1):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        q = obj.get('query', '')
        h = hashlib.md5(q.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            sys.stdout.write(line + '\n')
            kept += 1
    except (json.JSONDecodeError, KeyError):
        continue
    if i % 500000 == 0:
        print(f'  Processed {i:,} / deduped to {kept:,}', file=sys.stderr)

print(f'  Final: {kept:,} unique records from {i:,} total', file=sys.stderr)
" < "$TMP_DIR/merged.jsonl" > "$TMP_DIR/deduped.jsonl"

TOTAL_AFTER=$(wc -l < "$TMP_DIR/deduped.jsonl")
REMOVED=$((TOTAL_BEFORE - TOTAL_AFTER))
echo "  Before: $TOTAL_BEFORE -> After: $TOTAL_AFTER (removed $REMOVED duplicates)"

# Step 3: Shuffle
echo -e "${YELLOW}[3/5]${NC} Shuffling..."
shuf "$TMP_DIR/deduped.jsonl" > "$TMP_DIR/shuffled.jsonl"

# Step 4: Split train/val
echo -e "${YELLOW}[4/5]${NC} Splitting train/val (val_ratio=$VAL_RATIO)..."
VAL_COUNT=$(python3 -c "import math; print(math.ceil($TOTAL_AFTER * $VAL_RATIO))")
head -n "$VAL_COUNT" "$TMP_DIR/shuffled.jsonl" > "$OUTPUT_DIR/val.jsonl"
tail -n +"$((VAL_COUNT + 1))" "$TMP_DIR/shuffled.jsonl" > "$TMP_DIR/train_all.jsonl"
TRAIN_COUNT=$((TOTAL_AFTER - VAL_COUNT))
echo "  Train: $TRAIN_COUNT / Val: $VAL_COUNT"

# Step 5: Shard
echo -e "${YELLOW}[5/5]${NC} Sharding (shard_size=$SHARD_SIZE)..."
# Remove old shards
rm -f "$OUTPUT_DIR"/train_shard_*.jsonl

cd "$OUTPUT_DIR"
split -l "$SHARD_SIZE" -d -a 3 --additional-suffix=.jsonl "$TMP_DIR/train_all.jsonl" train_shard_
cd - > /dev/null

SHARD_COUNT=$(ls "$OUTPUT_DIR"/train_shard_*.jsonl | wc -l)

# Generate metadata
python3 -c "
import json, os, glob
shards = sorted(glob.glob('$OUTPUT_DIR/train_shard_*.jsonl'))
meta = {
    'total_train': $TRAIN_COUNT,
    'total_val': $VAL_COUNT,
    'total_unique': $TOTAL_AFTER,
    'total_before_dedup': $TOTAL_BEFORE,
    'duplicates_removed': $REMOVED,
    'shard_size': $SHARD_SIZE,
    'num_shards': len(shards),
    'val_ratio': $VAL_RATIO,
    'dedup_method': 'exact_query_md5',
    'shards': [os.path.basename(s) for s in shards]
}
with open('$OUTPUT_DIR/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

# Cleanup
rm -rf "$TMP_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}V29 Data Build Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  Train: ${GREEN}$TRAIN_COUNT${NC} records ($SHARD_COUNT shards)"
echo -e "  Val:   ${GREEN}$VAL_COUNT${NC} records"
echo -e "  Dedup: removed $REMOVED duplicates ($(python3 -c "print(f'{$REMOVED/$TOTAL_BEFORE*100:.1f}%')"))"
echo -e "  Output: $OUTPUT_DIR/"

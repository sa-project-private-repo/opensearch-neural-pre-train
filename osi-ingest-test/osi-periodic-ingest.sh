#!/bin/bash

# OSI Periodic Ingestion Script
# Usage: ./osi-periodic-ingest.sh [interval_seconds] [batch_size]

OSI_ENDPOINT="https://osi-vpc-internal-connection-6scbjbll7ezxlhrbigvyzx3squ.us-east-1.osis.amazonaws.com"
OSI_PATH="/http-094569/logs"
REGION="us-east-1"

INTERVAL="${1:-1}"
BATCH_SIZE="${2:-3}"

LEVELS=("DEBUG" "INFO" "WARN" "ERROR" "FATAL")
SOURCES=("api-gateway" "auth-service" "user-service" "payment-service" "order-service")
ACTIONS=("login" "logout" "create" "update" "delete" "read" "search" "export")
STATUSES=("success" "failure" "pending" "timeout" "cancelled")

random_element() {
    local arr=("$@")
    echo "${arr[$RANDOM % ${#arr[@]}]}"
}

random_ip() {
    echo "$((RANDOM % 255 + 1)).$((RANDOM % 256)).$((RANDOM % 256)).$((RANDOM % 256))"
}

generate_record() {
    local ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local level=$(random_element "${LEVELS[@]}")
    local source=$(random_element "${SOURCES[@]}")
    local action=$(random_element "${ACTIONS[@]}")
    local status=$(random_element "${STATUSES[@]}")
    local user_id="user-$((RANDOM % 10000 + 1))"
    local request_id=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)
    local duration=$((RANDOM % 5000 + 1))
    local ip=$(random_ip)

    cat <<EOF
{"timestamp":"$ts","level":"$level","source":"$source","action":"$action","status":"$status","user_id":"$user_id","request_id":"$request_id","duration_ms":$duration,"ip":"$ip"}
EOF
}

generate_batch() {
    local size=$1
    local records=""
    for ((i=0; i<size; i++)); do
        [[ -n "$records" ]] && records+=","
        records+=$(generate_record)
    done
    echo "[$records]"
}

send_batch() {
    local data="$1"
    awscurl --service osis --region "$REGION" \
        -X POST "${OSI_ENDPOINT}${OSI_PATH}" \
        -H "Content-Type: application/json" \
        -d "$data" 2>&1
}

echo "OSI Periodic Ingestion"
echo "  Interval: ${INTERVAL}s"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Press Ctrl+C to stop"
echo ""

count=0
while true; do
    batch=$(generate_batch "$BATCH_SIZE")
    result=$(send_batch "$batch")
    count=$((count + BATCH_SIZE))
    ts=$(date +"%H:%M:%S")

    if [[ "$result" == *"200 OK"* ]]; then
        echo "[$ts] Sent $BATCH_SIZE docs (total: $count)"
    else
        echo "[$ts] Failed: $result"
    fi

    sleep "$INTERVAL"
done

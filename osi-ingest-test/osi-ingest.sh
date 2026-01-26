#!/bin/bash

# OpenSearch Ingestion Pipeline Script
# Usage: ./osi-ingest.sh [options]

OSI_ENDPOINT="https://osi-vpc-internal-connection-6scbjbll7ezxlhrbigvyzx3squ.us-east-1.osis.amazonaws.com"
OSI_PATH="/http-094569/logs"
REGION="us-east-1"
OPENSEARCH_ENDPOINT="https://vpc-vpc-connection-osi-gkmktczxestjbjh2d4qobybv2i.us-east-1.es.amazonaws.com"
INDEX_NAME="http-094569-index-$(date +%Y-%m-%d)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  send <json_array>    Send JSON array data to OSI pipeline"
    echo "  send-file <file>     Send JSON array from file to OSI pipeline"
    echo "  test                 Send test data to OSI pipeline"
    echo "  search [query]       Search indexed data (optional query string)"
    echo "  count                Get document count in today's index"
    echo "  indices              List all indices"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 send '[{\"message\": \"hello\"}]'"
    echo "  $0 send-file data.json"
    echo "  $0 search"
    echo "  $0 search 'message:error'"
    exit 1
}

send_data() {
    local data="$1"
    echo -e "${YELLOW}Sending data to OSI pipeline...${NC}"

    response=$(awscurl --service osis --region "$REGION" \
        -X POST "${OSI_ENDPOINT}${OSI_PATH}" \
        -H "Content-Type: application/json" \
        -d "$data" 2>&1)

    if [[ "$response" == *"200 OK"* ]]; then
        echo -e "${GREEN}✓ Data sent successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to send data: $response${NC}"
        return 1
    fi
}

send_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}✗ File not found: $file${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Sending data from file: $file${NC}"

    response=$(awscurl --service osis --region "$REGION" \
        -X POST "${OSI_ENDPOINT}${OSI_PATH}" \
        -H "Content-Type: application/json" \
        -d "@$file" 2>&1)

    if [[ "$response" == *"200 OK"* ]]; then
        echo -e "${GREEN}✓ Data sent successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to send data: $response${NC}"
        return 1
    fi
}

send_test() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local test_data="[
        {\"timestamp\": \"$timestamp\", \"message\": \"Test log entry\", \"level\": \"INFO\", \"source\": \"osi-ingest-script\"},
        {\"timestamp\": \"$timestamp\", \"message\": \"Another test entry\", \"level\": \"DEBUG\", \"source\": \"osi-ingest-script\"}
    ]"

    echo -e "${YELLOW}Sending test data...${NC}"
    echo "$test_data" | jq . 2>/dev/null || echo "$test_data"
    echo ""

    send_data "$test_data"
}

search_data() {
    local query="${1:-*}"
    echo -e "${YELLOW}Searching index: $INDEX_NAME${NC}"
    echo -e "${YELLOW}Query: $query${NC}"
    echo ""

    awscurl --service es --region "$REGION" \
        "${OPENSEARCH_ENDPOINT}/${INDEX_NAME}/_search?pretty&q=$query"
}

count_docs() {
    echo -e "${YELLOW}Document count in: $INDEX_NAME${NC}"
    awscurl --service es --region "$REGION" \
        "${OPENSEARCH_ENDPOINT}/${INDEX_NAME}/_count?pretty"
}

list_indices() {
    echo -e "${YELLOW}Listing all indices...${NC}"
    awscurl --service es --region "$REGION" \
        "${OPENSEARCH_ENDPOINT}/_cat/indices?v&s=index"
}

# Main
case "${1:-}" in
    send)
        [[ -z "${2:-}" ]] && usage
        send_data "$2"
        ;;
    send-file)
        [[ -z "${2:-}" ]] && usage
        send_file "$2"
        ;;
    test)
        send_test
        ;;
    search)
        search_data "${2:-}"
        ;;
    count)
        count_docs
        ;;
    indices)
        list_indices
        ;;
    *)
        usage
        ;;
esac

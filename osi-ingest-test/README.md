# OSI Ingest Test

OpenSearch Ingestion 파이프라인 테스트 스크립트

## 스크립트

### osi-ingest.sh

OSI 파이프라인에 데이터 전송 및 OpenSearch 조회

```bash
./osi-ingest.sh [command] [options]
```

| Command | Description |
|---------|-------------|
| `send <json_array>` | JSON 배열 데이터를 OSI 파이프라인에 전송 |
| `send-file <file>` | 파일에서 JSON 배열을 읽어서 전송 |
| `test` | 테스트 데이터 전송 |
| `search [query]` | 색인된 데이터 검색 (선택적 쿼리) |
| `count` | 오늘 인덱스의 문서 수 조회 |
| `indices` | 모든 인덱스 목록 조회 |

**Examples:**
```bash
# 테스트 데이터 전송
./osi-ingest.sh test

# JSON 배열 직접 전송
./osi-ingest.sh send '[{"message": "hello"}]'

# 파일에서 전송
./osi-ingest.sh send-file data.json

# 검색
./osi-ingest.sh search
./osi-ingest.sh search 'level:ERROR'

# 문서 수 조회
./osi-ingest.sh count

# 인덱스 목록
./osi-ingest.sh indices
```

### osi-periodic-ingest.sh

주기적으로 랜덤 로그 데이터를 OSI 파이프라인에 전송

```bash
./osi-periodic-ingest.sh [interval_seconds] [batch_size]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval_seconds` | 1 | 전송 간격 (초) |
| `batch_size` | 3 | 배치당 문서 수 |

**Examples:**
```bash
# 기본값: 1초 간격, 3건씩
./osi-periodic-ingest.sh

# 2초 간격, 5건씩
./osi-periodic-ingest.sh 2 5

# 백그라운드 실행
nohup ./osi-periodic-ingest.sh 1 3 > periodic.log 2>&1 &
```

**생성되는 로그 필드:**
- `timestamp`: ISO 8601 형식
- `level`: DEBUG, INFO, WARN, ERROR, FATAL
- `source`: api-gateway, auth-service, user-service, payment-service, order-service
- `action`: login, logout, create, update, delete, read, search, export
- `status`: success, failure, pending, timeout, cancelled
- `user_id`: user-1 ~ user-10000
- `request_id`: UUID
- `duration_ms`: 1 ~ 5000
- `ip`: 랜덤 IP 주소

## 요구 사항

- awscurl (`pip install awscurl`)
- AWS credentials 설정 (default profile)

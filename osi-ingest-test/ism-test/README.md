# ISM Test Log Sender

OpenSearch Ingestion 파이프라인으로 로그를 전송하는 Go 바이너리

## 대상 파이프라인

- **Pipeline**: `ism-logs-test`
- **Endpoint**: `https://ism-logs-test-mwo3we5wqvdymi5s37uo77nd3y.us-east-1.osis.amazonaws.com`
- **Path**: `/logs`
- **Region**: `us-east-1`
- **Target Index**: `logs-index-%{yyyy.MM.dd}` (예: `logs-index-2026.01.28`)
- **OpenSearch Domain**: `ltr-vector.awsbuddy.com`

## 요구 사항

- Go 1.21+ (`/usr/local/go/bin/go`)
- AWS credentials (default profile)

## 빌드

```bash
make build
```

## 실행 옵션

```bash
./ism-log-sender [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-endpoint` | (내장) | OSI 엔드포인트 URL |
| `-path` | `/logs` | OSI 경로 |
| `-region` | `us-east-1` | AWS 리전 |
| `-interval` | `1s` | 전송 간격 |
| `-batch` | `3` | 배치당 문서 수 |
| `-count` | `0` | 총 문서 수 (0=무한) |

## Make 타겟

```bash
# 빌드
make build

# 기본 실행 (1초 간격, 3건씩, 무한)
make run

# 커스텀 실행
make send INTERVAL=2s BATCH=5 COUNT=100

# 빠른 테스트 (10건)
make test

# ISM 테스트 (1초 간격, 10건씩, 무한)
make ism-test

# 정리
make clean

# 도움말
make help
```

## 사용 예시

```bash
# 기본 실행
make run

# 2초 간격, 5건씩 전송
make send INTERVAL=2s BATCH=5

# 1000건 전송 후 종료
make send COUNT=1000

# 백그라운드 실행
nohup ./ism-log-sender -interval=1s -batch=10 > sender.log 2>&1 &

# 직접 실행
./ism-log-sender -interval=500ms -batch=20 -count=1000
```

## 생성되는 로그 필드

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 형식 |
| `level` | string | DEBUG, INFO, WARN, ERROR, FATAL |
| `source` | string | 서비스 이름 |
| `action` | string | login, logout, create 등 |
| `status` | string | success, failure, pending 등 |
| `user_id` | string | user-1 ~ user-10000 |
| `request_id` | string | UUID |
| `duration_ms` | int | 1 ~ 5000 |
| `ip` | string | 랜덤 IP |
| `message` | string | 레벨별 메시지 |

## 종료

`Ctrl+C`로 종료하면 전송 총계를 출력합니다.

## 파이프라인 설정

파이프라인 설정 파일: `pipeline-config2.yaml`

```yaml
version: "2"
ism-logs-test:
  source:
    http:
      path: "/logs"
  processor:
    - date:
        from_time_received: true
        destination: "@timestamp"
  sink:
    - opensearch:
        hosts:
          - "https://search-ltr-vector-hnevayhl2dk5ntem55fexzf6ji.us-east-1.es.amazonaws.com"
        index: "logs-index-%{yyyy.MM.dd}"
        aws:
          sts_role_arn: "arn:aws:iam::505725882051:role/OSIUBIPipelineRole"
          region: "us-east-1"
```

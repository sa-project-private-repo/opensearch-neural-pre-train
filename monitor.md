# Server Monitoring Setup Guide

ML/DL 학습 환경을 위한 종합 모니터링 시스템 구성 가이드입니다.

---

## 📊 모니터링 대상 메트릭

### 1. GPU 메트릭
- GPU 사용률 (%)
- GPU 메모리 사용량 (MB/GB)
- GPU 온도 (°C)
- GPU 전력 소비 (W)
- GPU 클럭 속도 (MHz)
- Compute/Memory Utilization
- PCIe 대역폭 사용량

### 2. CPU 메트릭
- CPU 사용률 (per core, average)
- CPU 온도 (°C)
- 프로세스별 CPU 사용량
- Load Average (1/5/15분)

### 3. 메모리 메트릭
- RAM 사용량 (MB/GB)
- RAM 사용률 (%)
- Swap 사용량
- 프로세스별 메모리 사용량

### 4. 디스크 메트릭
- 디스크 사용량 (GB)
- 디스크 I/O (read/write MB/s)
- 디스크 대기 시간 (latency)
- inode 사용량

### 5. 네트워크 메트릭
- 네트워크 대역폭 (in/out MB/s)
- 패킷 전송률
- 연결 상태

### 6. 학습 메트릭
- Training loss
- Validation loss
- Learning rate
- Batch processing time
- Epoch progress

---

## 🎯 모니터링 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Grafana Dashboard                       │
│              (Web UI - Port 3000)                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Prometheus Server                        │
│              (Metrics Storage - Port 9090)                  │
└───┬─────────────┬─────────────┬─────────────┬───────────────┘
    │             │             │             │
    ▼             ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐
│  Node   │ │  DCGM   │ │ Process │ │  Custom Python  │
│ Exporter│ │Exporter │ │ Exporter│ │    Exporter     │
│(Port    │ │(Port    │ │(Port    │ │  (Port 8000)    │
│ 9100)   │ │ 9400)   │ │ 9256)   │ └─────────────────┘
└─────────┘ └─────────┘ └─────────┘
    │             │             │
    ▼             ▼             ▼
System        NVIDIA GPU    Processes
Metrics       Metrics       Metrics
```

---

## 🚀 Phase 1: 기본 모니터링 (명령줄 도구)

### 1.1 NVIDIA GPU 모니터링

#### nvidia-smi (기본)
```bash
# 실시간 모니터링 (1초 간격)
watch -n 1 nvidia-smi

# 상세 정보 출력
nvidia-smi -q

# 특정 메트릭만 출력
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

#### nvtop (실시간 TUI)
```bash
# 설치
sudo apt-get install nvtop  # Ubuntu/Debian
# OR
sudo dnf install nvtop      # Amazon Linux 2023

# 실행
nvtop
```

**nvtop 특징**:
- htop과 유사한 TUI 인터페이스
- 다중 GPU 지원
- 프로세스별 GPU 사용량
- 실시간 그래프

### 1.2 시스템 모니터링

#### htop (CPU/메모리)
```bash
# 설치
sudo apt-get install htop   # Ubuntu/Debian
sudo dnf install htop       # Amazon Linux 2023

# 실행
htop
```

#### sensors (온도)
```bash
# 설치
sudo apt-get install lm-sensors
sudo sensors-detect  # 센서 감지
sudo sensors-detect --auto  # 자동 감지

# 실행
sensors
watch -n 2 sensors
```

#### iotop (디스크 I/O)
```bash
# 설치
sudo apt-get install iotop

# 실행
sudo iotop -o  # I/O 사용 중인 프로세스만 표시
```

#### iftop (네트워크)
```bash
# 설치
sudo apt-get install iftop

# 실행
sudo iftop -i eth0  # 네트워크 인터페이스 지정
```

---

## 🔧 Phase 2: Python 기반 모니터링 스크립트

### 2.1 모니터링 유틸리티 생성

**파일**: `src/monitoring/system_monitor.py`

```python
"""
System monitoring utilities for ML training
"""

import time
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List, Optional
import json


class SystemMonitor:
    """실시간 시스템 메트릭 수집"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.start_time = time.time()

    def get_gpu_metrics(self) -> List[Dict]:
        """GPU 메트릭 수집"""
        gpus = GPUtil.getGPUs()
        metrics = []

        for gpu in gpus:
            metrics.append({
                'gpu_id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,  # %
                'memory_used': gpu.memoryUsed,  # MB
                'memory_total': gpu.memoryTotal,  # MB
                'memory_util': gpu.memoryUtil * 100,  # %
                'temperature': gpu.temperature,  # °C
            })

        return metrics

    def get_cpu_metrics(self) -> Dict:
        """CPU 메트릭 수집"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'load_avg': psutil.getloadavg(),
        }

    def get_memory_metrics(self) -> Dict:
        """메모리 메트릭 수집"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total': mem.total / (1024**3),  # GB
            'available': mem.available / (1024**3),
            'used': mem.used / (1024**3),
            'percent': mem.percent,
            'swap_total': swap.total / (1024**3),
            'swap_used': swap.used / (1024**3),
            'swap_percent': swap.percent,
        }

    def get_disk_metrics(self) -> Dict:
        """디스크 메트릭 수집"""
        disk = psutil.disk_usage('/')
        io = psutil.disk_io_counters()

        return {
            'total': disk.total / (1024**3),  # GB
            'used': disk.used / (1024**3),
            'free': disk.free / (1024**3),
            'percent': disk.percent,
            'read_mb': io.read_bytes / (1024**2) if io else None,
            'write_mb': io.write_bytes / (1024**2) if io else None,
        }

    def get_network_metrics(self) -> Dict:
        """네트워크 메트릭 수집"""
        net = psutil.net_io_counters()

        return {
            'bytes_sent': net.bytes_sent / (1024**2),  # MB
            'bytes_recv': net.bytes_recv / (1024**2),
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
        }

    def get_all_metrics(self) -> Dict:
        """모든 메트릭 수집"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.start_time,
            'gpu': self.get_gpu_metrics(),
            'cpu': self.get_cpu_metrics(),
            'memory': self.get_memory_metrics(),
            'disk': self.get_disk_metrics(),
            'network': self.get_network_metrics(),
        }

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        return metrics

    def print_summary(self):
        """메트릭 요약 출력"""
        metrics = self.get_all_metrics()

        print("="*70)
        print("📊 System Metrics")
        print("="*70)

        # GPU
        if metrics['gpu']:
            print("\n🎮 GPU:")
            for gpu in metrics['gpu']:
                print(f"  [{gpu['gpu_id']}] {gpu['name']}")
                print(f"      Load: {gpu['load']:.1f}%")
                print(f"      Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({gpu['memory_util']:.1f}%)")
                print(f"      Temp: {gpu['temperature']:.1f}°C")

        # CPU
        print(f"\n💻 CPU:")
        print(f"  Usage: {metrics['cpu']['cpu_percent']:.1f}%")
        print(f"  Load Avg: {metrics['cpu']['load_avg']}")

        # Memory
        mem = metrics['memory']
        print(f"\n🧠 Memory:")
        print(f"  Used: {mem['used']:.1f}/{mem['total']:.1f} GB ({mem['percent']:.1f}%)")
        if mem['swap_total'] > 0:
            print(f"  Swap: {mem['swap_used']:.1f}/{mem['swap_total']:.1f} GB ({mem['swap_percent']:.1f}%)")

        # Disk
        disk = metrics['disk']
        print(f"\n💾 Disk:")
        print(f"  Used: {disk['used']:.1f}/{disk['total']:.1f} GB ({disk['percent']:.1f}%)")

        print("="*70)


def monitor_training(interval: int = 5, log_file: str = "training_metrics.jsonl"):
    """
    학습 중 실시간 모니터링

    Usage:
        In notebook or script:
        from src.monitoring.system_monitor import monitor_training
        monitor_training(interval=5)
    """
    monitor = SystemMonitor(log_file=log_file)

    print(f"🔍 Monitoring started (interval: {interval}s)")
    print(f"📝 Logging to: {log_file}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            monitor.print_summary()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


if __name__ == "__main__":
    # Standalone monitoring
    monitor_training(interval=5)
```

### 2.2 필요한 패키지 설치

```bash
# requirements-monitoring.txt
pip install psutil gputil py3nvml
```

### 2.3 Jupyter Notebook에서 사용

```python
# 노트북 셀에 추가
from src.monitoring.system_monitor import SystemMonitor

# 학습 전 메트릭 확인
monitor = SystemMonitor()
monitor.print_summary()

# 학습 시작...

# 학습 중간에 메트릭 확인
monitor.print_summary()
```

---

## 📈 Phase 3: Prometheus + Grafana (프로덕션)

### 3.1 아키텍처 개요

**장점**:
- 장기간 메트릭 저장
- 웹 기반 대시보드
- 알림 설정 가능
- 다중 서버 모니터링

### 3.2 설치 및 구성

#### Step 1: Prometheus 설치

```bash
# Prometheus 다운로드 (최신 버전)
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
sudo mv prometheus-2.48.0.linux-amd64 /opt/prometheus

# Systemd 서비스 생성
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/opt/prometheus/prometheus \\
  --config.file=/opt/prometheus/prometheus.yml \\
  --storage.tsdb.path=/var/lib/prometheus/ \\
  --web.console.templates=/opt/prometheus/consoles \\
  --web.console.libraries=/opt/prometheus/console_libraries

[Install]
WantedBy=multi-user.target
EOF

# 사용자 및 디렉토리 생성
sudo useradd --no-create-home --shell /bin/false prometheus
sudo mkdir -p /var/lib/prometheus
sudo chown -R prometheus:prometheus /var/lib/prometheus /opt/prometheus

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl start prometheus
sudo systemctl enable prometheus

# 상태 확인
sudo systemctl status prometheus
```

#### Step 2: Node Exporter 설치 (시스템 메트릭)

```bash
# Node Exporter 다운로드
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
tar xvfz node_exporter-*.tar.gz
sudo mv node_exporter-1.7.0.linux-amd64/node_exporter /usr/local/bin/

# Systemd 서비스 생성
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

# 사용자 생성
sudo useradd --no-create-home --shell /bin/false node_exporter

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl start node_exporter
sudo systemctl enable node_exporter

# 메트릭 확인
curl http://localhost:9100/metrics
```

#### Step 3: DCGM Exporter 설치 (NVIDIA GPU 메트릭)

```bash
# NVIDIA DCGM 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager

# DCGM Exporter Docker 실행
docker run -d --gpus all --rm \
  -p 9400:9400 \
  --name dcgm-exporter \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04

# 메트릭 확인
curl http://localhost:9400/metrics
```

**Docker 없이 설치** (선택):
```bash
# Go 설치 후 소스에서 빌드
git clone https://github.com/NVIDIA/dcgm-exporter.git
cd dcgm-exporter
make binary
sudo cp dcgm-exporter /usr/local/bin/

# Systemd 서비스 생성 (위와 유사)
```

#### Step 4: Prometheus 설정

**파일**: `/opt/prometheus/prometheus.yml`

```yaml
# Prometheus configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: []

# Rule files
rule_files:
  - "rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus 자체 모니터링
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter (시스템 메트릭)
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
        labels:
          instance: 'ml-training-server'

  # DCGM Exporter (GPU 메트릭)
  - job_name: 'gpu'
    static_configs:
      - targets: ['localhost:9400']
        labels:
          instance: 'ml-training-server'

  # Process Exporter (프로세스 메트릭)
  - job_name: 'process'
    static_configs:
      - targets: ['localhost:9256']
        labels:
          instance: 'ml-training-server'

  # Custom Python Exporter (학습 메트릭)
  - job_name: 'training'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          instance: 'ml-training-server'
```

**설정 리로드**:
```bash
sudo systemctl reload prometheus
# OR
curl -X POST http://localhost:9090/-/reload
```

#### Step 5: Grafana 설치

```bash
# Grafana 저장소 추가
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

# 설치
sudo apt-get update
sudo apt-get install grafana

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# 상태 확인
sudo systemctl status grafana-server
```

**접속**:
- URL: http://localhost:3000
- 기본 ID/PW: admin/admin

#### Step 6: Grafana 데이터 소스 설정

1. Grafana 웹 UI 접속
2. Configuration > Data Sources
3. Add data source > Prometheus
4. URL: `http://localhost:9090`
5. Save & Test

### 3.3 Custom Python Exporter (학습 메트릭)

**파일**: `src/monitoring/training_exporter.py`

```python
"""
Prometheus exporter for ML training metrics
"""

from prometheus_client import start_http_server, Gauge, Counter
import time
import json
from pathlib import Path


class TrainingMetricsExporter:
    """학습 메트릭을 Prometheus로 export"""

    def __init__(self, port: int = 8000):
        self.port = port

        # Gauge 메트릭 정의
        self.train_loss = Gauge('training_loss', 'Training loss')
        self.val_loss = Gauge('validation_loss', 'Validation loss')
        self.learning_rate = Gauge('learning_rate', 'Current learning rate')
        self.epoch = Gauge('current_epoch', 'Current epoch number')
        self.batch_time = Gauge('batch_processing_time', 'Time per batch (seconds)')

        # Counter 메트릭
        self.batches_processed = Counter('batches_processed_total', 'Total batches processed')

    def update_metrics(self, metrics: dict):
        """메트릭 업데이트"""
        if 'train_loss' in metrics:
            self.train_loss.set(metrics['train_loss'])
        if 'val_loss' in metrics:
            self.val_loss.set(metrics['val_loss'])
        if 'learning_rate' in metrics:
            self.learning_rate.set(metrics['learning_rate'])
        if 'epoch' in metrics:
            self.epoch.set(metrics['epoch'])
        if 'batch_time' in metrics:
            self.batch_time.set(metrics['batch_time'])

    def start(self):
        """Exporter 시작"""
        start_http_server(self.port)
        print(f"✅ Training metrics exporter started on port {self.port}")
        print(f"   Metrics URL: http://localhost:{self.port}/metrics")


# Usage in training script
if __name__ == "__main__":
    exporter = TrainingMetricsExporter(port=8000)
    exporter.start()

    # 학습 루프에서 메트릭 업데이트
    while True:
        # 예시 메트릭
        exporter.update_metrics({
            'train_loss': 0.5,
            'val_loss': 0.6,
            'learning_rate': 0.001,
            'epoch': 1,
            'batch_time': 0.5,
        })
        time.sleep(10)
```

**requirements-monitoring.txt에 추가**:
```
prometheus-client==0.19.0
```

### 3.4 Grafana 대시보드 구성

#### GPU 대시보드 템플릿

```json
{
  "dashboard": {
    "title": "ML Training - GPU Monitoring",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Temperature",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_TEMP"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

**대시보드 Import**:
1. Grafana UI > Dashboards > Import
2. 위 JSON 붙여넣기 또는
3. 커뮤니티 대시보드 사용:
   - NVIDIA DCGM Exporter: Dashboard ID `12239`
   - Node Exporter Full: Dashboard ID `1860`

---

## 🔔 Phase 4: 알림 설정 (Alertmanager)

### 4.1 Alertmanager 설치

```bash
# Alertmanager 다운로드
cd /tmp
wget https://github.com/prometheus/alertmanager/releases/download/v0.26.0/alertmanager-0.26.0.linux-amd64.tar.gz
tar xvfz alertmanager-*.tar.gz
sudo mv alertmanager-0.26.0.linux-amd64 /opt/alertmanager

# 설정 파일 생성
sudo tee /opt/alertmanager/alertmanager.yml > /dev/null <<EOF
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email-notifications'

receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'your-email@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
EOF

# Systemd 서비스 생성 및 시작
# (Prometheus와 유사한 방식)
```

### 4.2 알림 규칙 설정

**파일**: `/opt/prometheus/rules/ml_training_alerts.yml`

```yaml
groups:
  - name: ml_training_alerts
    interval: 30s
    rules:
      # GPU 온도 알림
      - alert: HighGPUTemperature
        expr: DCGM_FI_DEV_GPU_TEMP > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature is high"
          description: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C"

      # GPU 메모리 부족 알림
      - alert: HighGPUMemoryUsage
        expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU {{ $labels.gpu }} memory usage is {{ $value }}%"

      # 시스템 메모리 부족 알림
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "System memory usage is high"
          description: "Memory usage is {{ $value }}%"

      # 디스크 공간 부족 알림
      - alert: LowDiskSpace
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is {{ $value }}% remaining"

      # 학습 손실 증가 알림
      - alert: TrainingLossIncreasing
        expr: rate(training_loss[5m]) > 0
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Training loss is increasing"
          description: "Training loss has been increasing for 10 minutes"
```

---

## 📱 Phase 5: 간단한 웹 대시보드 (Flask)

간단한 커스텀 대시보드가 필요한 경우:

**파일**: `src/monitoring/web_dashboard.py`

```python
"""
Simple web dashboard for training monitoring
"""

from flask import Flask, render_template, jsonify
from src.monitoring.system_monitor import SystemMonitor
import threading
import time

app = Flask(__name__)
monitor = SystemMonitor()

# 최근 메트릭 저장
latest_metrics = {}

def update_metrics():
    """백그라운드에서 메트릭 업데이트"""
    global latest_metrics
    while True:
        latest_metrics = monitor.get_all_metrics()
        time.sleep(5)

# 백그라운드 스레드 시작
thread = threading.Thread(target=update_metrics, daemon=True)
thread.start()

@app.route('/')
def index():
    """메인 대시보드"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """메트릭 API"""
    return jsonify(latest_metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**실행**:
```bash
python src/monitoring/web_dashboard.py
# 접속: http://localhost:5000
```

---

## 📊 사용 시나리오별 권장사항

### 시나리오 1: 개발/실험 단계
**권장**: Phase 1 + Phase 2
- nvidia-smi, nvtop으로 실시간 확인
- Python 스크립트로 메트릭 로깅
- 간단하고 빠르게 시작 가능

### 시나리오 2: 장기 학습
**권장**: Phase 2 + Phase 3
- Python 스크립트로 메트릭 수집
- Prometheus + Grafana로 시각화
- 장기간 메트릭 보관 및 분석

### 시나리오 3: 프로덕션 환경
**권장**: Phase 3 + Phase 4
- 완전한 Prometheus + Grafana 스택
- Alertmanager로 알림 설정
- 다중 서버 모니터링

---

## 🎯 빠른 시작 (Quick Start)

### Option A: 명령줄 도구만 사용

```bash
# 터미널 1: GPU 모니터링
watch -n 1 nvidia-smi

# 터미널 2: 시스템 모니터링
htop

# 터미널 3: 온도 모니터링
watch -n 2 sensors
```

### Option B: Python 스크립트 사용

```bash
# 1. 패키지 설치
pip install psutil gputil py3nvml

# 2. 모니터링 스크립트 실행
python -c "from src.monitoring.system_monitor import monitor_training; monitor_training(interval=5)"
```

### Option C: Prometheus + Grafana (완전한 솔루션)

```bash
# 설치 스크립트 실행
./scripts/setup_monitoring.sh

# 서비스 시작
sudo systemctl start prometheus
sudo systemctl start node_exporter
sudo systemctl start grafana-server

# Grafana 접속
open http://localhost:3000
```

---

## 📝 체크리스트

### 설치 체크리스트

- [ ] nvidia-smi 작동 확인
- [ ] nvtop 설치 및 실행
- [ ] sensors 설정 및 온도 확인
- [ ] psutil, gputil 설치
- [ ] Prometheus 설치 및 실행
- [ ] Node Exporter 설치
- [ ] DCGM Exporter 설치 (GPU)
- [ ] Grafana 설치 및 접속
- [ ] 데이터 소스 연결
- [ ] 대시보드 import
- [ ] 알림 규칙 설정
- [ ] 알림 테스트

### 모니터링 체크리스트

학습 시작 전:
- [ ] GPU 사용 가능 여부 확인
- [ ] GPU 메모리 충분 확인
- [ ] 디스크 공간 확인 (10GB 이상)
- [ ] 시스템 메모리 확인
- [ ] 온도 정상 범위 확인

학습 중:
- [ ] GPU 사용률 80% 이상 유지
- [ ] GPU 온도 85°C 이하 유지
- [ ] 메모리 leak 없음
- [ ] 디스크 I/O 정상
- [ ] Loss 정상적으로 감소

---

## 🔧 문제 해결

### nvidia-smi 작동 안 함
```bash
# 드라이버 재설치
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-driver-535

# 재부팅
sudo reboot
```

### sensors 온도 표시 안 됨
```bash
# 센서 재감지
sudo sensors-detect --auto
sudo systemctl restart lm-sensors
```

### Prometheus 메트릭 수집 안 됨
```bash
# Exporter 상태 확인
systemctl status node_exporter
systemctl status dcgm-exporter

# 포트 확인
netstat -tlnp | grep 9100
netstat -tlnp | grep 9400

# 방화벽 확인
sudo ufw allow 9100
sudo ufw allow 9400
```

---

## 📚 참고 자료

### 공식 문서
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Node Exporter](https://github.com/prometheus/node_exporter)

### 커뮤니티 대시보드
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- NVIDIA GPU: ID `12239`
- Node Exporter: ID `1860`

### 유용한 도구
- [gpustat](https://github.com/wookayin/gpustat) - nvidia-smi 대체
- [glances](https://github.com/nicolargo/glances) - 통합 모니터링
- [netdata](https://www.netdata.cloud/) - 실시간 모니터링

#!/bin/bash
# install_monitoring.sh
# 모니터링 도구 설치 스크립트

set -e

echo "======================================================================"
echo "🔧 Installing Monitoring Tools"
echo "======================================================================"

# Python 패키지 설치
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements-monitoring.txt

# 시스템 패키지 확인
echo ""
echo "🔍 Checking system packages..."

# OS 감지
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "⚠️  Cannot detect OS"
    OS="unknown"
fi

# htop 설치
if ! command -v htop &> /dev/null; then
    echo "📦 Installing htop..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y htop
    elif [ "$OS" = "amzn" ]; then
        sudo dnf install -y htop
    fi
else
    echo "✅ htop already installed"
fi

# nvtop 설치 (선택)
if ! command -v nvtop &> /dev/null; then
    echo "📦 Installing nvtop (GPU monitoring)..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt-get install -y nvtop
    elif [ "$OS" = "amzn" ]; then
        echo "⚠️  nvtop not available in Amazon Linux repos"
        echo "   You can build from source: https://github.com/Syllo/nvtop"
    fi
else
    echo "✅ nvtop already installed"
fi

# lm-sensors 설치
if ! command -v sensors &> /dev/null; then
    echo "📦 Installing lm-sensors (temperature monitoring)..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt-get install -y lm-sensors
        echo "Running sensors-detect..."
        sudo sensors-detect --auto
    elif [ "$OS" = "amzn" ]; then
        sudo dnf install -y lm_sensors
        sudo sensors-detect --auto
    fi
else
    echo "✅ lm-sensors already installed"
fi

# 디렉토리 생성
echo ""
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p src/monitoring

echo ""
echo "======================================================================"
echo "✅ Installation complete!"
echo "======================================================================"
echo ""
echo "Quick start:"
echo "  1. Test monitoring: python -m src.monitoring.system_monitor"
echo "  2. In notebook: from src.monitoring import monitor_training"
echo "  3. View logs: tail -f logs/training_metrics.jsonl"
echo ""
echo "For advanced setup (Prometheus + Grafana), see monitor.md"
echo ""

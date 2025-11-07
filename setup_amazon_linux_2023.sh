#!/bin/bash
#
# OpenSearch Korean Neural Sparse Model - Amazon Linux 2023 설치 스크립트
#
# 이 스크립트는 Amazon Linux 2023 환경에서 필요한 모든 의존성을 설치합니다.
#

set -e  # 에러 발생 시 중단

echo "============================================================"
echo "OpenSearch Korean Neural Sparse - Amazon Linux 2023 설치"
echo "============================================================"
echo ""

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트 중..."
sudo dnf update -y

# 2. Python 및 개발 도구 설치
echo ""
echo "🐍 Python 3.12 및 개발 도구 설치 중..."
sudo dnf install -y python3.12 python3.12-pip python3.12-devel
sudo dnf install -y gcc gcc-c++ make git

# Python 3.12를 기본으로 설정
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.12 1

python3 --version
pip3 --version

# 3. Java 설치 (Mecab 빌드용)
echo ""
echo "☕ OpenJDK 설치 중..."
sudo dnf install -y java-17-amazon-corretto-devel

# 4. 한국어 처리를 위한 추가 패키지
echo ""
echo "🇰🇷 한국어 처리 패키지 설치 중..."
sudo dnf install -y automake libtool

# 5. Python 가상 환경 생성 (권장)
echo ""
echo "🔧 Python 가상 환경 생성 중..."
python3 -m venv ~/opensearch-neural-env
source ~/opensearch-neural-env/bin/activate

echo "가상 환경 활성화됨: ~/opensearch-neural-env"

# 6. Python 패키지 업그레이드
echo ""
echo "📦 pip 업그레이드 중..."
pip3 install --upgrade pip setuptools wheel

# 7. GPU 확인 및 PyTorch 설치
echo ""
echo "🖥️  GPU 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU 감지됨!"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "🔥 PyTorch 설치 중 (GPU 버전 - CUDA 12.1 for Tesla T4)..."
    pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️  GPU 미감지 - CPU 버전 설치"
    echo "🔥 PyTorch 설치 중 (CPU 버전)..."
    pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# 8. requirements.txt에서 나머지 패키지 설치
echo ""
echo "📦 requirements.txt에서 패키지 설치 중..."
if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
    echo "✓ requirements.txt 패키지 설치 완료"
else
    echo "⚠️  requirements.txt 파일이 없습니다. 개별 패키지 설치 진행..."
    echo ""
    echo "🤗 Transformers 및 관련 라이브러리 설치 중..."
    pip3 install transformers==4.46.3
    pip3 install datasets==3.1.0
    pip3 install accelerate==1.1.1
    pip3 install huggingface-hub==0.26.2

    echo ""
    echo "📊 데이터 과학 라이브러리 설치 중..."
    pip3 install numpy==2.1.3
    pip3 install pandas==2.2.3
    pip3 install scikit-learn==1.5.2
    pip3 install matplotlib==3.9.2
    pip3 install seaborn==0.13.2
    pip3 install tqdm==4.66.6
fi

# 10. Mecab 설치 (한국어 형태소 분석기)
echo ""
echo "🔤 Mecab 한국어 형태소 분석기 설치 중..."

# Mecab 엔진 설치
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar -zxvf mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
sudo make install
sudo ldconfig

# Mecab 한국어 사전 설치
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./autogen.sh
./configure
make
sudo make install

# mecab-python3 설치
pip3 install mecab-python3==1.0.9

# 11. KoNLPy 설치
echo ""
echo "🇰🇷 KoNLPy 설치 중..."
pip3 install konlpy==0.6.0

# 12. Jupyter 설치 (선택사항)
echo ""
echo "📓 Jupyter Notebook 설치 중..."
pip3 install jupyter==1.1.1
pip3 install ipywidgets==8.1.5
pip3 install notebook==7.2.2

# 13. 설치 확인
echo ""
echo "============================================================"
echo "✅ 설치 완료! 확인 중..."
echo "============================================================"
echo ""

echo "Python 버전:"
python3 --version

echo ""
echo "설치된 패키지:"
pip3 list | grep -E "torch|transformers|konlpy|mecab"

echo ""
echo "Mecab 테스트:"
python3 -c "from konlpy.tag import Mecab; m = Mecab(); print(m.morphs('한국어 형태소 분석 테스트'))" || echo "⚠️  Mecab 설정 필요"

echo ""
echo "============================================================"
echo "🎉 모든 설치가 완료되었습니다!"
echo "============================================================"
echo ""
echo "다음 명령어로 가상 환경을 활성화하세요:"
echo "  source ~/opensearch-neural-env/bin/activate"
echo ""
echo "테스트 실행:"
echo "  python3 demo_idf_korean.py"
echo ""
echo "전체 학습 실행:"
echo "  python3 test_korean_neural_sparse.py"
echo ""
echo "Jupyter 노트북 실행:"
echo "  jupyter notebook korean_neural_sparse_training.ipynb"
echo ""

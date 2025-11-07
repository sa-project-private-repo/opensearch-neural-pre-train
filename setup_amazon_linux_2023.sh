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
echo "🐍 Python 3.11 및 개발 도구 설치 중..."
sudo dnf install -y python3.11 python3.11-pip python3.11-devel
sudo dnf install -y gcc gcc-c++ make git

# Python 3.11을 기본으로 설정
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

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

# 7. PyTorch 설치 (CPU 버전)
echo ""
echo "🔥 PyTorch 설치 중 (CPU 버전)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU가 있는 경우 (선택사항):
# pip3 install torch torchvision torchaudio

# 8. Transformers 및 관련 라이브러리
echo ""
echo "🤗 Transformers 및 관련 라이브러리 설치 중..."
pip3 install transformers==4.35.0
pip3 install datasets==2.14.0
pip3 install accelerate==0.24.0
pip3 install huggingface-hub==0.19.0

# 9. 데이터 과학 라이브러리
echo ""
echo "📊 데이터 과학 라이브러리 설치 중..."
pip3 install numpy==1.24.3
pip3 install pandas==2.0.3
pip3 install scikit-learn==1.3.0
pip3 install matplotlib==3.7.2
pip3 install seaborn==0.12.2
pip3 install tqdm==4.66.1

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
pip3 install mecab-python3==1.0.6

# 11. KoNLPy 설치
echo ""
echo "🇰🇷 KoNLPy 설치 중..."
pip3 install konlpy==0.6.0

# 12. Jupyter 설치 (선택사항)
echo ""
echo "📓 Jupyter Notebook 설치 중..."
pip3 install jupyter==1.0.0
pip3 install ipywidgets==8.1.0

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

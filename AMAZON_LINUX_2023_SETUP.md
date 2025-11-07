# OpenSearch Korean Neural Sparse Model - Amazon Linux 2023 설치 가이드

이 가이드는 Amazon Linux 2023 환경에서 OpenSearch 한국어 neural sparse 모델을 설치하고 실행하는 방법을 설명합니다.

## 📋 시스템 요구사항

- **OS**: Amazon Linux 2023
- **Python**: 3.11+
- **RAM**: 최소 8GB (학습 시 16GB 권장)
- **저장 공간**: 10GB 이상
- **CPU**: 멀티코어 권장 (학습 속도 향상)
- **GPU**: 선택사항 (있으면 학습 속도 대폭 향상)

## 🚀 빠른 시작

### 방법 1: 자동 설치 스크립트 (권장)

```bash
# 저장소 클론
git clone <repository-url>
cd opensearch-neural-pre-train

# 실행 권한 부여
chmod +x setup_amazon_linux_2023.sh

# 설치 실행
./setup_amazon_linux_2023.sh

# 가상 환경 활성화
source ~/opensearch-neural-env/bin/activate

# 테스트 실행
python3 demo_idf_korean.py
```

### 방법 2: 수동 설치

#### 1. 시스템 패키지 업데이트

```bash
sudo dnf update -y
```

#### 2. Python 3.11 설치

```bash
# Python 3.11 및 개발 도구
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# 개발 도구
sudo dnf install -y gcc gcc-c++ make git

# Python 3.11을 기본으로 설정
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# 확인
python3 --version  # Python 3.11.x
```

#### 3. Java 설치 (Mecab 빌드용)

```bash
sudo dnf install -y java-17-amazon-corretto-devel
java -version
```

#### 4. Python 가상 환경 생성

```bash
python3 -m venv ~/opensearch-neural-env
source ~/opensearch-neural-env/bin/activate
pip3 install --upgrade pip setuptools wheel
```

#### 5. PyTorch 설치

**CPU 버전** (권장 - 시작용):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU 버전** (CUDA 지원 시):
```bash
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 6. 필수 Python 패키지 설치

```bash
# Transformers 및 Hugging Face 라이브러리
pip3 install transformers==4.35.0
pip3 install datasets==2.14.0
pip3 install accelerate==0.24.0
pip3 install huggingface-hub==0.19.0

# 데이터 과학 라이브러리
pip3 install numpy==1.24.3
pip3 install pandas==2.0.3
pip3 install scikit-learn==1.3.0
pip3 install matplotlib==3.7.2
pip3 install seaborn==0.12.2
pip3 install tqdm==4.66.1
```

#### 7. Mecab 한국어 형태소 분석기 설치

```bash
# 필수 패키지
sudo dnf install -y automake libtool

# Mecab 엔진
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar -zxvf mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
sudo make install
sudo ldconfig

# Mecab 한국어 사전
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./autogen.sh
./configure
make
sudo make install

# Python 바인딩
pip3 install mecab-python3==1.0.6
pip3 install konlpy==0.6.0
```

#### 8. Jupyter Notebook 설치 (선택사항)

```bash
pip3 install jupyter==1.0.0
pip3 install ipywidgets==8.1.0
```

## ✅ 설치 확인

```bash
# Python 및 패키지 확인
python3 --version
pip3 list | grep -E "torch|transformers|konlpy"

# PyTorch 확인
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# CUDA 확인 (GPU 버전 설치 시)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Mecab 확인
python3 -c "from konlpy.tag import Mecab; m = Mecab(); print(m.morphs('한국어 테스트'))"
```

## 🧪 테스트 실행

### 1. 간단한 데모 (의존성 최소)

```bash
python3 demo_idf_korean.py
```

**예상 출력**:
```
============================================================
OpenSearch Inference-Free Neural Sparse - IDF 데모
============================================================

📚 샘플 데이터:
  문서: 15개
  쿼리: 8개

✓ 96개 토큰의 IDF 계산 완료
...
```

### 2. 전체 모델 학습 테스트

```bash
python3 test_korean_neural_sparse.py
```

**예상 소요 시간**:
- CPU: 30-60분
- GPU: 5-10분

### 3. Jupyter 노트북

```bash
# Jupyter 서버 시작
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 또는 로컬에서
jupyter notebook korean_neural_sparse_training.ipynb
```

## 📁 프로젝트 구조

```
opensearch-neural-pre-train/
├── setup_amazon_linux_2023.sh          # Amazon Linux 2023 설치 스크립트
├── AMAZON_LINUX_2023_SETUP.md          # 이 가이드
├── korean_neural_sparse_training.ipynb # 전체 학습 노트북
├── test_korean_neural_sparse.py        # 전체 테스트 스크립트
├── demo_idf_korean.py                  # 간단한 데모
└── demo_idf.json                       # 생성된 IDF 샘플
```

## 🔧 문제 해결

### 1. Mecab 설치 오류

**증상**: `from konlpy.tag import Mecab` 실패

**해결**:
```bash
# 라이브러리 경로 확인
sudo ldconfig

# 환경 변수 설정
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 재설치
pip3 uninstall mecab-python3
pip3 install mecab-python3
```

### 2. PyTorch 메모리 부족

**증상**: `CUDA out of memory` 또는 시스템 메모리 부족

**해결**:
```python
# 배치 사이즈 줄이기
BATCH_SIZE = 8  # 기본 16에서 감소

# 데이터 샘플링
train_data = train_data[:5000]  # 데이터 수 제한
```

### 3. Transformers 다운로드 느림

**해결**:
```bash
# Hugging Face 미러 사용 (중국)
export HF_ENDPOINT=https://hf-mirror.com

# 또는 캐시 디렉토리 변경
export TRANSFORMERS_CACHE=/path/to/large/disk
```

### 4. 패키지 버전 충돌

**해결**:
```bash
# 가상 환경 재생성
deactivate
rm -rf ~/opensearch-neural-env
python3 -m venv ~/opensearch-neural-env
source ~/opensearch-neural-env/bin/activate

# 재설치
./setup_amazon_linux_2023.sh
```

## 🚀 EC2 인스턴스 권장 사항

### 개발/테스트용

**인스턴스 타입**: `t3.xlarge` 또는 `t3.2xlarge`
- vCPU: 4-8
- 메모리: 16-32GB
- 스토리지: 50GB EBS (gp3)
- 비용: $0.16-0.33/시간 (us-east-1)

```bash
# EC2 인스턴스 생성 (AWS CLI)
aws ec2 run-instances \
  --image-id ami-0dfcb1ef8550277af \  # Amazon Linux 2023
  --instance-type t3.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]'
```

### 프로덕션 학습용 (GPU)

**인스턴스 타입**: `g4dn.xlarge` 또는 `g5.xlarge`
- GPU: NVIDIA T4 또는 A10G
- vCPU: 4-4
- 메모리: 16-16GB
- GPU 메모리: 16GB
- 비용: $0.53-1.01/시간

```bash
# Deep Learning AMI 사용 (권장)
aws ec2 run-instances \
  --image-id ami-0c9424a408e18bcc9 \  # Deep Learning AMI
  --instance-type g4dn.xlarge \
  --key-name your-key
```

## 📊 성능 벤치마크 (Amazon Linux 2023)

| 인스턴스 타입 | 학습 시간 (10 epochs) | 추론 속도 (쿼리) | 비용 (USD/시간) |
|--------------|---------------------|----------------|----------------|
| t3.xlarge (CPU) | ~45분 | ~50ms | $0.16 |
| t3.2xlarge (CPU) | ~25분 | ~30ms | $0.33 |
| g4dn.xlarge (GPU) | ~8분 | ~5ms | $0.53 |
| g5.xlarge (GPU) | ~5분 | ~3ms | $1.01 |

## 🔒 보안 고려사항

### 1. IAM 역할 설정

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-model-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "opensearch:*"
      ],
      "Resource": "arn:aws:es:region:account:domain/your-domain/*"
    }
  ]
}
```

### 2. 보안 그룹

```bash
# SSH (개발 중에만)
22/tcp from your-ip

# Jupyter Notebook (선택사항)
8888/tcp from your-ip

# OpenSearch (프로덕션)
443/tcp from vpc-cidr
```

### 3. 모델 저장

```bash
# S3에 모델 업로드
aws s3 cp ./opensearch-korean-neural-sparse-v1/ \
  s3://your-bucket/models/ --recursive

# S3에서 다운로드
aws s3 sync s3://your-bucket/models/opensearch-korean-neural-sparse-v1/ \
  ./model/
```

## 📚 다음 단계

1. **모델 학습**: `python3 test_korean_neural_sparse.py`
2. **모델 평가**: BEIR 벤치마크 실행
3. **OpenSearch 통합**: 모델을 OpenSearch에 업로드
4. **프로덕션 배포**: Docker 컨테이너로 패키징
5. **모니터링**: CloudWatch로 성능 추적

## 🤝 지원

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **AWS Support**: EC2, OpenSearch 관련 문제
- **Hugging Face Forum**: 모델 학습 관련 질문

## 📖 참고 자료

- [Amazon Linux 2023 User Guide](https://docs.aws.amazon.com/linux/al2023/)
- [OpenSearch Neural Sparse Documentation](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/)

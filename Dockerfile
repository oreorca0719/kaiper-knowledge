# 1. 베이스 이미지 설정
FROM python:3.11-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 환경 설정
ENV PYTHONUNBUFFERED=1

# 4. OS-level 의존성 (sentence-transformers/torch 빌드 의존)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Python 의존성 설치
COPY requirements.txt .

# torch 는 CPU 전용 인덱스에서 먼저 설치한다.
# PyPI 의 기본 torch wheel 은 nvidia-cudnn / nvidia-cublas / triton 등 CUDA
# 런타임을 의존성으로 끌고 온다 (aarch64 도 동일). 이 컨테이너는 CPU 전용
# 이라 단 한 번도 로드되지 않으면서 이미지만 ~6GB 부풀린다.
#   측정: CUDA 포함 8.32GB -> CPU 전용 약 2.5GB
RUN pip install --no-cache-dir     --index-url https://download.pytorch.org/whl/cpu     torch==2.10.0

RUN pip install --no-cache-dir -r requirements.txt

# 6. Cross-encoder reranker 모델 사전 다운로드 (이미지에 포함 → 컨테이너 시작 시 빠른 cold start)
#    ~2.3GB. CPU-only 추론이므로 GPU 의존성 없음.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)"

# 7. 프로젝트 코드 복사
COPY . .

# 8. 포트 노출
EXPOSE 8080

# 9. 실행 명령
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

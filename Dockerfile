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

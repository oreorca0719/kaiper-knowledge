#!/usr/bin/env bash
# 배포 스크립트 — 반드시 이 경로로만 배포한다.
#
# 【이 스크립트가 존재하는 이유】
# 문법 오류가 있는 builder.py 를 커밋·배포한 사고가 있었다. 개별 파일 컴파일
# 확인이 실패했는데도 뒤이은 커밋·배포 명령이 그대로 실행됐다 (셸 체이닝이
# `;` 였고, 실패를 막는 게이트가 없었다).
#
# 여기서는 set -e 로 어느 단계든 실패하면 즉시 중단하고, EC2 쪽에서도
# 압축 해제 직후 컴파일을 한 번 더 검사한 뒤에만 컨테이너를 재시작한다.
# 로컬 검사를 통과하지 못한 코드는 서버에 반영되지 않는다.
set -euo pipefail

REGION=ap-northeast-1
BUCKET=langgraph-rag-333347414948-ap-northeast-1
INSTANCE=i-0d76ba6d090ba5069
cd "$(dirname "$0")/.."

echo "[1/4] 전체 Python 컴파일 검사"
python -m compileall -q app/ >/dev/null
echo "      OK"

echo "[2/4] 아카이브 생성 (HEAD 기준 — 커밋되지 않은 변경은 배포되지 않는다)"
git archive --format=tar.gz -o /tmp/app.tar.gz HEAD

echo "[3/4] S3 업로드"
aws s3 cp /tmp/app.tar.gz "s3://$BUCKET/deploy/app.tar.gz" --region "$REGION" >/dev/null

echo "[4/4] EC2 반영 + 헬스체크"
aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE" \
  --document-name AWS-RunShellScript --timeout-seconds 900 \
  --parameters "commands=[
    \"cd /opt/langgraph/app\",
    \"aws s3 cp s3://$BUCKET/deploy/app.tar.gz . --region $REGION >/dev/null\",
    \"tar xzf app.tar.gz && rm app.tar.gz\",
    \"docker exec kaiper python -m compileall -q /app/app || exit 1\",
    \"docker restart kaiper >/dev/null\",
    \"sleep 40\",
    \"curl -sf -m 10 localhost:8080/health || exit 1\"
  ]" --output text --query Command.CommandId

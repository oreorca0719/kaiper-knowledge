from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from app.auth.dynamo import _region


def _log_table_name() -> str:
    return (os.getenv("ROUTING_LOG_TABLE") or "langgraph_routing_logs").strip()


def _log_table():
    return boto3.resource("dynamodb", region_name=_region()).Table(_log_table_name())


def ensure_routing_log_table() -> None:
    """CREATE_ROUTING_LOG_TABLE=1 일 때 테이블을 생성합니다."""
    create_enabled = (os.getenv("CREATE_ROUTING_LOG_TABLE") or "").strip() in ("1", "true", "True", "YES", "yes")
    if not create_enabled:
        return

    client = boto3.client("dynamodb", region_name=_region())
    name = _log_table_name()

    try:
        client.describe_table(TableName=name)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

    client.create_table(
        TableName=name,
        AttributeDefinitions=[{"AttributeName": "log_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "log_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=name)
    print(f"[ROUTING_LOG] Table created: {name}")


def scan_user_recent_logs(user_id: str, limit: int = 5) -> list:
    """특정 사용자의 최근 라우팅 로그를 반환합니다."""
    try:
        from boto3.dynamodb.conditions import Attr
        tbl = _log_table()
        resp = tbl.scan(
            FilterExpression=Attr("user_id").eq(user_id),
        )
        items = []
        for raw in resp.get("Items", []):
            items.append({
                "timestamp": float(raw.get("timestamp", 0)),
                "input_text": raw.get("input_text", ""),
                "final_task": raw.get("final_task", ""),
            })
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items[:limit]
    except Exception as e:
        print(f"[ROUTING_LOG] scan_user failed: {e}")
        return []


def save_routing_log(
    user_id: str,
    input_text: str,
    final_task: str,
    routing_debug: Dict[str, Any],
) -> None:
    """라우팅 결정 1건을 DynamoDB에 기록합니다. 실패해도 요청 흐름에 영향 없음."""
    try:
        item: Dict[str, Any] = {
            "log_id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "user_id": user_id or "unknown",
            "input_text": (input_text or "")[:2000],
            "final_task": final_task or "unknown",
            # float 직렬화 문제 방지: JSON 문자열로 저장
            "routing_debug": json.dumps(routing_debug, ensure_ascii=False),
            # 빠른 분석을 위한 핵심 지표 평탄화
            "mode": routing_debug.get("mode", ""),
            "semantic_decision": routing_debug.get("decision", ""),
            "top1_score": str(routing_debug.get("top1_score", "")),
            "margin": str(routing_debug.get("margin", "")),
        }
        _log_table().put_item(Item=item)
    except Exception as e:
        print(f"[ROUTING_LOG] save failed (non-fatal): {e}")
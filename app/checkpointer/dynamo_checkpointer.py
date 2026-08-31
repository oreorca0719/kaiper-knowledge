from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Iterator, Optional, Sequence, Tuple

import boto3
from botocore.exceptions import ClientError

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
except ImportError:
    JsonPlusSerializer = None  # type: ignore

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


# ──────────────────────────────────────────────
# 환경변수
# ──────────────────────────────────────────────

CHECKPOINT_TABLE       = os.getenv("CHECKPOINT_TABLE",       "langgraph_checkpoints")
CHECKPOINT_TTL_DAYS    = int(os.getenv("CHECKPOINT_TTL_DAYS",    "7"))
CHECKPOINT_MAX_MESSAGES = int(os.getenv("CHECKPOINT_MAX_MESSAGES", "40"))


def _region() -> str:
    return (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-northeast-1").strip()


# ──────────────────────────────────────────────
# 테이블 초기화
# ──────────────────────────────────────────────

def ensure_checkpoints_table() -> None:
    """체크포인트 테이블이 없으면 생성합니다."""
    client = boto3.client("dynamodb", region_name=_region())
    name = CHECKPOINT_TABLE
    try:
        client.describe_table(TableName=name)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            print(f"[CHECKPOINT] 테이블 확인 실패 (non-fatal): {e}")
            return

    try:
        client.create_table(
            TableName=name,
            AttributeDefinitions=[{"AttributeName": "thread_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "thread_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=name)
        print(f"[CHECKPOINT] 테이블 생성 완료: {name}")
    except Exception as e:
        print(f"[CHECKPOINT] 테이블 생성 실패 (non-fatal): {e}")


# ──────────────────────────────────────────────
# messages 트리밍
# ──────────────────────────────────────────────

def _trim_messages(checkpoint: Checkpoint, max_messages: int) -> Checkpoint:
    """channel_values 내 messages를 최근 N개로 트리밍합니다."""
    try:
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages")
        if messages and len(messages) > max_messages:
            trimmed = list(messages)[-max_messages:]
            return {
                **checkpoint,
                "channel_values": {**channel_values, "messages": trimmed},
            }
    except Exception:
        pass
    return checkpoint


# ──────────────────────────────────────────────
# DynamoDB 체크포인터
# ──────────────────────────────────────────────

class DynamoDBCheckpointer(BaseCheckpointSaver):
    """
    LangGraph BaseCheckpointSaver — DynamoDB 구현체.

    - thread_id 당 최신 checkpoint 1개만 유지
    - put 시점에 messages를 CHECKPOINT_MAX_MESSAGES 개로 트리밍
    - TTL 자동 만료 (CHECKPOINT_TTL_DAYS)
    - DynamoDB 장애 시 graceful degradation (에러 로그 후 계속 동작)
    """

    def __init__(
        self,
        table_name: str = CHECKPOINT_TABLE,
        region: str | None = None,
        max_messages: int = CHECKPOINT_MAX_MESSAGES,
        ttl_days: int = CHECKPOINT_TTL_DAYS,
    ) -> None:
        serde = JsonPlusSerializer() if JsonPlusSerializer else None
        super().__init__(serde=serde)
        self.table_name = table_name
        self.region = region or _region()
        self.max_messages = max_messages
        self.ttl_days = ttl_days

    def _table(self):
        return boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)

    # ── 직렬화 ──

    def _serialize(self, obj: Any) -> str:
        if self.serde is not None:
            try:
                type_str, data = self.serde.dumps_typed(obj)
                return json.dumps({
                    "type": type_str,
                    "data": base64.b64encode(data).decode("ascii"),
                })
            except Exception:
                pass
        return json.dumps(obj, default=str, ensure_ascii=False)

    def _deserialize(self, raw: str) -> Any:
        if self.serde is not None:
            try:
                wrapper = json.loads(raw)
                if "type" in wrapper and "data" in wrapper:
                    data = base64.b64decode(wrapper["data"])
                    return self.serde.loads_typed((wrapper["type"], data))
            except Exception:
                pass
        return json.loads(raw)

    # ── BaseCheckpointSaver 인터페이스 ──

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        try:
            resp = self._table().get_item(Key={"thread_id": thread_id})
            item = resp.get("Item")
            if not item:
                return None
            checkpoint = self._deserialize(item["checkpoint_data"])
            metadata   = self._deserialize(item["metadata_data"])

            # pending_writes 복원 (interrupt 상태 재구성에 필요)
            pending_writes = None
            pw_raw  = item.get("pending_writes_data")
            task_id = str(item.get("pending_task_id") or "")
            if pw_raw and task_id:
                try:
                    raw_list = self._deserialize(pw_raw)   # [(channel, value), ...]
                    pending_writes = [
                        (task_id, channel, value)
                        for channel, value in raw_list
                    ]
                except Exception as e:
                    print(f"[CHECKPOINT] pending_writes 역직렬화 실패 (non-fatal): {e}")

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
                pending_writes=pending_writes,
            )
        except Exception as e:
            print(f"[CHECKPOINT] get_tuple 실패 (thread_id={thread_id}): {e}")
            return None

    def list(
        self, config: dict, *, filter=None, before=None, limit=None
    ) -> Iterator[CheckpointTuple]:
        result = self.get_tuple(config)
        if result:
            yield result

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any = None,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        try:
            trimmed = _trim_messages(checkpoint, self.max_messages)
            self._table().put_item(Item={
                "thread_id":       thread_id,
                "checkpoint_data": self._serialize(trimmed),
                "metadata_data":   self._serialize(metadata),
                "updated_at":      int(time.time()),
                "ttl":             int(time.time()) + self.ttl_days * 86400,
            })
        except Exception as e:
            print(f"[CHECKPOINT] put 실패 (thread_id={thread_id}): {e}")
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": "latest",
            },
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """interrupt() 호출 시 발생하는 pending_writes를 DynamoDB에 저장합니다."""
        if not writes:
            return
        thread_id = config["configurable"]["thread_id"]
        try:
            serialized = self._serialize([(channel, value) for channel, value in writes])
            self._table().update_item(
                Key={"thread_id": thread_id},
                UpdateExpression="SET pending_writes_data = :pw, pending_task_id = :tid",
                ExpressionAttributeValues={
                    ":pw":  serialized,
                    ":tid": task_id,
                },
            )
        except Exception as e:
            print(f"[CHECKPOINT] put_writes 실패 (thread_id={thread_id}): {e}")

    # ── async 래퍼 (sync 위임) ──

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        for item in self.list(config):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions=None) -> dict:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id) -> None:
        self.put_writes(config, writes, task_id)

    def delete(self, thread_id: str) -> None:
        """thread_id에 해당하는 체크포인트를 삭제합니다."""
        try:
            self._table().delete_item(Key={"thread_id": thread_id})
        except Exception as e:
            print(f"[CHECKPOINT] delete 실패 (thread_id={thread_id}): {e}")

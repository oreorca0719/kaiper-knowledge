from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


def _region() -> str:
    return (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-northeast-1").strip()


def _table_name() -> str:
    return (os.getenv("USERS_TABLE") or "langgraph_users").strip()


def dynamodb_resource():
    # App Runner에서는 "Instance role" (IAM role)로 자동 크리덴셜이 공급됩니다.
    return boto3.resource("dynamodb", region_name=_region())


def users_table():
    return dynamodb_resource().Table(_table_name())


def ensure_users_table_if_enabled() -> None:
    """
    기본은 '테이블이 이미 존재한다' 가정.
    - CREATE_USERS_TABLE=1 이면 없을 때 생성 시도(권한 필요).
    - App Runner에서 보통 IaC로 테이블을 만들고, 앱은 접근만 하는 구성이 안전합니다.
    """
    create_enabled = (os.getenv("CREATE_USERS_TABLE") or "").strip() in ("1", "true", "True", "YES", "yes")
    if not create_enabled:
        return

    client = boto3.client("dynamodb", region_name=_region())
    name = _table_name()

    try:
        client.describe_table(TableName=name)
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code != "ResourceNotFoundException":
            raise

    # PK: email (S)
    client.create_table(
        TableName=name,
        AttributeDefinitions=[{"AttributeName": "email", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "email", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )

    # 활성화 대기
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=name)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    if not email:
        return None
    tbl = users_table()
    try:
        resp = tbl.get_item(Key={"email": email.strip().lower()})
        return resp.get("Item")
    except Exception:
        return None


def create_user_if_not_exists(email: str, name: str, password_hash: str) -> Dict[str, Any]:
    """
    최초 로그인/회원가입 겸용:
    - 없으면 생성(approved=False, is_admin=False)
    - 있으면 기존 반환(비번은 여기서 변경 X)
    """
    now = int(time.time())
    email_l = email.strip().lower()
    tbl = users_table()

    item = {
        "email": email_l,
        "user_id": str(uuid.uuid4()),
        "name": (name or "").strip() or email_l.split("@")[0],
        "password_hash": password_hash,
        "approved": False,
        "is_admin": False,
        "created_at": now,
        "updated_at": now,
    }

    try:
        tbl.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(email)",
        )
        return item
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
        # 이미 있으면 조회 반환
        existing = get_user_by_email(email_l)
        return existing or item


def update_login_timestamp(email: str) -> None:
    tbl = users_table()
    try:
        tbl.update_item(
            Key={"email": email.strip().lower()},
            UpdateExpression="SET updated_at = :u",
            ExpressionAttributeValues={":u": int(time.time())},
        )
    except Exception:
        pass


def ensure_admin_user(email: str, name: str, password_hash: str) -> None:
    """
    main.py의 ensure_initial_admin()에서 호출.
    - 없으면 생성(approved=True, is_admin=True)
    - 있으면 approved/is_admin만 True로 보정(비밀번호는 변경하지 않음)
    """
    now = int(time.time())
    email_l = email.strip().lower()
    tbl = users_table()

    # 1) 없으면 생성
    item = {
        "email": email_l,
        "user_id": str(uuid.uuid4()),
        "name": (name or "").strip() or "Admin",
        "password_hash": password_hash,
        "approved": True,
        "is_admin": True,
        "created_at": now,
        "updated_at": now,
    }
    try:
        tbl.put_item(Item=item, ConditionExpression="attribute_not_exists(email)")
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise

    # 2) 이미 있으면 승인/관리자만 보정
    tbl.update_item(
        Key={"email": email_l},
        UpdateExpression="SET approved = :a, is_admin = :i, updated_at = :u",
        ExpressionAttributeValues={":a": True, ":i": True, ":u": now},
    )


def list_users(limit: int = 200) -> list[Dict[str, Any]]:
    tbl = users_table()
    resp = tbl.scan(Limit=limit)
    return resp.get("Items", [])


def approve_user(email: str, approved: bool = True) -> None:
    tbl = users_table()
    tbl.update_item(
        Key={"email": email.strip().lower()},
        UpdateExpression="SET approved = :a, updated_at = :u",
        ExpressionAttributeValues={":a": bool(approved), ":u": int(time.time())},
    )
def set_admin(email: str, is_admin: bool = True) -> None:
    tbl = users_table()
    tbl.update_item(
        Key={"email": email.strip().lower()},
        UpdateExpression="SET is_admin = :i, updated_at = :u",
        ExpressionAttributeValues={":i": bool(is_admin), ":u": int(time.time())},
    )


def delete_user(email: str) -> None:
    tbl = users_table()
    tbl.delete_item(Key={"email": email.strip().lower()})


def set_department(email: str, department: str) -> None:
    tbl = users_table()
    tbl.update_item(
        Key={"email": email.strip().lower()},
        UpdateExpression="SET department = :d, updated_at = :u",
        ExpressionAttributeValues={":d": department.strip(), ":u": int(time.time())},
    )

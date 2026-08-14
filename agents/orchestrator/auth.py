"""
用户鉴权模块（简单登录系统）

提供用户名/密码注册与登录，密码使用标准库 hashlib.pbkdf2_hmac 加盐哈希，
不引入额外依赖。登录成功后返回 user_id，前端据此区分用户并隔离长期记忆。

说明：本项目为教学演示，user_id 直接由前端携带、无签名 token，
仅用于「区分不同用户」，不提供真实的安全鉴权能力。
"""

import hashlib
import os
import secrets

import mysql.connector

from agents.orchestrator.session_db import _get_db_config

_ITERATIONS = 120_000
_SALT_BYTES = 16


def _hash_password(password: str, salt: bytes | None = None) -> str:
    """PBKDF2 加盐哈希，返回 "salt_hex$hash_hex"。"""
    if salt is None:
        salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """校验密码是否匹配存储的哈希。"""
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    candidate = _hash_password(password, salt)
    return secrets.compare_digest(candidate, stored)


def register(username: str, password: str) -> tuple[int, str] | None:
    """注册新用户，成功返回 (user_id, username)，用户名重复返回 None。"""
    username = (username or "").strip()
    if not username or not password:
        raise ValueError("用户名和密码不能为空")

    config = _get_db_config()
    password_hash = _hash_password(password)
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    (username, password_hash),
                )
            except mysql.connector.IntegrityError:
                return None
            return cur.lastrowid, username


def login(username: str, password: str) -> tuple[int, str] | None:
    """校验用户名密码，成功返回 (user_id, username)，失败返回 None。"""
    username = (username or "").strip()
    if not username or not password:
        return None

    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
    if not row:
        return None
    user_id, stored_username, password_hash = row
    if not _verify_password(password, password_hash):
        return None
    return user_id, stored_username


def user_exists(user_id: int) -> bool:
    """校验用户是否存在。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            return cur.fetchone() is not None

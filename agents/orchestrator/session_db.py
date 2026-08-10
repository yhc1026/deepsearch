"""
会话与对话持久化模块

将会话元数据和对话轮次存入 MySQL，支持前端历史会话列表和上下文加载。
"""

import json
import logging
import os
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_db_config() -> dict:
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "connection_timeout": int(os.getenv("MYSQL_CONNECTION_TIMEOUT", "10")),
    }
    config = {k: v for k, v in config.items() if v is not None}
    for key in ("user", "password", "database"):
        if key not in config:
            raise ValueError(f"缺失数据库配置：{key}")
    return config


def upsert_session(thread_id: str, title: str) -> int:
    """创建或更新会话，返回 sessions.id。首次创建时 title 使用首轮 query 截断。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (thread_id, title) VALUES (%s, %s)"
                " ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP, title = VALUES(title)",
                (thread_id, title[:200]),
            )
            cur.execute("SELECT id FROM sessions WHERE thread_id = %s", (thread_id,))
            row = cur.fetchone()
            return row[0] if row else 0


def finish_turn(thread_id: str) -> None:
    """标记会话完成一轮，更新 status 和 turn_count。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = 'done', turn_count = turn_count + 1,"
                " updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                (thread_id,),
            )


def save_conversation(
    thread_id: str,
    user_query: str,
    assistant_result: str,
    files: list[dict] | None = None,
) -> int:
    """保存一轮对话，自动计算 turn_index。返回 conversations.id。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE thread_id = %s", (thread_id,))
            session_row = cur.fetchone()
            if not session_row:
                session_id = upsert_session(thread_id, user_query[:50])
            else:
                session_id = session_row[0]

            cur.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM conversations WHERE session_id = %s",
                (session_id,),
            )
            next_index = cur.fetchone()[0] + 1

            files_json = json.dumps(files, ensure_ascii=False) if files else None
            now = datetime.now()
            cur.execute(
                "INSERT INTO conversations (session_id, turn_index, user_query,"
                " assistant_result, files, created_at, finished_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (session_id, next_index, user_query, assistant_result, files_json, now, now),
            )
            return cur.lastrowid or 0


def list_sessions() -> list[dict]:
    """返回所有会话摘要，按更新时间倒序。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT id, thread_id, title, status, turn_count,"
                " DATE_FORMAT(updated_at, '%Y-%m-%d %H:%i:%S') AS updated_at"
                " FROM sessions ORDER BY updated_at DESC"
            )
            return cur.fetchall() or []


def get_conversations(thread_id: str) -> list[dict]:
    """返回某个会话的所有对话轮次，按 turn_index 升序。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE thread_id = %s", (thread_id,))
            session_row = cur.fetchone()
            if not session_row:
                return []
            session_id = session_row[0]

        with conn.cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT turn_index, user_query, assistant_result, files,"
                " DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%S') AS created_at"
                " FROM conversations WHERE session_id = %s ORDER BY turn_index ASC",
                (session_id,),
            )
            rows = cur.fetchall()
            for row in rows:
                if row.get("files") and isinstance(row["files"], str):
                    row["files"] = json.loads(row["files"])
                elif not row.get("files"):
                    row["files"] = []
            return rows or []


def get_context_summary(thread_id: str) -> str:
    """获取会话的全局上下文摘要（由异步摘要 Agent 写入）。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT context_summary FROM sessions WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else ""


def delete_session(thread_id: str) -> bool:
    """删除会话及关联的所有对话（级联删除）。返回是否成功。"""
    config = _get_db_config()
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE thread_id = %s", (thread_id,))
            return cur.rowcount > 0

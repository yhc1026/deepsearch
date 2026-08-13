"""
DAG 执行断点续跑模块

用 SQLite 持久化 orchestrator 的执行进度，支持进程崩溃后从断点恢复，
避免重跑已完成的子智能体调用（网络搜索 / 数据库查询 / 向量检索等）。

设计：
- 每个 session_id 一条记录，存完整 Plan + 已完成步骤结果 + batch 游标。
- 恢复条件：存在断点记录且 task_query 与本次请求一致；否则当作新任务。
- 正常完成后 clear；异常/取消则保留，由下次 load 的 query 匹配逻辑决定续跑或作废。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from agents.orchestrator.planner import Plan

# 放在 output 之外，避免被前端的 /api/files、/api/download 暴露
_DB_PATH = Path(__file__).parents[2] / "checkpoints" / "checkpoints.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id   TEXT PRIMARY KEY,
    task_query   TEXT NOT NULL,
    plan_json    TEXT NOT NULL,
    results_json TEXT NOT NULL,
    codes_json   TEXT NOT NULL,
    batch_index  INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
)
"""


class CheckpointStore:
    """SQLite 持久化的 DAG 执行断点存储（异步 aiosqlite 实现）。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_init(self) -> None:
        """建表（幂等），首次使用时执行。"""
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(_SCHEMA)
            await conn.commit()
        self._initialized = True

    async def save(
        self,
        session_id: str,
        task_query: str,
        plan: Plan,
        results: dict[str, str],
        result_codes: dict[str, str],
        batch_index: int,
    ) -> None:
        """保存断点。batch_index 为已完成的最后一个 batch 索引，-1 表示尚未开始执行。"""
        await self._ensure_init()
        now = datetime.now().isoformat(timespec="seconds")
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints
                    (session_id, task_query, plan_json, results_json,
                     codes_json, batch_index, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    task_query = excluded.task_query,
                    plan_json = excluded.plan_json,
                    results_json = excluded.results_json,
                    codes_json = excluded.codes_json,
                    batch_index = excluded.batch_index,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    task_query,
                    plan.model_dump_json(),
                    json.dumps(results, ensure_ascii=False),
                    json.dumps(result_codes, ensure_ascii=False),
                    batch_index,
                    now,
                ),
            )
            await conn.commit()

    async def load(self, session_id: str, task_query: str) -> Optional[dict[str, Any]]:
        """读取可恢复断点。query 不一致时作废旧断点并返回 None。"""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            stored_query = row["task_query"]
            plan_json = row["plan_json"]
            results_json = row["results_json"]
            codes_json = row["codes_json"]
            batch_index = int(row["batch_index"])

        if stored_query != task_query:
            await self.clear(session_id)
            return None

        return {
            "plan": Plan.model_validate_json(plan_json),
            "results": json.loads(results_json),
            "result_codes": json.loads(codes_json),
            "batch_index": batch_index,
        }

    async def clear(self, session_id: str) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE session_id = ?", (session_id,)
            )
            await conn.commit()

"""
长期记忆工具模块

为长期记忆 Agent 提供 LangChain 工具，底层复用 ChromaDB 的 long_term_memory 集合：
1. search_memory — 检索某用户的相似旧记忆（用于覆盖检测）
2. write_memory  — 写入或覆盖一条事实记忆（带溯源字段）
3. delete_memory — 删除一条记忆（应对「你记错了」的修正）

每条记忆是一个 chunk：text=事实，metadata 携带 user_id + 溯源字段。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from shared.agent_result import ERROR, HIT, MISS, make_result
from shared.monitor import monitor

from agents.vector_search.vector_store import vector_store

MEMORY_COLLECTION = "long_term_memory"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@tool
def search_memory(query: str, user_id: str) -> str:
    """检索某用户的已有长期记忆，用于检测是否需要覆盖旧记忆。

    返回与 query 语义/关键词相近的已有记忆及其 memory_id 和来源。

    :param query: 一条待写入的事实（自包含，用于相似检索）
    :param user_id: 用户 ID（隔离不同用户的记忆）
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(
        tool_name="search_memory", args={"query": query, "user_id": user_id}
    )
    try:
        results = vector_store.hybrid_search(
            collection_name=MEMORY_COLLECTION,
            query=query,
            top_k=5,
            where={"user_id": user_id},
        )
        if not results:
            return make_result(MISS, "未检索到该用户的相似旧记忆")

        items: list[str] = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            items.append(
                f"[{i + 1}] memory_id={r['id']}\n"
                f"    事实: {r['text'][:300]}\n"
                f"    来源: session={meta.get('source_session_id', '')[:8]}"
                f" turn={meta.get('source_turn_index', '')}"
            )
        return make_result(
            HIT, "检索到以下相似旧记忆：\n\n" + "\n".join(items)
        )

    except Exception as e:  # noqa: BLE001
        return make_result(ERROR, f"检索记忆失败: {str(e)}")


@tool
def write_memory(
    fact: str,
    user_id: str,
    source_session_id: str,
    source_turn_index: str,
    source_query: str,
    overwrite_memory_id: str = "",
) -> str:
    """写入或覆盖一条长期记忆（事实）。

    :param fact: 事实性内容，自包含的完整陈述
    :param user_id: 用户 ID
    :param source_session_id: 该事实来自的会话 ID
    :param source_turn_index: 该事实来自的轮次索引
    :param source_query: 该事实对应的用户原始问题
    :param overwrite_memory_id: 若要覆盖某条冲突旧记忆，传该旧记忆的 memory_id；新增则留空
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(
        tool_name="write_memory",
        args={"fact": fact, "user_id": user_id, "overwrite_memory_id": overwrite_memory_id},
    )
    try:
        fact_text = (fact or "").strip()
        if not fact_text:
            return make_result(ERROR, "事实内容不能为空")

        now = _now_iso()
        memory_id = overwrite_memory_id.strip() or f"mem_{uuid.uuid4().hex}"
        metadata: dict[str, Any] = {
            "user_id": user_id,
            "source_session_id": source_session_id,
            "source_turn_index": source_turn_index,
            "source_query": (source_query or "")[:500],
            "created_at": now,
            "updated_at": now,
        }

        vector_store.upsert_chunks(
            collection_name=MEMORY_COLLECTION,
            chunks=[{"id": memory_id, "text": fact_text, "metadata": metadata}],
        )

        action = "覆盖" if overwrite_memory_id.strip() else "新增"
        return make_result(
            HIT, f"已{action}记忆 memory_id={memory_id}: {fact_text[:200]}"
        )

    except Exception as e:  # noqa: BLE001
        return make_result(ERROR, f"写入记忆失败: {str(e)}")


@tool
def delete_memory(memory_id: str) -> str:
    """删除一条指定 memory_id 的长期记忆（应对用户指出「你记错了」的修正）。

    :param memory_id: 要删除的记忆 ID
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(tool_name="delete_memory", args={"memory_id": memory_id})
    try:
        if not memory_id:
            return make_result(ERROR, "memory_id 不能为空")
        n = vector_store.delete_chunks(MEMORY_COLLECTION, [memory_id])
        if n > 0:
            return make_result(HIT, f"已删除记忆 {memory_id}")
        return make_result(MISS, f"未找到记忆 {memory_id}")
    except Exception as e:  # noqa: BLE001
        return make_result(ERROR, f"删除记忆失败: {str(e)}")

"""
确定性长期记忆召回模块

供 orchestrator 直接调用（不经 A2A / LLM），从 long_term_memory 集合按 user_id 召回
与该用户当前问题相关的长期记忆。检索策略：dense 向量召回 → BGE rerank 精排 → 阈值门控
→ 时间衰减 → 来源多样性重排，无 BM25。

user_id 由调用方以 int 直接传入，直接拼进 where 过滤，全程不经 LLM。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from shared.agent_result import ERROR, HIT, MISS
from shared.monitor import monitor

from agents.vector_search.rerank import rerank
from agents.vector_search.vector_store import vector_store

from .settings import (
    MEMORY_RECALL_CANDIDATE_K,
    MEMORY_RECALL_DENSE_THRESHOLD,
    MEMORY_RECALL_HALF_LIFE_DAYS,
    MEMORY_RECALL_MAX_PER_SESSION,
    MEMORY_RECALL_THRESHOLD,
    MEMORY_RECALL_TOP_K,
)

MEMORY_COLLECTION = "long_term_memory"


def _age_days(created_at: Any) -> float:
    """解析 created_at（ISO 时间戳）并返回距今天数；解析失败返回 0（视为近期）。"""
    if not created_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - dt
        return max(age.total_seconds() / 86400.0, 0.0)
    except (ValueError, TypeError):
        return 0.0


def _time_decay(rerank_score: float, created_at: Any) -> float:
    """对 rerank 分数做时间衰减：final = score * exp(-age_days / half_life)。"""
    age = _age_days(created_at)
    return rerank_score * math.exp(-age / MEMORY_RECALL_HALF_LIFE_DAYS)


def _diversity_select(
    ranked: list[dict], max_per_session: int, top_k: int
) -> list[dict]:
    """贪心按 final 降序选取，source_session_id 每会话最多保留 max_per_session 条。"""
    selected: list[dict] = []
    session_count: dict[str, int] = {}
    for item in ranked:
        if len(selected) >= top_k:
            break
        session = str(item["metadata"].get("source_session_id", ""))
        if session and session_count.get(session, 0) >= max_per_session:
            continue
        selected.append(item)
        if session:
            session_count[session] = session_count.get(session, 0) + 1
    return selected


def recall_memory(user_id: int, query: str) -> tuple[str, str]:
    """确定性召回该用户的长期记忆，返回 (code, content)。

    :param user_id: 用户 ID（int，直接拼 where 过滤）
    :param query: 用户当前问题（自包含，用于相似检索）
    :return: (code, content)，HIT 时 content 为逐条事实文本
    """
    monitor.report_tool(
        tool_name="recall_memory",
        args={"user_id": user_id, "query": query},
    )
    try:
        candidates = vector_store.dense_search(
            collection_name=MEMORY_COLLECTION,
            query=query,
            top_k=MEMORY_RECALL_CANDIDATE_K,
            where={"user_id": str(user_id)},
        )
        if not candidates:
            return MISS, ""

        texts = [c["text"] for c in candidates]
        try:
            rerank_scores = {
                int(r.get("index", -1)): float(r.get("relevance_score", 0.0))
                for r in rerank(query, texts)
            }
            threshold = MEMORY_RECALL_THRESHOLD
            use_rerank = True
        except Exception:
            rerank_scores = {}
            threshold = MEMORY_RECALL_DENSE_THRESHOLD
            use_rerank = False

        scored: list[dict] = []
        for i, c in enumerate(candidates):
            if use_rerank:
                raw = rerank_scores.get(i, 0.0)
                if raw < threshold:
                    continue
                final = _time_decay(raw, c["metadata"].get("created_at"))
                scored.append({**c, "rerank_score": raw, "final_score": final})
            else:
                raw = float(c.get("score", 0.0))
                if raw < threshold:
                    continue
                scored.append({**c, "rerank_score": raw, "final_score": raw})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        selected = _diversity_select(
            scored, MEMORY_RECALL_MAX_PER_SESSION, MEMORY_RECALL_TOP_K
        )

        if not selected:
            return MISS, ""

        content = "\n".join(f"- {s['text']}" for s in selected)
        return HIT, content

    except Exception as e:  # noqa: BLE001
        return ERROR, f"记忆召回失败: {str(e)}"

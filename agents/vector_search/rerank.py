"""
Rerank 精排模块

调用硅基流动 BGE-Reranker-v2-m3，对粗排候选 chunk 与原始 query 逐对精排，
返回每个候选的相关性分数，供粗排/精排分数融合。
"""

from __future__ import annotations

from typing import Any

import requests

from .settings import RERANK_API_KEY, RERANK_BASE_URL, RERANK_MODEL


def rerank(query: str, documents: list[str]) -> list[dict[str, Any]]:
    """对候选 documents 做 query-chunk 精排。

    :param query: 原始查询（与候选 chunk 逐一比对）
    :param documents: 候选 chunk 文本列表
    :return: [{"index": int, "relevance_score": float}]，按 relevance_score 降序
    """
    if not documents:
        return []
    if not RERANK_API_KEY:
        raise RuntimeError("未配置 RERANK_API_KEY，无法调用精排服务")

    resp = requests.post(
        f"{RERANK_BASE_URL.rstrip('/')}/rerank",
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "return_documents": False,
        },
        headers={"Authorization": f"Bearer {RERANK_API_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)

"""
向量检索工具模块

为 Vector Search 子智能体提供 LangChain 工具：
1. search_knowledge_base — 多 query 混合检索（核心检索入口）
2. list_collections    — 列出可用知识库
3. ingest_document     — 将文档内容写入知识库

检索策略：
- LLM 将原始 query 改写为 3 个不同视角的查询变体
- 每个变体执行 hybrid_search（向量 + 关键词，RRF 融合）
- 所有结果合并去重，按 RRF 分数排序
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from shared.llm import model
from shared.agent_result import ERROR, HIT, MISS, make_result
from shared.monitor import monitor

from .ingestion import ingest_text
from .rerank import rerank
from .settings import (
    HYBRID_TOP_K,
    QUERY_VARIANTS,
    RERANK_CANDIDATE_K,
    RERANK_TOP_K,
)
from .vector_store import vector_store

_MULTI_QUERY_SYSTEM = """你是一个查询改写专家。你的任务是把用户的一个检索问题改写为 N 个不同视角的查询变体。

要求:
1. 每个变体从不同角度切入（不同关键词、不同表述方式）
2. 变体之间保持多样性，避免高度重复
3. 每个变体必须自包含，包含检索所需的完整关键词
4. 不要添加解释，直接输出 JSON 数组"""


def _generate_query_variants(query: str, n: int = 3) -> list[str]:
    """调用 LLM 生成 N 个不同视角的查询变体。"""
    if n <= 1:
        return [query]

    prompt = f"""原始查询：{query}

请生成 {n} 个不同视角的查询变体。输出格式：
["变体1", "变体2", "变体3"]"""

    try:
        resp = model.invoke([
            SystemMessage(content=_MULTI_QUERY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        content = str(resp.content) if resp.content else "[]"
        # 去掉可能的 markdown 围栏
        content = content.strip()
        if content.startswith("```"):
            first_nl = content.find("\n")
            if first_nl > 0:
                content = content[first_nl + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        variants = json.loads(content)
        if isinstance(variants, list) and len(variants) > 0:
            return [str(v) for v in variants[:n]]
    except Exception:
        pass

    return [query]


def _deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 text 内容去重，合并相同文本的分数和来源。"""
    seen: dict[str, dict[str, Any]] = {}
    for item in results:
        key = item["text"][:200]  # 以前 200 字符作为去重指纹
        if key in seen:
            seen[key]["score"] = max(seen[key]["score"], item["score"])
            existing_sources = seen[key].get("sources", [])
            new_sources = item.get("sources", [])
            seen[key]["sources"] = list(set(existing_sources + new_sources))
        else:
            seen[key] = item
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


def _minmax(values: list[float]) -> list[float]:
    """min-max 归一化到 [0,1]，单值或全相等时统一返回 1.0。"""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _fuse_scores(
    candidates: list[dict[str, Any]],
    rerank_scores: dict[int, float],
) -> list[dict[str, Any]]:
    """粗排分与精排分各自 min-max 归一化后按 50:50 加权，返回按总分降序的候选。"""
    coarse = [c["score"] for c in candidates]
    coarse_norm = _minmax(coarse)

    n = len(candidates)
    rerank_vals = [rerank_scores.get(i, 0.0) for i in range(n)]
    rerank_norm = _minmax(rerank_vals) if rerank_scores else [0.0] * n

    for i, c in enumerate(candidates):
        c["rerank_score"] = rerank_vals[i]
        c["final_score"] = 0.5 * coarse_norm[i] + 0.5 * rerank_norm[i]

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)


# =============================================================================
# LangChain Tools
# =============================================================================


@tool
def search_knowledge_base(
    query: str,
    collection: str = "default",
) -> str:
    """在向量知识库中执行多路混合检索。

    检索流程：
    1. 将 query 改写为 3 个不同视角的变体（多路召回）
    2. 每个变体执行 hybrid_search（语义向量 + 关键词匹配，RRF 融合）粗排召回
    3. 合并去重后取候选池，用 BGE-Reranker 对原始 query 精排
    4. 粗排/精排分数各自归一化后按 50:50 加权，取 top 结果

    :param query: 自包含的完整检索问题
    :param collection: 目标知识库名称，默认 "default"
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(
        tool_name="search_knowledge_base",
        args={"query": query, "collection": collection},
    )

    try:
        # Step 1: 生成多查询变体
        variants = _generate_query_variants(query, n=QUERY_VARIANTS)
        print(f"\033[37m[VectorSearch] 多路查询变体 ({len(variants)}):\033[0m")
        for vi, v in enumerate(variants):
            print(f"  {vi + 1}. {v[:100]}")

        # Step 2: 每个变体执行混合检索（粗排召回，RRF 融合）
        all_results: list[dict[str, Any]] = []
        for v in variants:
            results = vector_store.hybrid_search(
                collection_name=collection,
                query=v,
                top_k=HYBRID_TOP_K,
            )
            all_results.extend(results)

        # Step 3: 去重 + 排序，取候选池进入精排
        merged = _deduplicate_results(all_results)
        candidates = merged[:RERANK_CANDIDATE_K]

        if not candidates:
            return make_result(MISS, f"知识库「{collection}」中未找到相关内容")

        # Step 4: rerank 精排（原始 query 与候选 chunk 逐对比对）
        candidate_texts = [c["text"] for c in candidates]
        rerank_scores: dict[int, float] = {}
        try:
            for r in rerank(query, candidate_texts):
                rerank_scores[int(r.get("index", -1))] = float(r.get("relevance_score", 0.0))
            print(f"\033[37m[VectorSearch] 精排完成: {len(rerank_scores)}/{len(candidates)} 条\033[0m")
        except Exception as e:
            print(f"\033[33m[VectorSearch] 精排失败，降级为纯粗排: {e}\033[0m")

        # Step 5: 粗排/精排分数各自 min-max 归一化后按 50:50 加权，取 top
        final = _fuse_scores(candidates, rerank_scores)
        final = final[:RERANK_TOP_K]

        # Step 6: 格式化输出
        items: list[str] = []
        for i, item in enumerate(final):
            sources = item.get("sources", [])
            source_tag = "+".join(sources)
            meta = item.get("metadata", {})
            src_file = meta.get("source", "未知来源")
            items.append(
                f"[{i + 1}] (综合: {item['final_score']:.3f}, 粗排: {item['score']:.3f}, "
                f"精排: {item.get('rerank_score', 0.0):.3f}, 召回: {source_tag}, 来源: {src_file})\n"
                f"{item['text'][:600]}"
            )

        result_text = (
            f"从知识库「{collection}」检索到 {len(final)} 条最相关内容"
            f"（{len(variants)}路粗排召回 + rerank 精排）：\n\n"
            + "\n\n---\n\n".join(items)
        )

        print(f"\033[37m[VectorSearch] 检索完成: 返回 {len(final)} 条结果\033[0m")
        return make_result(HIT, result_text)

    except Exception as e:
        return make_result(ERROR, f"向量检索异常: {str(e)}")


@tool
def list_collections() -> str:
    """列出所有可用的向量知识库。

    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(tool_name="list_collections")

    try:
        collections = vector_store.list_collections()
        if not collections:
            return make_result(MISS, "当前没有任何向量知识库")

        lines = ["可用的向量知识库:"]
        for c in collections:
            stats = vector_store.collection_stats(c)
            lines.append(f"  - {c} ({stats.get('chunk_count', 0)} 个文档块)")
        return make_result(HIT, "\n".join(lines))

    except Exception as e:
        return make_result(ERROR, f"查询知识库列表异常: {str(e)}")


@tool
def ingest_document(
    content: str,
    collection: str = "default",
    source_name: str = "",
) -> str:
    """将文本内容分块并写入向量知识库。

    自动将长文本按语义边界切分为多个 chunk，生成 Embedding 后写入 ChromaDB。
    支持纯文本和 Markdown 格式。

    :param content: 要写入的文档内容（纯文本或 Markdown）
    :param collection: 目标知识库名称，默认 "default"
    :param source_name: 文档来源标识（如文件名），用于结果溯源
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(
        tool_name="ingest_document",
        args={"collection": collection, "source_name": source_name, "length": len(content)},
    )

    try:
        metadata = {}
        if source_name:
            metadata["source"] = source_name

        count = ingest_text(content, collection_name=collection, metadata=metadata)
        return make_result(
            HIT,
            f"已成功将文档「{source_name or '(未命名)'}」写入知识库「{collection}」"
            f"（共 {count} 个文本块）",
        )

    except Exception as e:
        return make_result(ERROR, f"文档写入异常: {str(e)}")

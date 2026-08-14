"""
ChromaDB 向量库封装

提供三个检索能力：
1. dense_search  — 纯向量语义检索
2. keyword_search — 基于 BM25 的关键词检索
3. hybrid_search  — 融合以上两者，RRF 重排序

每个知识库对应一个 ChromaDB Collection。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embedding import embed, embed_query
from .settings import CHROMA_PERSIST_DIR

# ---------------------------------------------------------------------------
# 中文 / 英文关键词分词
# ---------------------------------------------------------------------------

# 常见停用词，匹配时跳过
_STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "than", "too",
    "very", "just", "about", "now", "also",
}


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：中文按 2-gram 切分，英文按空格切词。"""
    tokens: list[str] = []
    # 英文/数字段：按空格和标点切分
    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if word not in _STOP_WORDS and len(word) > 1:
            tokens.append(word)
    # 中文段：用 2-gram
    chinese_chars = re.findall(r"[一-鿿]+", text)
    for segment in chinese_chars:
        for i in range(len(segment) - 1):
            bigram = segment[i : i + 2]
            if bigram not in _STOP_WORDS:
                tokens.append(bigram)
        # 也加入单字（捕获单字关键词）
        for ch in segment:
            if ch not in _STOP_WORDS and ch.strip():
                tokens.append(ch)
    return tokens


# BM25 超参数
_BM25_K1 = 1.5   # 词频饱和度
_BM25_B = 0.75   # 文档长度归一化强度


def _idf(df: int, n_docs: int) -> float:
    """BM25 逆文档频率：稀有词权重高，常见词被抑制。"""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_len: int,
    avgdl: float,
    df: dict[str, int],
    n_docs: int,
) -> float:
    """计算文档相对查询的 BM25 得分。

    标准 BM25 公式：
        Σ IDF(q) · f(q,D) · (k1+1) / (f(q,D) + k1 · (1 - b + b · |D|/avgdl))
    """
    if not query_tokens:
        return 0.0

    # 文档词频
    tf: dict[str, int] = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    score = 0.0
    for q in query_tokens:
        f = tf.get(q, 0)
        if f == 0:
            continue
        idf = _idf(df.get(q, 0), n_docs)
        norm = 1.0 - _BM25_B + _BM25_B * (doc_len / avgdl)
        score += idf * (f * (_BM25_K1 + 1.0)) / (f + _BM25_K1 * norm)
    return score


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

def _clean_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """清洗 metadata：ChromaDB 只接受 str/int/float/bool 值。"""
    clean: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

class VectorStore:
    """ChromaDB 向量存储与混合检索。

    每个知识库是一个 Collection，文档分块以 (id, text, metadata) 形式存储。
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        if persist_dir is None:
            persist_dir = CHROMA_PERSIST_DIR
        # 支持相对路径：转为相对于项目根目录
        if not os.path.isabs(persist_dir):
            project_root = Path(__file__).parents[2]
            persist_dir = str(project_root / persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """列出所有知识库名称。"""
        return [c.name for c in self._client.list_collections()]

    def get_or_create_collection(self, name: str):
        """获取或创建 Collection。"""
        return self._client.get_or_create_collection(name=name)

    def delete_collection(self, name: str) -> None:
        """删除知识库。"""
        try:
            self._client.delete_collection(name=name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 文档写入
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """批量写入文档块。

        每个 chunk 格式: {"text": str, "metadata": dict | None}
        metadata 可包含: source, page, title, chunk_index 等自由字段。

        返回写入的块数量。
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        texts = [c["text"] for c in chunks]
        vectors = embed(texts)
        ids = [
            f"{collection_name}_{i}_{hash(t[:40])}"
            for i, t in enumerate(texts)
        ]
        metadatas = [_clean_metadata(c.get("metadata") or {}) for c in chunks]

        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        return len(chunks)

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """按稳定 id 写入/覆盖文档块（用于长期记忆更新）。

        每个 chunk 格式: {"id": str, "text": str, "metadata": dict | None}
        id 相同则覆盖原块，否则新增。返回处理的块数量。
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        vectors = embed(texts)
        metadatas = [_clean_metadata(c.get("metadata") or {}) for c in chunks]

        collection.upsert(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )
        return len(chunks)

    def delete_chunks(self, collection_name: str, ids: list[str]) -> int:
        """按 id 删除文档块。返回删除数量（ChromaDB 不返回计数时按请求 id 数）。"""
        if not ids:
            return 0
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            return 0
        collection.delete(ids=ids)
        return len(ids)

    def get_chunks(
        self,
        collection_name: str,
        where: dict | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按条件读取文档块（含 id/text/metadata）。"""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            return []
        kwargs: dict[str, Any] = {"where": where}
        if limit is not None:
            kwargs["limit"] = limit
        data = collection.get(**kwargs)
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        result: list[dict[str, Any]] = []
        for i in range(len(ids)):
            result.append({
                "id": ids[i] if i < len(ids) else "",
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            })
        return result

    # ------------------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------------------

    def dense_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """纯向量语义检索。可选 where 过滤（ChromaDB 语法，如 {"user_id": "3"}）。"""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            return []

        query_vec = embed_query(query)
        results = collection.query(
            query_embeddings=[query_vec], n_results=top_k, where=where
        )
        return _flatten_chroma_results(results)

    # ------------------------------------------------------------------
    # 关键词检索
    # ------------------------------------------------------------------

    def keyword_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """基于 BM25 的关键词检索。

        原理：从 Collection 中取出文档（可按 where 过滤），预计算语料统计
        （词频、文档频率、平均文档长度）后按 BM25 分数排序。
        适合小规模知识库（< 1 万条），大规模场景建议改用外部 BM25 索引。
        """
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            return []

        # ChromaDB get 默认返回全部，limit 可控制上限
        all_data = collection.get(where=where)
        documents = all_data["documents"]
        if not documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # 预计算语料统计：每个文档的分词、长度，以及全库文档频率
        doc_tokens_list: list[list[str]] = []
        doc_lengths: list[int] = []
        df: dict[str, int] = {}
        for doc_text in documents:
            tokens = _tokenize(str(doc_text)) if doc_text is not None else []
            doc_tokens_list.append(tokens)
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        n_docs = len(documents)
        avgdl = sum(doc_lengths) / n_docs if n_docs else 0.0

        scored: list[tuple[float, dict[str, Any]]] = []
        for i, doc_text in enumerate(documents):
            if doc_text is None:
                continue
            score = _bm25_score(
                query_tokens,
                doc_tokens_list[i],
                doc_lengths[i],
                avgdl,
                df,
                n_docs,
            )
            if score > 0:
                meta = (all_data["metadatas"] or [{}])[i] if i < len(all_data["metadatas"] or []) else {}
                scored.append((
                    score,
                    {
                        "id": (all_data["ids"] or [""])[i] if i < len(all_data["ids"] or []) else "",
                        "text": str(doc_text),
                        "metadata": meta or {},
                        "score": score,
                        "source": "keyword",
                    },
                ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    # 混合检索
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索：向量语义 + 关键词匹配，RRF 融合排序。

        并行执行两种检索，然后用 Reciprocal Rank Fusion 合并结果，
        同时利用语义理解和精确关键词匹配的优势。
        """
        dense_results = self.dense_search(collection_name, query, top_k=top_k * 2, where=where)
        keyword_results = self.keyword_search(collection_name, query, top_k=top_k * 2, where=where)

        # RRF 融合
        merged: dict[str, dict[str, Any]] = {}
        rrf_k = 60  # RRF 平滑常数

        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if doc_id in merged:
                merged[doc_id]["score"] = merged[doc_id]["score"] + rrf_score
                merged[doc_id]["sources"] = merged[doc_id].get("sources", []) + ["dense"]
            else:
                merged[doc_id] = {**item, "score": rrf_score, "sources": ["dense"]}

        for rank, item in enumerate(keyword_results):
            doc_id = item["id"]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if doc_id in merged:
                merged[doc_id]["score"] = merged[doc_id]["score"] + rrf_score
                merged[doc_id]["sources"] = merged[doc_id].get("sources", []) + ["keyword"]
            else:
                merged[doc_id] = {**item, "score": rrf_score, "sources": ["keyword"]}

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def collection_stats(self, collection_name: str) -> dict[str, Any]:
        """返回知识库基本统计。"""
        try:
            collection = self._client.get_collection(name=collection_name)
            count = collection.count()
            return {"name": collection_name, "chunk_count": count}
        except Exception:
            return {"name": collection_name, "chunk_count": 0, "error": "知识库不存在"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _flatten_chroma_results(results: dict) -> list[dict[str, Any]]:
    """把 ChromaDB query 返回的嵌套结构拍平为统一格式。"""
    items: list[dict[str, Any]] = []
    ids_list = results.get("ids", [[]])[0] if results.get("ids") else []
    docs_list = results.get("documents", [[]])[0] if results.get("documents") else []
    metas_list = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    for i in range(len(ids_list)):
        item: dict[str, Any] = {
            "id": ids_list[i] if i < len(ids_list) else "",
            "text": docs_list[i] if i < len(docs_list) else "",
            "metadata": metas_list[i] if i < len(metas_list) else {},
            "score": 1.0 - distances[i] if i < len(distances) else 0.0,
            "source": "dense",
        }
        items.append(item)
    return items


# 全局单例
vector_store = VectorStore()

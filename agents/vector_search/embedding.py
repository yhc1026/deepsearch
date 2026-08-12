"""
OpenAI Embedding 封装

复用项目 .env 中的 API_KEY 和 BASE_URL，
对上层暴露 embed / embed_query 两个简洁接口。
"""

from openai import OpenAI

from .settings import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL

_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)


_BATCH_SIZE = 10  # API 单次最大条数（阿里百炼 text-embedding-v4 限制 10）

def embed(texts: list[str]) -> list[list[float]]:
    """批量文本转向量，自动分批并截断超长文本。"""
    truncated = [t[:8000] for t in texts]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(truncated), _BATCH_SIZE):
        batch = truncated[i : i + _BATCH_SIZE]
        resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        by_index = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend(d.embedding for d in by_index)

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """单条查询转向量。"""
    return embed([text])[0]

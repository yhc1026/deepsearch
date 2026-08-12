"""
向量检索配置加载

复用项目 .env 中的 LLM API 配置（API_KEY / BASE_URL），
额外读取 ChromaDB 持久化路径和 Embedding 模型名。
"""

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# Embedding API 配置：优先用独立配置，未设置时回退到 LLM 的 API_KEY/BASE_URL
# 因为 DeepSeek 等部分 LLM 提供商不支持 Embedding 接口
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ChromaDB 持久化目录
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 检索参数
HYBRID_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "10"))
QUERY_VARIANTS = int(os.getenv("VECTOR_SEARCH_VARIANTS", "3"))
CHUNK_SIZE = int(os.getenv("VECTOR_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("VECTOR_CHUNK_OVERLAP", "150"))

"""
向量检索智能体服务 (Port 8004)

独立的 A2A Agent 服务，封装基于 ChromaDB + OpenAI Embedding 的混合检索能力。
接收自包含的检索问题，执行多路召回（3 query 变体 × 向量+关键词混合检索）后返回汇总结果。

启动方式: uv run uvicorn agents.vector_search.server:app --port 8004
"""

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from agents.vector_search.retrieval_tools import (
    ingest_document,
    list_collections,
    search_knowledge_base,
)

vector_config = sub_agents_content["vector_search"]

service = A2AAgentService(
    name=vector_config["name"],
    description=vector_config["description"],
    tools=[search_knowledge_base, list_collections, ingest_document],
    system_prompt=vector_config["system_prompt"],
    skills_dir=vector_config.get("skills_dir"),
    base_url="http://localhost:8004",
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)

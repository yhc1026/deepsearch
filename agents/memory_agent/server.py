"""
长期记忆智能体服务 (Port 8005)

独立的 A2A Agent 服务，负责从对话中提取事实性内容，检索冲突后写入用户的
全局长期记忆（ChromaDB long_term_memory 集合，按 user_id 隔离并带溯源字段）。

启动方式: uv run uvicorn agents.memory_agent.server:app --port 8005
"""

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from agents.memory_agent.memory_tools import (
    delete_memory,
    search_memory,
    write_memory,
)

memory_config = sub_agents_content["memory"]

service = A2AAgentService(
    name=memory_config["name"],
    description=memory_config["description"],
    tools=[search_memory, write_memory, delete_memory],
    system_prompt=memory_config["system_prompt"],
    skills_dir=memory_config.get("skills_dir"),
    base_url="http://localhost:8005",
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)

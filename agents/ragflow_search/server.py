"""
RAGFlow 知识库智能体服务 (Port 8003)

独立的 A2A Agent 服务，封装 RAGFlow 企业内部非结构化文档检索能力。
接收自包含的检索问题，执行 get_assistant_list → create_ask_delete 工作流后返回结果。

启动方式: uv run uvicorn agents.ragflow_search.server:app --port 8003
"""

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from agents.ragflow_search.ragflow_tools import create_ask_delete, get_assistant_list

ragflow_config = sub_agents_content["ragflow"]

service = A2AAgentService(
    name=ragflow_config["name"],
    description=ragflow_config["description"],
    tools=[get_assistant_list, create_ask_delete],
    system_prompt=ragflow_config["system_prompt"],
    skills_dir=ragflow_config.get("skills_dir"),
    base_url="http://localhost:8003",
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)

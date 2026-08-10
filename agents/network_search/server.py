"""
网络搜索智能体服务 (Port 8001)

独立的 A2A Agent 服务，封装 Tavily 互联网搜索能力。
接收自包含的搜索 query，执行多角度检索后返回汇总结果。

启动方式: uv run uvicorn agents.network_search.server:app --port 8001
"""

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from agents.network_search.tavily_tool import internet_search

tavily_config = sub_agents_content["tavily"]

service = A2AAgentService(
    name=tavily_config["name"],
    description=tavily_config["description"],
    tools=[internet_search],
    system_prompt=tavily_config["system_prompt"],
    skills_dir=tavily_config.get("skills_dir"),
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

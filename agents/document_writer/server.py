"""
文档撰写智能体服务 (Port 8006)

独立的 A2A Agent 服务，封装 Markdown 文档生成与 Markdown→PDF 转换能力。
接收自包含的撰写请求（需求 + 资料），由内部 LLM 组织正文并写入文档。

启动方式: uv run uvicorn agents.document_writer.server:app --port 8006
"""

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from agents.document_writer.markdown_tools import generate_markdown
from agents.document_writer.pdf_tools import convert_md_to_pdf

doc_config = sub_agents_content["document_writer"]

service = A2AAgentService(
    name=doc_config["name"],
    description=doc_config["description"],
    tools=[generate_markdown, convert_md_to_pdf],
    system_prompt=doc_config["system_prompt"],
    skills_dir=doc_config.get("skills_dir"),
    base_url="http://localhost:8006",
)

service.create_agent()
app = service.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)

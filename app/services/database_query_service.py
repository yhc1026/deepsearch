"""
数据库查询智能体服务 (Port 8002)

独立的 A2A Agent 服务，封装 MySQL 结构化数据查询能力。
接收自包含的数据查询描述，优先通过 MCP 协议调用标准化数据库工具；
当 MCP 工具不足以应对复杂查询时，回退到手写 SQL 模式。

MCP Server 通过 stdio 子进程自动启动，无需独立端口。
MCP 工具在 FastAPI lifespan 中异步加载，利用 uvicorn 已有的事件循环。

启动方式: uv run uvicorn app.services.database_query_service:app --port 8002
"""

import traceback
from contextlib import asynccontextmanager

from app.services.base import A2AAgentService
from app.shared.prompts import sub_agents_content

db_config = sub_agents_content["db"]

# 兜底工具：始终可用的直连 MySQL 工具（MCP 不可用时自动回退）
from app.tools.db_tools import execute_sql_query, get_table_data, list_sql_tables

_fallback_tools = [list_sql_tables, get_table_data, execute_sql_query]

# 服务实例先用兜底工具创建，MCP 工具在 lifespan 中异步加载后再更新
service = A2AAgentService(
    name=db_config["name"],
    description=db_config["description"],
    tools=_fallback_tools,
    system_prompt=db_config["system_prompt"],
    skills_dir=db_config.get("skills_dir"),
)


@asynccontextmanager
async def db_lifespan(app):
    """FastAPI lifespan：在 uvicorn 事件循环内异步加载 MCP 工具并创建 agent。"""
    global service
    mcp_exit_stack = None

    try:
        from app.mcp.client import load_mcp_tools_async

        mcp_tools, mcp_exit_stack = await load_mcp_tools_async()
        all_tools = list(mcp_tools) + _fallback_tools

        _mcp_tool_names = "\n".join(
            f"       - {t.name}: {t.description.split(chr(10))[0][:80]}"
            for t in mcp_tools
        )
        _fallback_tool_names = "\n".join(
            f"       - {t.name}: {t.description.split(chr(10))[0][:80]}"
            for t in _fallback_tools
        )
        _mcp_instruction = f"""
【MCP 工具（优先使用，标准化协议）】
{_mcp_tool_names}

【兜底工具（MCP 工具无法满足需求时使用，支持精细化手写 SQL）】
{_fallback_tool_names}

策略：优先使用 MCP 工具完成常规查询。只有当 MCP 工具确实无法满足复杂需求
（如特殊的多表嵌套子查询、窗口函数等）时，才回退到兜底工具手写 SQL。
"""

        service.tools = all_tools
        service.system_prompt = db_config["system_prompt"] + "\n" + _mcp_instruction
        service.create_agent()
        print(
            f"[MySQL Agent] MCP 工具加载成功: {[t.name for t in mcp_tools]}"
        )
    except Exception:
        mcp_load_error = traceback.format_exc()
        service.create_agent()
        print(
            f"[MySQL Agent] MCP 加载失败，使用直连工具兜底:\n{mcp_load_error}"
        )

    yield

    if mcp_exit_stack:
        await mcp_exit_stack.aclose()


app = service.build_app(lifespan=db_lifespan)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)

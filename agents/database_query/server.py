"""
数据库查询智能体服务 (Port 8002)

独立的 A2A Agent 服务，封装 MySQL 结构化数据查询能力。
接收自包含的数据查询描述，优先通过 MCP 协议调用标准化数据库工具；
当 MCP 工具不足以应对复杂查询时，回退到手写 SQL 模式。

MCP Server 作为独立 HTTP 服务运行（默认 port 8100），Agent 通过后台定时轮询
自动发现工具变更，实现热插拔。

启动方式: uv run uvicorn agents.database_query.server:app --port 8002
"""

import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager

from shared.A2A_base_service import A2AAgentService
from shared.prompts import sub_agents_content
from shared.logger import setup_logging

logger = logging.getLogger(__name__)

db_config = sub_agents_content["db"]

# 兜底工具：始终可用的直连 MySQL 工具（MCP 不可用时自动回退）
from agents.database_query.db_tools import execute_sql_query, get_table_data, list_sql_tables

_fallback_tools = [list_sql_tables, get_table_data, execute_sql_query]

# 服务实例先用兜底工具创建，MCP 工具在 lifespan 中异步加载后再更新
service = A2AAgentService(
    name=db_config["name"],
    description=db_config["description"],
    tools=_fallback_tools,
    system_prompt=db_config["system_prompt"],
    skills_dir=db_config.get("skills_dir"),
)

# 后台刷新间隔（秒），可通过环境变量调整
REFRESH_INTERVAL = int(os.getenv("MCP_TOOL_REFRESH_INTERVAL", "30"))


def _build_tool_instruction(mcp_tools: list, fallback_tools: list) -> str:
    """构建注入 system_prompt 的工具使用说明"""
    mcp_lines = "\n".join(
        f"       - {t.name}: {t.description.split(chr(10))[0][:80]}"
        for t in mcp_tools
    )
    fallback_lines = "\n".join(
        f"       - {t.name}: {t.description.split(chr(10))[0][:80]}"
        for t in fallback_tools
    )
    return f"""
【MCP 工具（优先使用，标准化协议）】
{mcp_lines}

【兜底工具（MCP 工具无法满足需求时使用，支持精细化手写 SQL）】
{fallback_lines}

策略：优先使用 MCP 工具完成常规查询。只有当 MCP 工具确实无法满足复杂需求
（如特殊的多表嵌套子查询、窗口函数等）时，才回退到兜底工具手写 SQL。
"""


async def _refresh_loop():
    """后台定时刷新 MCP 工具列表，检测到变化时热更新 agent"""
    global service

    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        try:
            from agents.database_query.mcp_client import refresh_mcp_tools_async

            mcp_tools, changed = await refresh_mcp_tools_async()
            if changed:
                all_tools = list(mcp_tools) + _fallback_tools
                instruction = _build_tool_instruction(mcp_tools, _fallback_tools)
                service.recreate_agent(
                    tools=all_tools,
                    system_prompt=db_config["system_prompt"] + "\n" + instruction,
                )
                logger.info(f"工具列表已热更新: {[t.name for t in mcp_tools]}")
        except Exception:
            logger.debug(f"后台刷新跳过（MCP Server 暂不可达，缓存兜底）")


@asynccontextmanager
async def db_lifespan(app):
    """FastAPI lifespan：首次加载 MCP 工具，启动后台刷新任务。"""
    global service

    setup_logging()

    # 首次加载 MCP 工具
    try:
        from agents.database_query.mcp_client import get_mcp_tools

        mcp_tools = await get_mcp_tools(force_refresh=True)
        all_tools = list(mcp_tools) + _fallback_tools

        instruction = _build_tool_instruction(mcp_tools, _fallback_tools)
        service.tools = all_tools
        service.system_prompt = db_config["system_prompt"] + "\n" + instruction
        service.create_agent()
        logger.info(f"MCP 工具首次加载成功: {[t.name for t in mcp_tools]}")
    except Exception:
        mcp_load_error = traceback.format_exc()
        service.create_agent()
        logger.error(f"MCP 首次加载失败，使用直连工具兜底:\n{mcp_load_error}")

    # 启动后台刷新任务
    refresh_task = asyncio.create_task(_refresh_loop())

    yield

    # 关闭后台任务和 MCP 缓存连接
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    try:
        from agents.database_query.mcp_client import close_mcp_cache

        await close_mcp_cache()
    except Exception:
        pass


app = service.build_app(lifespan=db_lifespan)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)

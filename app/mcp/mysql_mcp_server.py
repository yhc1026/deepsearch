"""
MySQL MCP Server (MCP 2.0 / Streamable HTTP + stdio)

通过 Streamable HTTP 或 stdio transport 向 MCP 客户端暴露数据库查询工具。
工具定义从 mcp_tools.yaml 动态加载，修改配置后重启即可增减工具。

启动方式:
  独立 HTTP 服务（生产）: uv run uvicorn app.mcp.mysql_mcp_server:http_app --port 8100
  stdio 模式（本地测试）: uv run python -m app.mcp.mysql_mcp_server
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(_project_root))

from dotenv import find_dotenv, load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from mysql.connector import Error as MySQLError
from mysql.connector import connect

load_dotenv(find_dotenv(), override=True)

# ── 配置文件路径 ──────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "mcp_tools.yaml"

# ── 数据库连接 ────────────────────────────────────────────────


def _get_db_config():
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
        "connection_timeout": int(os.getenv("MYSQL_CONNECTION_TIMEOUT", "10")),
    }


def _execute_sql(sql: str) -> str:
    config = {k: v for k, v in _get_db_config().items() if v is not None}
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        return f"数据库配置缺失：{', '.join(missing_keys)}"

    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                description = cursor.description
                if not description:
                    return "SQL 执行成功，但没有返回结果集。"
                columns = [desc[0] for desc in description]
                rows = cursor.fetchall()
                results = [",".join(map(str, row)) for row in rows]
                return ",".join(columns) + "\n" + "\n".join(results)
    except MySQLError as e:
        return f"查询异常：{str(e)}"


# ── YAML 工具加载 ─────────────────────────────────────────────


def load_tools_config() -> list[dict[str, Any]]:
    """从 mcp_tools.yaml 加载工具定义，每次调用都重新读取文件（支持热加载）。"""
    if not _CONFIG_PATH.exists():
        logger.warning(f"配置文件不存在: {_CONFIG_PATH}，使用内置默认工具")
        return _get_default_tools()

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tools = config.get("tools", [])
        if not tools:
            logger.warning("配置文件中无工具定义，使用内置默认工具")
            return _get_default_tools()
        return tools
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用内置默认工具")
        return _get_default_tools()


def _get_default_tools() -> list[dict[str, Any]]:
    """内置默认工具（兜底）"""
    return [
        {
            "name": "mysql_list_tables",
            "description": "查询当前数据库中所有可用的表名。返回值格式: '可用的表有：表1, 表2, 表3...'。这是所有数据库查询的第一步，必须先用此工具确认真实表名。",
            "handler": "list_tables",
            "params": [],
        },
        {
            "name": "mysql_get_schema",
            "description": "查询指定表的前 100 行数据，用于了解表结构和字段格式。调用此工具前应先用 mysql_list_tables 确认表名。返回值: CSV 格式，第一行是列名，之后是数据行。",
            "handler": "get_schema",
            "params": [
                {"name": "table_name", "type": "string", "description": "要预览的表名", "required": True},
            ],
        },
        {
            "name": "mysql_execute",
            "description": "执行一条自定义 SQL 查询（SELECT / SHOW 等只读语句）。适用于多表关联、筛选、聚合、排序等复杂查询。调用前请先通过 mysql_list_tables 和 mysql_get_schema 确认表名和字段名。返回值: CSV 格式，第一行是列名，之后是数据行。",
            "handler": "execute",
            "params": [
                {"name": "sql", "type": "string", "description": "要执行的 SQL 查询语句", "required": True},
            ],
        },
    ]


def _build_input_schema(params: list[dict]) -> dict:
    """从 YAML params 列表构建 MCP Tool inputSchema"""
    properties: dict[str, dict] = {}
    required: list[str] = []
    for p in params:
        properties[p["name"]] = {
            "type": p.get("type", "string"),
            "description": p.get("description", ""),
        }
        if p.get("required", False):
            required.append(p["name"])
    return {"type": "object", "properties": properties, "required": required}


# ── 工具执行 ──────────────────────────────────────────────────


def _list_tables_text() -> str:
    logger.debug(f"\033[94m查询数据库表名: SHOW TABLES\033[0m")
    result = _execute_sql("SHOW TABLES")
    if result.startswith("查询异常"):
        return result
    if not result.strip():
        return "没有可用的表"
    lines = result.strip().split("\n")
    if len(lines) <= 1:
        return f"可用的表有：{lines[0]}" if lines[0] else "没有可用的表"
    table_names = [line.split(",")[0] for line in lines[1:]]
    return f"可用的表有：{', '.join(table_names)}"


def _get_schema_text(table_name: str) -> str:
    logger.debug(f"\033[94m预览表数据: SELECT * FROM {table_name} LIMIT 100\033[0m")
    return _execute_sql(f"SELECT * FROM {table_name} LIMIT 100")


def _execute_text(sql: str) -> str:
    logger.debug(f"\n{'='*60}\n执行 SQL 查询:\n  \033[94m{sql}\033[0m\n{'='*60}\n")
    return _execute_sql(sql)


_HANDLER_MAP = {
    "list_tables": _list_tables_text,
    "get_schema": _get_schema_text,
    "execute": _execute_text,
}


# ── MCP Handlers ──────────────────────────────────────────────


async def on_list_tools(ctx, params: PaginatedRequestParams | None) -> ListToolsResult:
    tools_config = load_tools_config()
    tools = []
    for tc in tools_config:
        tools.append(
            Tool(
                name=tc["name"],
                description=tc["description"],
                inputSchema=_build_input_schema(tc.get("params", [])),
            )
        )
    return ListToolsResult(tools=tools)


async def on_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    tool_name = params.name
    args = params.arguments or {}

    tools_config = load_tools_config()
    handler_name = None
    for tc in tools_config:
        if tc["name"] == tool_name:
            handler_name = tc.get("handler", "")
            break

    if handler_name is None:
        return CallToolResult(
            content=[TextContent(type="text", text=f"未知工具: {tool_name}")],
            is_error=True,
        )

    handler = _HANDLER_MAP.get(handler_name)
    if handler is None:
        return CallToolResult(
            content=[TextContent(type="text", text=f"未注册的 handler: {handler_name}")],
            is_error=True,
        )

    # 按 handler 类型分发参数
    if handler_name == "list_tables":
        text = handler()
    elif handler_name == "get_schema":
        text = handler(args.get("table_name", ""))
    elif handler_name == "execute":
        text = handler(args.get("sql", ""))
    else:
        text = handler(**args)

    return CallToolResult(content=[TextContent(type="text", text=text)])


# ── Server 实例 ───────────────────────────────────────────────

server = Server(
    "MySQL MCP Server",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)

# Streamable HTTP 应用（供 uvicorn 挂载，独立部署用）
http_app = server.streamable_http_app()

# ── stdio 入口（本地测试用）───────────────────────────────────


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

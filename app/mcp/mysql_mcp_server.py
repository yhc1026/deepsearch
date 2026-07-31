"""
MySQL MCP Server (stdio 模式 / MCP 2.0)

通过 stdio transport 向 MCP 客户端暴露三个数据库查询工具。
工具逻辑和 db_tools.py 完全一致，只是调用协议从本地函数调用改为 MCP JSON-RPC。

启动方式: 由 database_query_service.py 作为子进程自动启动，不需要手动运行。
"""

import asyncio
import os
import sys
from pathlib import Path

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
from mysql.connector import Error, connect

load_dotenv(find_dotenv(), override=True)


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
    """执行一条 SQL 并返回 CSV 格式结果"""
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
    except Error as e:
        return f"查询异常：{str(e)}"


def _list_tables_text() -> str:
    print(f"[MySQL MCP] 查询数据库表名: \033[94mSHOW TABLES\033[0m", file=sys.stderr)
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
    print(
        f"[MySQL MCP] 预览表数据: \033[94mSELECT * FROM {table_name} LIMIT 100\033[0m",
        file=sys.stderr,
    )
    return _execute_sql(f"SELECT * FROM {table_name} LIMIT 100")


def _execute_text(sql: str) -> str:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[MySQL MCP] 执行 SQL 查询:", file=sys.stderr)
    print(f"  \033[94m{sql}\033[0m", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    return _execute_sql(sql)


async def on_list_tools(ctx, params: PaginatedRequestParams | None) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="mysql_list_tables",
                description=(
                    "查询当前数据库中所有可用的表名。"
                    "返回值格式: '可用的表有：表1, 表2, 表3...'。"
                    "这是所有数据库查询的第一步，必须先用此工具确认真实表名。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="mysql_get_schema",
                description=(
                    "查询指定表的前 100 行数据，用于了解表结构和字段格式。"
                    "调用此工具前应先用 mysql_list_tables 确认表名。"
                    "返回值: CSV 格式，第一行是列名，之后是数据行。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "要预览的表名",
                        }
                    },
                    "required": ["table_name"],
                },
            ),
            Tool(
                name="mysql_execute",
                description=(
                    "执行一条自定义 SQL 查询（SELECT / SHOW 等只读语句）。"
                    "适用于多表关联、筛选、聚合、排序等复杂查询。"
                    "调用前请先通过 mysql_list_tables 和 mysql_get_schema 确认表名和字段名。"
                    "返回值: CSV 格式，第一行是列名，之后是数据行。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "要执行的 SQL 查询语句",
                        }
                    },
                    "required": ["sql"],
                },
            ),
        ]
    )


async def on_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    tool_name = params.name
    args = params.arguments or {}

    if tool_name == "mysql_list_tables":
        text = _list_tables_text()
    elif tool_name == "mysql_get_schema":
        text = _get_schema_text(args.get("table_name", ""))
    elif tool_name == "mysql_execute":
        text = _execute_text(args.get("sql", ""))
    else:
        text = f"未知工具: {tool_name}"

    return CallToolResult(content=[TextContent(type="text", text=text)])


# MCP 2.0：通过 on_* 构造函数注册 handler，params 是 RequestParams 而非完整 Request
server = Server(
    "MySQL MCP Server",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
MCP 客户端工具加载模块

封装通过 stdio transport 连接 MCP Server、列出工具并包装为可调用函数的逻辑。
database_query_service.py 通过此模块加载 MySQL MCP 工具。
"""

import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client


def _json_schema_to_pydantic_field(schema: dict[str, Any]) -> Any:
    """将 JSON Schema 属性转为 Pydantic Field 参数，尽力而为"""
    field_type = str
    if "type" in schema:
        t = schema["type"]
        if t == "integer":
            field_type = int
        elif t == "number":
            field_type = float
        elif t == "boolean":
            field_type = bool
    return field_type


async def load_mcp_tools_async() -> tuple[list, AsyncExitStack]:
    """异步连接 MySQL MCP Server，获取 LangChain 工具列表。

    返回 (tools, exit_stack)，exit_stack 需保持存活直到服务关闭。
    调用方应通过 await close_mcp(exit_stack) 来清理资源。
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.mysql_mcp_server"],
        env={**os.environ},
    )

    exit_stack = AsyncExitStack()
    transports = await exit_stack.enter_async_context(stdio_client(server_params))
    read_stream, write_stream = transports
    session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    list_result = await session.list_tools()
    tools = []

    for tool_def in list_result.tools:
        tool_name = tool_def.name
        tool_description = tool_def.description or ""

        input_schema = tool_def.input_schema or {}
        properties = input_schema.get("properties", {}) or {}
        required = input_schema.get("required", []) or []
        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required
            default = ... if is_required else None
            field_kwargs: dict[str, Any] = {"default": default}
            desc = prop_schema.get("description", "")
            if desc:
                field_kwargs["description"] = desc
            fields[prop_name] = (
                _json_schema_to_pydantic_field(prop_schema),
                Field(**field_kwargs),
            )
        # 无参工具也要给空 object schema，避免 StructuredTool 推断出 kwargs 污染参数
        args_model = create_model(f"{tool_name}_args", **fields)

        def _make_caller(name: str):
            async def _call_mcp_tool(**kwargs):
                # 过滤掉 None，避免把可选空值传给 MCP 触发参数校验失败
                arguments = {k: v for k, v in kwargs.items() if v is not None}
                # 蓝色日志：打印 MCP 工具名和参数
                args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
                msg = f"[MySQL MCP] 调用工具: {name}({args_str})" if arguments else f"[MySQL MCP] 调用工具: {name}"
                print(f"\033[94m{msg}\033[0m")
                result = await session.call_tool(name, arguments=arguments)
                texts = []
                for c in getattr(result, "content", None) or []:
                    if hasattr(c, "text") and c.text:
                        texts.append(c.text)
                body = "\n".join(texts) if texts else str(result)
                # MCP 2.0 CallToolResult 字段为 is_error
                if getattr(result, "is_error", False) or getattr(result, "isError", False):
                    return f"MCP 工具执行失败: {body}"
                return body

            return _call_mcp_tool

        tool = StructuredTool.from_function(
            name=tool_name,
            description=tool_description,
            coroutine=_make_caller(tool_name),
            args_schema=args_model,
        )
        tools.append(tool)

    return tools, exit_stack

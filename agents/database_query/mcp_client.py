"""
MCP 客户端工具加载模块

通过 Streamable HTTP transport 连接独立部署的 MCP Server，列出工具并包装为 LangChain StructuredTool。
支持 TTL 缓存 + 后台定时刷新，实现工具热插拔。

连接策略：list_tools / call_tool 均使用短生命周期 session，在同一 asyncio Task 内
enter + aclose，避免 anyio cancel scope 跨 Task 退出报错。
"""

import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── 配置 ──────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_MYSQL_SERVER_URL", "http://127.0.0.1:8100/mcp")
TOOL_CACHE_TTL = int(os.getenv("MCP_TOOL_CACHE_TTL", "30"))  # 缓存有效期（秒）

# ── 缓存状态（仅缓存工具包装对象，不持有长连接）────────────────
_cached_tools: list = []
_cache_timestamp: float = 0
_cache_tool_names: list[str] = []


def _get_mcp_client():
    """惰性导入 streamable_http_client（避免模块加载时触发网络连接）"""
    from mcp.client.streamable_http import streamable_http_client

    return streamable_http_client


def _json_schema_to_pydantic_field(schema: dict[str, Any]) -> Any:
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


async def _with_mcp_session(fn: Callable[[Any], Awaitable[T]]) -> T:
    """在同一 Task 内打开 MCP session、执行回调、再关闭，避免 cancel scope 跨 Task。"""
    from mcp import ClientSession

    exit_stack = AsyncExitStack()
    try:
        streamable_http_client = _get_mcp_client()
        transports = await exit_stack.enter_async_context(
            streamable_http_client(MCP_SERVER_URL)
        )
        # streamable_http 可能返回 (read, write) 或 (read, write, get_session_id)
        read_stream, write_stream = transports[0], transports[1]
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return await fn(session)
    finally:
        try:
            await exit_stack.aclose()
        except Exception as e:
            # 关闭阶段偶发异常不影响业务；记录后吞掉避免污染日志
            logger.debug(f"MCP session 关闭时忽略异常: {e}")


async def _discover_tools_from_server() -> list:
    """连接 MCP Server 列出工具，包装为每次调用自建短连接的 LangChain StructuredTool。"""
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    async def _list_and_wrap(session) -> list:
        list_result = await session.list_tools()
        tools: list = []

        for tool_def in list_result.tools:
            tool_name = tool_def.name
            if tool_def.description:
                tool_description = tool_def.description
            else:
                tool_description = ""

            if tool_def.input_schema:
                input_schema = tool_def.input_schema
            else:
                input_schema = {}

            raw_properties = input_schema.get("properties", {})
            if raw_properties:
                properties = raw_properties
            else:
                properties = {}

            raw_required = input_schema.get("required", [])
            if raw_required:
                required = raw_required
            else:
                required = []
            fields: dict[str, Any] = {}
            for prop_name, prop_schema in properties.items():
                is_required = prop_name in required
                if is_required:
                    default = ...
                else:
                    default = None
                field_kwargs: dict[str, Any] = {"default": default}
                desc = prop_schema.get("description", "")
                if desc:
                    field_kwargs["description"] = desc
                fields[prop_name] = (
                    _json_schema_to_pydantic_field(prop_schema),
                    Field(**field_kwargs),
                )
            args_model = create_model(f"{tool_name}_args", **fields)

            def _make_caller(name: str):
                async def _call_mcp_tool(**kwargs):
                    arguments = {k: v for k, v in kwargs.items() if v is not None}
                    args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
                    if arguments:
                        msg = f"[MySQL MCP] 调用工具: {name}({args_str})"
                    else:
                        msg = f"[MySQL MCP] 调用工具: {name}"
                    logger.debug(f"\033[94m{msg}\033[0m")

                    async def _invoke(sess):
                        result = await sess.call_tool(name, arguments=arguments)
                        texts = []
                        content_items = getattr(result, "content", None)
                        if content_items:
                            items = content_items
                        else:
                            items = []
                        for c in items:
                            if hasattr(c, "text") and c.text:
                                texts.append(c.text)
                        if texts:
                            body = "\n".join(texts)
                        else:
                            body = str(result)
                        if getattr(result, "is_error", False) or getattr(
                            result, "isError", False
                        ):
                            return f"MCP 工具执行失败: {body}"
                        return body

                    return await _with_mcp_session(_invoke)

                return _call_mcp_tool

            tool = StructuredTool.from_function(
                name=tool_name,
                description=tool_description,
                coroutine=_make_caller(tool_name),
                args_schema=args_model,
            )
            tools.append(tool)

        return tools

    return await _with_mcp_session(_list_and_wrap)


def _is_cache_valid() -> bool:
    return bool(_cached_tools) and (time.time() - _cache_timestamp) < TOOL_CACHE_TTL


# ── 公开 API ──────────────────────────────────────────────────


async def get_mcp_tools(force_refresh: bool = False) -> list:
    """获取 MCP 工具列表，带 TTL 缓存。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names

    if not force_refresh and _is_cache_valid():
        return _cached_tools

    try:
        new_tools = await _discover_tools_from_server()
        new_names = [t.name for t in new_tools]

        if not force_refresh and new_names == _cache_tool_names and _cached_tools:
            _cache_timestamp = time.time()
            return _cached_tools

        _cached_tools = new_tools
        _cache_timestamp = time.time()
        _cache_tool_names = new_names
        return _cached_tools

    except Exception as e:
        logger.error(f"MCP 工具刷新失败: {e}")
        if _cached_tools:
            return _cached_tools
        raise


async def refresh_mcp_tools_async() -> tuple[list, bool]:
    """强制刷新 MCP 工具列表，返回 (tools, changed)。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names

    old_names = list(_cache_tool_names)

    try:
        new_tools = await _discover_tools_from_server()
        new_names = [t.name for t in new_tools]

        if new_names == old_names and _cached_tools:
            _cache_timestamp = time.time()
            return _cached_tools, False

        _cached_tools = new_tools
        _cache_timestamp = time.time()
        _cache_tool_names = new_names
        return _cached_tools, True

    except Exception as e:
        logger.error(f"MCP 工具后台刷新失败: {e}")
        return _cached_tools, False


async def load_mcp_tools_async() -> tuple[list, AsyncExitStack]:
    """首次加载 MCP 工具（兼容旧接口）。"""
    tools = await get_mcp_tools(force_refresh=True)
    return tools, AsyncExitStack()


async def close_mcp_cache():
    """清空 MCP 工具缓存（短连接模式无需显式关 session）。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names
    _cached_tools = []
    _cache_timestamp = 0
    _cache_tool_names = []

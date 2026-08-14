"""
MCP 客户端工具加载模块

通过 Streamable HTTP transport 连接独立部署的 MCP Server，列出工具并包装为 LangChain StructuredTool。
支持 TTL 缓存 + 后台定时刷新，实现工具热插拔。

连接策略：维持一个长连接 session，list_tools / call_tool 复用同一个 session。
session 断开时自动重连，避免频繁建连导致 Streamable HTTP 竞态报错。
"""

import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_MYSQL_SERVER_URL", "http://127.0.0.1:8100/mcp")
TOOL_CACHE_TTL = int(os.getenv("MCP_TOOL_CACHE_TTL", "30"))  # 缓存有效期（秒）

# ── 持久 session ─────────────────────────────────────────────
_session_lock = asyncio.Lock()
_exit_stack: AsyncExitStack | None = None
_persistent_session: Any = None


async def _ensure_session():
    """获取或创建持久 MCP session，断开时自动重连。"""
    from mcp import ClientSession

    global _exit_stack, _persistent_session

    if _persistent_session is not None:
        return _persistent_session

    async with _session_lock:
        if _persistent_session is not None:
            return _persistent_session

        streamable_http_client = _get_mcp_client()
        _exit_stack = AsyncExitStack()
        transports = await _exit_stack.enter_async_context(
            streamable_http_client(MCP_SERVER_URL)
        )
        read_stream, write_stream = transports[0], transports[1]
        _persistent_session = await _exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await _persistent_session.initialize()
        logger.debug("MCP 持久 session 已建立")
        return _persistent_session


async def _reset_session():
    """断开持久 session（下次调用 _ensure_session 时会自动重连）。"""
    global _exit_stack, _persistent_session

    _persistent_session = None
    if _exit_stack:
        try:
            await _exit_stack.aclose()
        except Exception:
            pass
        _exit_stack = None


def _get_mcp_client():
    """惰性导入 streamable_http_client（避免模块加载时触发网络连接）"""
    from mcp.client.streamable_http import streamable_http_client

    return streamable_http_client


# ── 缓存状态 ─────────────────────────────────────────────────
_cached_tools: list = []
_cache_timestamp: float = 0
_cache_tool_names: list[str] = []


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


async def _discover_tools_from_server() -> list:
    """通过持久 session 列出工具，包装为 LangChain StructuredTool。"""
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    session = await _ensure_session()
    list_result = await session.list_tools()
    tools: list = []

    for tool_def in list_result.tools:
        tool_name = tool_def.name
        tool_description = tool_def.description or ""

        input_schema = tool_def.input_schema or {}
        raw_properties = input_schema.get("properties", {})
        properties = raw_properties or {}
        raw_required = input_schema.get("required", [])
        required = raw_required or []

        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required
            field_kwargs: dict[str, Any] = {"default": ... if is_required else None}
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
                msg = f"[MySQL MCP] 调用工具: {name}"
                if arguments:
                    msg += f"({args_str})"
                logger.debug(f"\033[94m{msg}\033[0m")

                try:
                    sess = await _ensure_session()
                except Exception:
                    logger.debug("MCP session 获取失败，尝试重连")
                    await _reset_session()
                    sess = await _ensure_session()

                try:
                    result = await sess.call_tool(name, arguments=arguments)
                except Exception:
                    logger.debug("MCP call_tool 失败，重置 session 后重试")
                    await _reset_session()
                    sess = await _ensure_session()
                    result = await sess.call_tool(name, arguments=arguments)

                texts = []
                content_items = getattr(result, "content", None) or []
                for c in content_items:
                    if hasattr(c, "text") and c.text:
                        texts.append(c.text)
                body = "\n".join(texts) if texts else str(result)

                is_err = getattr(result, "is_error", False) or getattr(result, "isError", False)
                if is_err:
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

    return tools


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
        logger.debug(f"MCP 工具刷新失败（使用缓存兜底）: {e}")
        if _cached_tools:
            return _cached_tools
        raise


async def refresh_mcp_tools_async() -> tuple[list, bool]:
    """后台刷新 MCP 工具列表，失败时重置 session 确保下次可重连。返回 (tools, changed)。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names

    old_names = list(_cache_tool_names)

    try:
        new_tools = await _discover_tools_from_server()
    except Exception as e:
        logger.debug(f"MCP 工具后台刷新失败（使用缓存兜底）: {e}")
        await _reset_session()
        return _cached_tools, False

    new_names = [t.name for t in new_tools]

    if new_names == old_names and _cached_tools:
        _cache_timestamp = time.time()
        return _cached_tools, False

    _cached_tools = new_tools
    _cache_timestamp = time.time()
    _cache_tool_names = new_names
    logger.info(f"MCP 工具列表已更新: {new_names}")
    return _cached_tools, True


async def close_mcp_cache():
    """清空缓存并断开持久 session。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names
    _cached_tools = []
    _cache_timestamp = 0
    _cache_tool_names = []
    await _reset_session()

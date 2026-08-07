"""
MCP 客户端工具加载模块

通过 Streamable HTTP transport 连接独立部署的 MCP Server，列出工具并包装为 LangChain StructuredTool。
支持 TTL 缓存 + 后台定时刷新，实现工具热插拔。
"""

import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_MYSQL_SERVER_URL", "http://127.0.0.1:8100/mcp")
TOOL_CACHE_TTL = int(os.getenv("MCP_TOOL_CACHE_TTL", "30"))  # 缓存有效期（秒）

# ── 缓存状态 ──────────────────────────────────────────────────
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


async def _discover_tools_from_server() -> list:
    """连接 MCP Server 并通过 list_tools() 发现工具，包装为 LangChain StructuredTool。"""
    from langchain_core.tools import StructuredTool
    from mcp import ClientSession
    from pydantic import Field, create_model

    exit_stack = AsyncExitStack()
    try:
        streamable_http_client = _get_mcp_client()
        transports = await exit_stack.enter_async_context(
            streamable_http_client(MCP_SERVER_URL)
        )
        read_stream, write_stream = transports
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
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
            args_model = create_model(f"{tool_name}_args", **fields)

            def _make_caller(name: str, sess: ClientSession):
                async def _call_mcp_tool(**kwargs):
                    arguments = {k: v for k, v in kwargs.items() if v is not None}
                    args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
                    msg = (
                        f"[MySQL MCP] 调用工具: {name}({args_str})"
                        if arguments
                        else f"[MySQL MCP] 调用工具: {name}"
                    )
                    logger.debug(f"\033[94m{msg}\033[0m")
                    result = await sess.call_tool(name, arguments=arguments)
                    texts = []
                    for c in getattr(result, "content", None) or []:
                        if hasattr(c, "text") and c.text:
                            texts.append(c.text)
                    body = "\n".join(texts) if texts else str(result)
                    if getattr(result, "is_error", False) or getattr(result, "isError", False):
                        return f"MCP 工具执行失败: {body}"
                    return body

                return _call_mcp_tool

            tool = StructuredTool.from_function(
                name=tool_name,
                description=tool_description,
                coroutine=_make_caller(tool_name, session),
                args_schema=args_model,
            )
            tools.append(tool)

        # 将 exit_stack 挂到工具列表上，调用方负责在合适的时机清理
        tools._exit_stack = exit_stack  # type: ignore[attr-defined]
        return tools

    except Exception:
        await exit_stack.aclose()
        raise


def _is_cache_valid() -> bool:
    return bool(_cached_tools) and (time.time() - _cache_timestamp) < TOOL_CACHE_TTL


def _tools_changed(new_names: list[str]) -> bool:
    return new_names != _cache_tool_names


# ── 公开 API ──────────────────────────────────────────────────


async def get_mcp_tools(force_refresh: bool = False) -> list:
    """获取 MCP 工具列表，带 TTL 缓存。

    默认返回缓存（30s TTL），缓存过期或 force_refresh=True 时重新连接 MCP Server 获取。
    """
    global _cached_tools, _cache_timestamp, _cache_tool_names

    if not force_refresh and _is_cache_valid():
        return _cached_tools

    try:
        new_tools = await _discover_tools_from_server()
        new_names = [t.name for t in new_tools]

        if not force_refresh and new_names == _cache_tool_names:
            # 工具列表未变，仅刷新时间戳，复用旧工具实例（保持 session 有效）
            _cache_timestamp = time.time()
            return _cached_tools

        # 工具列表有变化，关闭旧连接，替换为新工具
        await _close_old_cache()
        _cached_tools = new_tools
        _cache_timestamp = time.time()
        _cache_tool_names = new_names
        return _cached_tools

    except Exception as e:
        logger.error(f"MCP 工具刷新失败: {e}")
        # 有旧缓存则返回旧缓存（降级）
        if _cached_tools:
            return _cached_tools
        raise


async def refresh_mcp_tools_async() -> tuple[list, bool]:
    """强制刷新 MCP 工具列表，返回 (tools, changed)。

    在后台定时任务中调用，changed=True 表示工具列表有变化，需要重建 agent。
    """
    global _cached_tools, _cache_timestamp, _cache_tool_names

    old_names = list(_cache_tool_names)

    try:
        new_tools = await _discover_tools_from_server()
        new_names = [t.name for t in new_tools]

        if new_names == old_names:
            _cache_timestamp = time.time()
            return _cached_tools, False

        await _close_old_cache()
        _cached_tools = new_tools
        _cache_timestamp = time.time()
        _cache_tool_names = new_names
        return _cached_tools, True

    except Exception as e:
        logger.error(f"MCP 工具后台刷新失败: {e}")
        return _cached_tools, False


async def _close_old_cache():
    """关闭旧缓存的 MCP 连接"""
    if _cached_tools and hasattr(_cached_tools, "_exit_stack"):
        try:
            await _cached_tools._exit_stack.aclose()
        except Exception:
            pass


async def load_mcp_tools_async() -> tuple[list, AsyncExitStack]:
    """首次加载 MCP 工具（兼容旧接口）。

    返回 (tools, exit_stack)，exit_stack 需保持存活直到不再使用工具。
    注意：使用新缓存机制后，exit_stack 由缓存内部管理。
    """
    tools = await get_mcp_tools(force_refresh=True)
    # 返回一个空的 exit_stack 以兼容旧调用方
    return tools, AsyncExitStack()


async def close_mcp_cache():
    """关闭 MCP 缓存连接，通常在服务关闭时调用。"""
    global _cached_tools, _cache_timestamp, _cache_tool_names
    await _close_old_cache()
    _cached_tools = []
    _cache_timestamp = 0
    _cache_tool_names = []

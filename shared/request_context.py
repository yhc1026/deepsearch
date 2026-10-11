"""
请求元数据上下文管理模块

负责在 A2A 服务链路中保存本次请求的元数据（如 user_id、source_session_id 等），
由 A2A 执行器在 agent 执行前从 request metadata 注入，深层工具可读取，
避免让 LLM 转写这些隔离/溯源字段（尤其是 user_id）。

ContextVar 是协程级上下文变量，可避免多个并发请求串台。
"""

from contextvars import ContextVar, Token
from typing import Any, Optional

_request_meta_ctx: ContextVar[dict[str, Any]] = ContextVar(
    "request_metadata",
    default={},
)


def set_request_metadata(meta: dict[str, Any]) -> Token:
    """设置当前请求链路的元数据。

    :param meta: 请求元数据字典（如 {"user_id": "3", ...}）
    :return: reset 时使用的上下文 token
    """
    return _request_meta_ctx.set(meta or {})


def get_request_metadata() -> dict[str, Any]:
    """获取当前请求链路的元数据。

    :return: 请求元数据字典；未设置时返回空 dict
    """
    return _request_meta_ctx.get()


def reset_request_metadata(token: Token) -> None:
    """恢复请求元数据上下文，避免残留到后续请求。"""
    _request_meta_ctx.reset(token)

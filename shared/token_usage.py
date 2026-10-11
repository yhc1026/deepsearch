"""
单次请求 token 用量累计模块

通过 ContextVar 按协程隔离，避免多个并发请求串台（与 request_context 同理）。
主智能体与各子智能体在各自进程内调用 add_usage 累计，
子智能体结束时把累计结果经 A2A DataPart 回传，由 orchestrator 汇总。

usage 统一为 {"input_tokens", "output_tokens", "total_tokens"}。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_usage_ctx: ContextVar[dict[str, int] | None] = ContextVar("token_usage", default=None)

_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _empty() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def begin_request() -> None:
    """在请求入口初始化一个空累计器。"""
    _usage_ctx.set(_empty())


def add_usage(usage: Any) -> None:
    """累加一次 LLM 调用的 usage。

    usage 可以是 langchain 的 UsageMetadata（本质是 dict）或普通 dict；
    为空 / 字段缺失时安全跳过。
    """
    if not usage:
        return

    acc = _usage_ctx.get()
    if acc is None:
        acc = _empty()
        _usage_ctx.set(acc)

    if isinstance(usage, dict):
        for key in _KEYS:
            acc[key] += int(usage.get(key, 0) or 0)
    else:
        for key in _KEYS:
            acc[key] += int(getattr(usage, key, 0) or 0)


def get_usage() -> dict[str, int]:
    """返回当前请求累计用量（副本）。"""
    acc = _usage_ctx.get()
    if acc is None:
        return _empty()
    return dict(acc)


def reset_request() -> None:
    """清空当前请求累计器，避免残留到后续请求。"""
    _usage_ctx.set(None)

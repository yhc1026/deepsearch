"""
长期记忆客户端

两个能力：
1. 提取：orchestrator 每轮任务结束后 fire-and-forget 调用长期记忆 Agent，
   把本轮对话中的事实性内容写入用户全局长期记忆（异步，不阻塞本轮响应）。
2. 召回：orchestrator 每轮任务开始前同步调用长期记忆 Agent，检索该用户
   与当前问题相关的旧记忆，供主智能体注入 prompt（同步，阻塞直至返回）。
"""

import asyncio
import logging

from agents.orchestrator.a2a_tools import call_memory_agent
from shared.agent_result import ERROR, parse_result

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 4000


def _build_memory_message(
    user_id: int,
    session_id: str,
    turn_index: int,
    task_query: str,
    last_result: str,
) -> str:
    result = (last_result or "")[:_RESULT_MAX_CHARS]
    return (
        "[记忆提取请求]\n"
        f"user_id: {user_id}\n"
        f"source_session_id: {session_id}\n"
        f"source_turn_index: {turn_index}\n"
        f"source_query: {task_query}\n"
        f"assistant_result:\n{result}\n\n"
        "请从上述对话中提取事实性内容，检索冲突后写入长期记忆。"
    )


def _build_recall_message(user_id: int, session_id: str, task_query: str) -> str:
    return (
        "[记忆召回请求]\n"
        f"user_id: {user_id}\n"
        f"source_session_id: {session_id}\n"
        f"query: {task_query}\n\n"
        "请检索该用户与上述问题相关的长期记忆，并原样返回检索结果。"
    )


async def _extract_memories(
    user_id: int,
    session_id: str,
    turn_index: int,
    task_query: str,
    last_result: str,
) -> None:
    try:
        message = _build_memory_message(
            user_id, session_id, turn_index, task_query, last_result
        )
        reply = await call_memory_agent(message)
        code, content = parse_result(reply)
        logger.info(
            f"[memory] session={session_id[:8]} turn={turn_index} -> {code}: {content[:200]}"
        )
    except Exception:  # noqa: BLE001
        logger.warning("长期记忆提取失败", exc_info=True)


def schedule_memory_extraction(
    user_id: int | None,
    session_id: str,
    turn_index: int,
    task_query: str,
    last_result: str,
) -> None:
    """调度一次长期记忆提取（fire-and-forget）。无用户时直接跳过。"""
    if user_id is None:
        return
    asyncio.create_task(
        _extract_memories(user_id, session_id, turn_index, task_query, last_result)
    )


async def recall_memories(
    user_id: int | None,
    session_id: str,
    task_query: str,
) -> tuple[str, str]:
    """同步召回该用户与当前问题相关的长期记忆，返回 (code, content)。

    命中（HIT）时 content 为检索到的记忆正文；未命中（MISS）或异常（ERROR）时
    无需注入 prompt。无用户时直接返回 MISS。
    """
    if user_id is None:
        return ERROR, ""
    try:
        message = _build_recall_message(user_id, session_id, task_query)
        reply = await call_memory_agent(message)
        code, content = parse_result(reply)
        logger.info(
            f"[memory-recall] session={session_id[:8]} -> {code}: {content[:200]}"
        )
        return code, content
    except Exception:  # noqa: BLE001
        logger.warning("长期记忆召回失败", exc_info=True)
        return ERROR, ""

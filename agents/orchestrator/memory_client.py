"""
长期记忆客户端

两个能力：
1. 提取：orchestrator 每轮任务结束后 fire-and-forget 调用长期记忆 Agent，
   把本轮对话中的事实性内容写入用户全局长期记忆（异步，不阻塞本轮响应）。
   user_id 及溯源字段经 A2A request metadata 传递，由服务端注入工具，不经 LLM 转写。
2. 召回：orchestrator 每轮任务开始前确定性调用本地召回管线（dense + rerank + 阈值
   + 时间衰减 + 来源多样性），检索该用户与当前问题相关的旧记忆，供主智能体注入
   prompt（同步，阻塞直至返回）。召回不经 A2A / LLM。
"""

import asyncio
import logging

from agents.memory_agent.recall import recall_memory
from agents.orchestrator.a2a_tools import call_memory_agent
from shared.agent_result import ERROR, parse_result

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 4000


def _build_memory_message(task_query: str, last_result: str) -> str:
    """构造记忆提取请求文本。只携带 LLM 提取所需内容，不含 user_id / 溯源字段。"""
    result = (last_result or "")[:_RESULT_MAX_CHARS]
    return (
        "[记忆提取请求]\n"
        f"source_query: {task_query}\n"
        f"assistant_result:\n{result}\n\n"
        "请从上述对话中提取事实性内容，检索冲突后写入长期记忆。"
    )


async def _extract_memories(
    user_id: int,
    session_id: str,
    turn_index: int,
    task_query: str,
    last_result: str,
) -> None:
    try:
        message = _build_memory_message(task_query, last_result)
        metadata = {
            "user_id": str(user_id),
            "source_session_id": session_id,
            "source_turn_index": str(turn_index),
            "source_query": task_query,
        }
        reply = await call_memory_agent(message, metadata=metadata)
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
    """确定性召回该用户与当前问题相关的长期记忆，返回 (code, content)。

    命中（HIT）时 content 为检索到的记忆正文；未命中（MISS）或异常（ERROR）时
    无需注入 prompt。无用户时直接返回 ERROR。
    """
    if user_id is None:
        return ERROR, ""
    try:
        # 召回含阻塞式 embedding / rerank 调用，放到线程池避免阻塞事件循环
        code, content = await asyncio.to_thread(recall_memory, user_id, task_query)
        logger.info(
            f"[memory-recall] session={session_id[:8]} -> {code}: {content[:200]}"
        )
        return code, content
    except Exception:  # noqa: BLE001
        logger.warning("长期记忆召回失败", exc_info=True)
        return ERROR, ""
